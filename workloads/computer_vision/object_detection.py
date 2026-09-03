"""Object detection: 12 detectors from 19M to 128M params, covering anchor-based two-stage
(Faster R-CNN), anchor-based one-stage (RetinaNet, SSD-style FPN heads), anchor-free (FCOS) and
set-prediction transformers (DETR, Conditional DETR, YOLOS). Swept over img_size 320/512/800 x
batch 2/4/8 x fp32/fp16/bf16 = 324 configs, each timed for one SGD training step. Detection
resolutions are much larger than image_classification's because below ~320px the anchor and
assignment Python dominates. This is the one workload that cannot be CUDA-graph captured, so it
reports GPU kernel time via the profiler (kernel_time_fn) rather than via capture (time_fn)."""

import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
from torchvision.models import detection as tvdet

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row, is_oom, progress_line
from profiler.profiler import kernel_time_fn

import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import AutoConfig, AutoModelForObjectDetection


# (warmup, repeats, iters) handed to kernel_time_fn. Unlike the other three workloads this one
# cannot be CUDA-graph captured -- torchvision's GeneralizedRCNNTransform and the HF detection
# loss both build host tensors inside the forward -- so GPU kernel time is read off the profiler
# instead. Eager wall-clock was 5% host at 800px/bs8 but 62% at 320px/bs2, so roughly a third of
# this grid would otherwise have been measuring the node's CPU.
# 5 repeats, not the 10 the graph-captured workloads use. A repeat is nearly free under CUDA
# graph replay but costs ~1.5s here, because reading any result off the profiler forces it to
# post-process the raw trace -- so repeats, not the steps, set this workload's runtime. The
# measurement does not need them: over ten repeats of fasterrcnn_resnet50_fpn 800px bs2 the
# spread was 121.69-121.97ms plus one 127.23ms outlier, and min-of-3 returned 121.69, identical
# to min-of-10. retinanet 320px bs2 agreed to 0.04%. 5 keeps margin for rejecting interference
# on a shared cloud node while halving the cost; with the raw-event change above it takes a
# config from ~24s to ~9s, and the 324-config sweep from ~3h to ~1h per GPU.
TRAIN_TIMING = (3, 5, 2)
TIMING_METHOD = "kernel_sum"   # the only workload not measured by CUDA graph capture

NUM_CLASSES = 16
BOXES_PER_IMAGE = 8

# fp32 runs without autocast; the other two run under it. No GradScaler on any of them: the
# other workloads dropped it for capture, and its unscale/inf-check kernels would otherwise
# show up in this workload's kernel time and nowhere else.
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class Hyperparams:
    model: str = "fasterrcnn_resnet50_fpn"   # torchvision builder name, or a HF repo id ("org/name")
    img_size: int = 800                      # square image side length in pixels
    batch_size: int = 4
    precision: str = "fp16"                  # fp32 | fp16 | bf16


def _torchvision_targets(batch_size: int, img_size: int, device) -> list[dict]:
    """Random boxes in absolute (x1, y1, x2, y2) pixels, which is what torchvision expects.
    Built so x2 > x1 and y2 > y1 always: degenerate boxes make torchvision raise."""
    targets = []
    for _ in range(batch_size):
        xy = torch.rand(BOXES_PER_IMAGE, 2, device=device) * img_size * 0.5
        wh = torch.rand(BOXES_PER_IMAGE, 2, device=device) * img_size * 0.4 + 8
        targets.append({
            "boxes": torch.cat([xy, (xy + wh).clamp(max=img_size)], dim=1),
            "labels": torch.randint(1, NUM_CLASSES, (BOXES_PER_IMAGE,), device=device),
        })
    return targets


def _hf_targets(batch_size: int, device) -> list[dict]:
    """Random boxes in normalised (cx, cy, w, h), which is what the DETR family expects."""
    return [{
        "class_labels": torch.randint(0, NUM_CLASSES, (BOXES_PER_IMAGE,), device=device),
        "boxes": torch.rand(BOXES_PER_IMAGE, 4, device=device).clamp(0.05, 0.9),
    } for _ in range(batch_size)]


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config (train). Returns one or more rows, one per recorded phase."""
    rows: list[dict] = []
    device = torch.device("cuda")
    img_size = hyperparams.img_size
    monitor = CUDAMonitor(interval_ms=10)
    is_hf = "/" in hyperparams.model

    try:
        if is_hf:
            # Unlike the other HF workloads this does NOT use no_init_weights: an uninitialised
            # detector emits NaN boxes, and the detection loss validates box geometry before the
            # Hungarian match, so the step raises instead of just timing garbage. Random init is
            # still checkpoint-free, so nothing is downloaded beyond config.json.
            cfg = AutoConfig.from_pretrained(hyperparams.model, num_labels=NUM_CLASSES)
            model = AutoModelForObjectDetection.from_config(cfg)
        else:
            # min_size/max_size pinned to img_size, otherwise GeneralizedRCNNTransform rescales
            # every input to its 800/1333 default and the img_size axis measures nothing.
            model = getattr(tvdet, hyperparams.model)(
                weights=None, weights_backbone=None, num_classes=NUM_CLASSES,
                min_size=img_size, max_size=img_size,
            )
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}",
                              timing_method=TIMING_METHOD))
        return rows
    model = model.to(device).train()

    # lr=0 because we measure kernel time and never inspect convergence. An untrained detector
    # at a real learning rate diverges within ~2 steps, and the set-prediction losses validate
    # their own box geometry before the Hungarian match, so NaN predictions raise rather than
    # merely timing garbage -- conditional-detr fails this way at lr=0.01. SGD still issues an
    # identical kernel sequence at lr=0 (param.add_(d_p, alpha=-0) is not short-circuited), so
    # the measurement is unchanged.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0, momentum=0.9)
    amp_dtype = AMP_DTYPES.get(hyperparams.precision)

    if is_hf:
        inputs = torch.rand(hyperparams.batch_size, 3, img_size, img_size, device=device)
        targets = _hf_targets(hyperparams.batch_size, device)
    else:
        inputs = [torch.rand(3, img_size, img_size, device=device) for _ in range(hyperparams.batch_size)]
        targets = _torchvision_targets(hyperparams.batch_size, img_size, device)

    def forward_loss():
        # Both families return their losses already reduced; torchvision hands back a dict of
        # head losses that the caller is expected to sum.
        if is_hf:
            return model(pixel_values=inputs, labels=targets).loss
        return sum(model(inputs, targets).values())

    def step():
        optimizer.zero_grad()
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = forward_loss()
        else:
            loss = forward_loss()
        loss.backward()
        optimizer.step()

    monitor.start()
    train_avg_ms, kernels, err, exc = None, None, "", None
    try:
        train_avg_ms, kernels = kernel_time_fn(step, *TRAIN_TIMING)
    except Exception as e:
        # Broader than the other workloads on purpose: the set-prediction losses raise
        # ValueError (not RuntimeError) when a prediction goes non-finite, which would
        # otherwise escape run() and lose the monitor metrics for this row.
        err, exc = f"train_failed: {type(e).__name__}: {e}", e
    finally:
        metrics = monitor.stop()
        # is_oom covers all three routes an out-of-VRAM config arrives by (a plain ValueError
        # never matches, memory pressure or not -- see cuda_monitor.is_oom).
        if exc is not None and is_oom(exc, metrics.get("max_memory_used_pct")):
            metrics["oom"], err = True, ""
        rows.append(build_row(hyperparams, "train", metrics=metrics,
                              error=err, avg_ms=train_avg_ms,
                              timing_method=TIMING_METHOD, kernel_count=kernels))
    torch.cuda.empty_cache()

    del model, optimizer, inputs, targets
    return rows


# torchvision builders take the raw name; HF detectors are "org/name" and route through the
# transformers branch above. ssd300_vgg16 and ssdlite320_mobilenet_v3_large are deliberately
# absent: their transforms hardcode 300/320, so all three img_size values would produce the
# same measurement under three different labels.
MODELS = [
    # anchor-based two-stage (Faster R-CNN)
    'fasterrcnn_mobilenet_v3_large_320_fpn', 'fasterrcnn_mobilenet_v3_large_fpn',
    'fasterrcnn_resnet50_fpn', 'fasterrcnn_resnet50_fpn_v2',
    # anchor-based one-stage
    'retinanet_resnet50_fpn', 'retinanet_resnet50_fpn_v2',
    # anchor-free one-stage
    'fcos_resnet50_fpn',
    # set-prediction transformers (bipartite matching loss, no anchors or NMS)
    'facebook/detr-resnet-50', 'facebook/detr-resnet-101',
    'microsoft/conditional-detr-resnet-50',
    'hustvl/yolos-small', 'hustvl/yolos-base',
]
IMG_SIZES = [320, 512, 800]
BATCH_SIZES = [2, 4, 8]
PRECISIONS = ['fp32', 'fp16', 'bf16']




CHECKPOINT_EVERY = 3   # models between partial uploads


def run_all(only_models: Optional[set] = None, skip_models: Optional[set] = None,
            checkpoint_fn: Optional[Callable[[pd.DataFrame], None]] = None) -> pd.DataFrame:
    """Iterate every config. Returns one row per recorded phase; main.py uploads the result.

    skip_models are already covered by a checkpoint from this same physical GPU, so they are
    not re-run. checkpoint_fn is called with the rows so far every CHECKPOINT_EVERY models:
    Salad containers reset at arbitrary points, and without it a node that runs for hours and
    is preempted near the end contributes nothing at all."""
    all_rows: list[dict] = []
    selected = [m for m in MODELS
                if (only_models is None or m in only_models)
                and (skip_models is None or m not in skip_models)]
    prev = None
    for i, model in enumerate(selected, 1):
        prev = progress_line(model, i, len(selected), "object detection", prev)
        for img_size, batch_size, precision in product(IMG_SIZES, BATCH_SIZES, PRECISIONS):
            params = {"model": model, "img_size": img_size, "batch_size": batch_size, "precision": precision}
            try:
                all_rows.extend(run(Hyperparams(**params)))
            except Exception as e:
                # Unexpected failure that escaped run() (CUDA hang, OOM in unprotected alloc, etc.)
                print(f"FAIL ({type(e).__name__}): {params} - {e}")
                all_rows.append({**params, "phase": "config_failed", "error": f"{type(e).__name__}: {e}"})
            finally:
                torch.cuda.empty_cache()

        if checkpoint_fn is not None and i % CHECKPOINT_EVERY == 0:
            try:
                checkpoint_fn(pd.DataFrame(all_rows))
            except Exception as e:
                print(f"  checkpoint after {model} failed: {type(e).__name__}: {e}", flush=True)

    return pd.DataFrame(all_rows)
