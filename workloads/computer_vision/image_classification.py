"""Image classification: 60 timm models from 1.5M to 303M params, covering CNNs (resnet,
efficientnet, regnety, convnext), transformers (vit, deit, swin, xcit, pvt) and hybrids
(coatnet, maxvit, mobilevit), plus MLP-only (mixer, resmlp). Each is swept over img_size
64/128/224 x batch 16/32/64 x fp32/fp16/bf16 = 1620 configs, each timed for one SGD training
step inside a captured CUDA graph, so the measurement is GPU kernel time with no host dispatch
in it. Inference is parked in inference.py.

Five models were dropped for cost rather than coverage: on a 5090 they were 59% of the sweep's
measured time, and vit_huge_patch14_224 alone was 33% while sitting last in the list -- the
worst possible position for a preemptible container. caformer_b36 is the only one whose
architecture family goes with it; the rest leave smaller siblings behind (maxvit_tiny,
efficientnet_b0/b3/b5, swin_tiny/base)."""

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row, is_oom, progress_line
from profiler.profiler import time_fn


# (warmup, repeats, iters) handed to time_fn. The step is measured inside a captured CUDA graph,
# so no Python dispatch lands in the timed window. That matters here more than anywhere else in
# the suite: timed eagerly, a small model at a small img_size spends up to 88% of its wall time
# waiting on the host, which buries the img_size signal entirely and varies with the node's CPU.
TRAIN_TIMING = (3, 10, 2)
TIMING_METHOD = "cuda_graph"

# fp32 runs without autocast; the other two run under it. No GradScaler on any of them: its
# inf-check ends in a .item() host sync, which cannot occur inside a graph capture.
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class Hyperparams:
    model: str = "resnet18"     # timm model name
    img_size: int = 224         # square image side length in pixels
    batch_size: int = 64
    precision: str = "fp16"     # fp32 | fp16 | bf16


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config (train). Returns one or more rows, one per recorded phase."""
    rows: list[dict] = []
    device = torch.device("cuda")
    img_size = hyperparams.img_size
    monitor = CUDAMonitor(interval_ms=10)

    try:
        try:
            model = timm.create_model(hyperparams.model, pretrained=False, num_classes=16, img_size=img_size)
        except TypeError:
            model = timm.create_model(hyperparams.model, pretrained=False, num_classes=16)
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}",
                              timing_method=TIMING_METHOD))
        return rows
    model = model.to(device).train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    amp_dtype = AMP_DTYPES.get(hyperparams.precision)

    # One fixed batch rather than a fresh one per step: a captured graph replays the recorded
    # kernels against the addresses they were captured with, so the input cannot move. The
    # values are irrelevant -- nothing here branches on data, only on shape and dtype.
    x = torch.randn(hyperparams.batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, 16, (hyperparams.batch_size,), device=device)

    def step():
        # set_to_none=False keeps the gradient buffers at fixed addresses across replays; the
        # default frees them, which would move allocations under the graph.
        optimizer.zero_grad(set_to_none=False)
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = criterion(model(x), y)
        else:
            loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()

    monitor.start()
    train_avg_ms, err, exc = None, "", None
    try:
        step()      # materialise gradients and the momentum buffer before the capture
        torch.cuda.synchronize()
        train_avg_ms = time_fn(step, *TRAIN_TIMING)
    except RuntimeError as e:
        err, exc = f"train_failed: {e}", e   # provisional; reclassified below if it was OOM
    finally:
        metrics = monitor.stop()
        # is_oom covers all three routes an out-of-VRAM config arrives by; see cuda_monitor.is_oom.
        if exc is not None and is_oom(exc, metrics.get("max_memory_used_pct")):
            metrics["oom"], err = True, ""
        rows.append(build_row(hyperparams, "train", metrics=metrics,
                              error=err, avg_ms=train_avg_ms, timing_method=TIMING_METHOD))
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass    # already-recorded row above must not be lost to a second, unrelated failure

    del model, optimizer, x, y
    return rows


MODELS = [
    # Deliberately NOT smallest-first. These eight produced the most memory-saturated rows on
    # the first three nodes -- 10 to 16 configs each at 100% of VRAM, where the allocator used
    # to grind instead of raising and reported times 30-66x above trend. With the memory cap in
    # main.configure_runtime they must now come back as oom rows in seconds. Running them first
    # means a node shows within a few minutes whether that holds, rather than after an hour of
    # small models. Note maxvit_tiny is here despite the name: it saturated 10 configs.
    'coatnet_2_rw_224', 'efficientnetv2_l', 'beit_large_patch16_224', 'convnext_large',
    'convnextv2_base', 'vit_large_patch16_224', 'maxvit_tiny_tf_224', 'pvt_v2_b5',
    # the rest, smallest first
    'coatnet_0_rw_224', 'coatnet_1_rw_224', 'convnext_small', 'convnext_tiny',
    'deit_small_patch16_224', 'deit_tiny_patch16_224', 'densenet121',
    'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5',
    'efficientnetv2_m', 'efficientnetv2_s',
    'fastvit_t8', 'fastvit_t12', 'mixer_b16_224',
    'mobilenetv2_100', 'mobilenetv3_large_100', 'mobilenetv3_small_100',
    'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'mobilevitv2_100',
    'nfnet_l0', 'pvt_v2_b0', 'pvt_v2_b2',
    'regnety_032', 'regnety_080', 'repvgg_a2', 'resmlp_24_224',
    'resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
    'swin_tiny_patch4_window7_224', 'swinv2_cr_tiny_ns_224',
    'twins_pcpvt_small', 'twins_svt_small',
    'vit_tiny_patch16_224', 'vit_small_patch16_224',
    'xcit_tiny_12_p16_224', 'xcit_small_12_p16_224',
    # ~60-100M params
    'resnet200', 'resnext101_64x4d',
    'regnety_160', 'vit_base_patch16_224', 'deit_base_patch16_224',
    'swin_base_patch4_window7_224', 'convnext_base',
    # ~115-200M params
    'wide_resnet101_2', 'dm_nfnet_f1', 'regnety_320',
]
IMG_SIZES = [64, 128, 224]
BATCH_SIZES = [16, 32, 64]
PRECISIONS = ['fp32', 'fp16', 'bf16']



CHECKPOINT_EVERY = 5   # models between partial uploads


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
        prev = progress_line(model, i, len(selected), "image classification", prev)
        for img_size, batch_size, precision in product(IMG_SIZES, BATCH_SIZES, PRECISIONS):
            params = {"model": model, "img_size": img_size, "batch_size": batch_size, "precision": precision}
            try:
                all_rows.extend(run(Hyperparams(**params)))
            except Exception as e:
                # Unexpected failure that escaped run() (CUDA hang, OOM in unprotected alloc, etc.)
                print(f"FAIL ({type(e).__name__}): {params} - {e}")
                all_rows.append({**params, "phase": "config_failed", "error": f"{type(e).__name__}: {e}"})
            finally:
                # A broken context here would otherwise escape this finally clause and take
                # every remaining model in this workload down with it -- and main.py has no
                # guard around run_all() at all, so the whole rest of the node's sweep too.
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        if checkpoint_fn is not None and i % CHECKPOINT_EVERY == 0:
            try:
                checkpoint_fn(pd.DataFrame(all_rows))
            except Exception as e:
                print(f"  checkpoint after {model} failed: {type(e).__name__}: {e}", flush=True)

    return pd.DataFrame(all_rows)
