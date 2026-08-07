"""Image classification: 65 timm models from 1.5M to 630M params, covering CNNs (resnet,
efficientnet, regnety, convnext), transformers (vit, deit, swin, xcit, pvt) and hybrids
(coatnet, maxvit, mobilevit), plus MLP-only (mixer, resmlp). Each is swept over img_size
64/128/224 x batch 16/32/64 x fp32/fp16/bf16 = 1755 configs, each timed for one SGD training
step inside a captured CUDA graph, so the measurement is GPU kernel time with no host dispatch
in it. Inference is parked in inference.py."""

import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import torch.nn as nn
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row
from profiler.profiler import time_fn


# (warmup, repeats, iters) handed to time_fn. The step is measured inside a captured CUDA graph,
# so no Python dispatch lands in the timed window. That matters here more than anywhere else in
# the suite: timed eagerly, a small model at a small img_size spends up to 88% of its wall time
# waiting on the host, which buries the img_size signal entirely and varies with the node's CPU.
TRAIN_TIMING = (3, 5, 2)
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
    train_avg_ms, oom, err = None, False, ""
    try:
        step()      # materialise gradients and the momentum buffer before the capture
        torch.cuda.synchronize()
        train_avg_ms = time_fn(step, *TRAIN_TIMING)
    except torch.cuda.OutOfMemoryError:
        oom = True   # recorded below rather than raised, so the sweep keeps going
    except RuntimeError as e:
        err = f"train_failed: {e}"
    finally:
        rows.append(build_row(hyperparams, "train", metrics=monitor.stop(oom=oom),
                              error=err, avg_ms=train_avg_ms, timing_method=TIMING_METHOD))
    torch.cuda.empty_cache()

    del model, optimizer, x, y
    def _fmt(avg_ms, batch):
        return "OOM" if avg_ms is None else f"{avg_ms / batch:.3f} ms/sample"
    print(f"train: {_fmt(train_avg_ms, hyperparams.batch_size)}  ({hyperparams.model} | {img_size}x{img_size} | bs={hyperparams.batch_size} | {hyperparams.precision})")
    return rows


MODELS = [
    'coatnet_0_rw_224', 'coatnet_1_rw_224', 'convnext_small', 'convnext_tiny',
    'deit_small_patch16_224', 'deit_tiny_patch16_224', 'densenet121',
    'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5',
    'efficientnetv2_m', 'efficientnetv2_s',
    'fastvit_t8', 'fastvit_t12', 'maxvit_tiny_tf_224', 'mixer_b16_224',
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
    'resnet200', 'resnext101_64x4d', 'efficientnet_b7', 'coatnet_2_rw_224',
    'regnety_160', 'pvt_v2_b5', 'vit_base_patch16_224', 'deit_base_patch16_224',
    'swin_base_patch4_window7_224', 'convnext_base', 'convnextv2_base', 'caformer_b36',
    # ~115-200M params
    'efficientnetv2_l', 'maxvit_base_tf_224', 'wide_resnet101_2', 'dm_nfnet_f1',
    'regnety_320', 'swin_large_patch4_window7_224', 'convnext_large',
    # ~300M+ params
    'vit_large_patch16_224', 'beit_large_patch16_224', 'vit_huge_patch14_224',
]
IMG_SIZES = [64, 128, 224]
BATCH_SIZES = [16, 32, 64]
PRECISIONS = ['fp32', 'fp16', 'bf16']



def run_all(only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config. Returns one row per recorded phase; main.py uploads the result."""
    all_rows: list[dict] = []
    selected = [m for m in MODELS if only_models is None or m in only_models]
    for model in selected:
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

    return pd.DataFrame(all_rows)
