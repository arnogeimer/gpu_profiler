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
from profiler.cuda_monitor import CUDAMonitor, build_row


WARMUP_BATCHES = 5
TIMED_BATCHES = 20


@dataclass
class Hyperparams:
    model: str = "resnet18"     # timm model name
    img_size: int = 224         # square image side length in pixels
    batch_size: int = 64
    precision: str = "fp16"     # fp32 | fp16


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config (train + infer). Returns one or more rows, one per recorded phase."""
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
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}"))
        return rows
    model = model.to(device).train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    use_fp16 = hyperparams.precision == "fp16"
    # GradScaler prevents underflow in fp16 gradients — kept across steps so its
    # adaptive scale factor can converge. See https://pytorch.org/docs/stable/notes/amp_examples.html
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    def make_batch():
        x = torch.randn(hyperparams.batch_size, 3, img_size, img_size, device=device)
        y = torch.randint(0, 16, (hyperparams.batch_size,), device=device)
        return x, y

    def step(x, y):
        optimizer.zero_grad()
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    # Warmup block
    try:
        for _ in range(WARMUP_BATCHES):
            step(*make_batch())
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, "train", error=f"forward_failed: {e}"))
        return rows

    monitor.start()
    times_ms = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            x, y = make_batch()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step(x, y)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True   # record below, then continue to inference (it doesn't need autograd memory)
    finally:
        train_metrics = monitor.stop(oom=oom)
        train_avg_ms = sum(times_ms) / len(times_ms) if times_ms else None
        rows.append(build_row(hyperparams, "train", metrics=train_metrics, avg_ms=train_avg_ms))
    torch.cuda.empty_cache()

    model.eval()
    x_inf = torch.randn(128, 3, img_size, img_size, device=device)

    def infer(x):
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    return model(x).argmax(dim=1)
            else:
                return model(x).argmax(dim=1)

    # Warmup block
    try:
        for _ in range(WARMUP_BATCHES):
            infer(x_inf)
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, "infer", error=f"forward_failed: {e}"))
        return rows

    monitor.start()
    inf_times_ms = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            infer(x_inf)
            end.record()
            torch.cuda.synchronize()
            inf_times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        infer_metrics = monitor.stop(oom=oom)
        inf_avg_ms = sum(inf_times_ms) / len(inf_times_ms) if inf_times_ms else None
        rows.append(build_row(hyperparams, "infer", metrics=infer_metrics, avg_ms=inf_avg_ms))

    del model, optimizer, x_inf
    def _fmt(avg_ms, batch):
        return "OOM" if avg_ms is None else f"{avg_ms / batch:.3f} ms/sample"
    print(f"train: {_fmt(train_avg_ms, hyperparams.batch_size)} | inference: {_fmt(inf_avg_ms, 128)}  ({hyperparams.model} | {img_size}x{img_size} | bs={hyperparams.batch_size} | {hyperparams.precision})")
    return rows


MODELS = [
    'coatnet_0_rw_224', 'coatnet_1_rw_224', 'convnext_small', 'convnext_tiny',
    'deit_small_patch16_224', 'deit_tiny_patch16_224', 'densenet121',
    'efficientformer_l1', 'efficientformer_l3',
    'efficientnet_b0', 'efficientnet_b3', 'efficientnet_b5',
    'efficientnetv2_m', 'efficientnetv2_s',
    'fastvit_t8', 'fastvit_t12', 'maxvit_tiny_tf_224', 'mixer_b16_224',
    'mobilenetv2_100', 'mobilenetv3_large_100',
    'mobilevit_xxs', 'mobilevit_xs', 'mobilevit_s', 'mobilevitv2_100',
    'nfnet_l0', 'pvt_v2_b0', 'pvt_v2_b2',
    'regnety_032', 'regnety_080', 'repvgg_a2', 'resmlp_24_224',
    'resnet18', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d',
    'swin_tiny_patch4_window7_224', 'swinv2_cr_tiny_ns_224',
    'twins_pcpvt_small', 'twins_svt_small',
    'vit_tiny_patch16_224', 'vit_small_patch16_224',
    'xcit_tiny_12_p16_224', 'xcit_small_12_p16_224',
]
IMG_SIZES = [32, 128, 224]
BATCH_SIZES = [16, 32, 64]
PRECISIONS = ['fp32', 'fp16']


def run_all(upload_fn: Optional[Callable[[pd.DataFrame], None]] = None,
            only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config; after each model is finished, call upload_fn(current DataFrame)."""
    all_rows: list[dict] = []
    for model in MODELS:
        if only_models is not None and model not in only_models:
            continue
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

        if upload_fn:
            try:
                upload_fn(pd.DataFrame(all_rows))
            except Exception as e:
                print(f"  upload after {model} failed: {e}")

    return pd.DataFrame(all_rows)
