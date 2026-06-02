import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler import gpu_info
from profiler.cuda_monitor import CUDAMonitor, record_run


WARMUP_BATCHES = 5
TIMED_BATCHES = 20

_DATASET_CFG = {
    "32x32":   {"img_size": 32},
    "224x224": {"img_size": 224},
}

_PRECISION = {
    "fp16": torch.float16,
    "fp32": torch.float32,
}


@dataclass
class Hyperparams:
    model: str = "resnet18"    # timm model name
    dataset: str = "32x32"  # 32x32 | 128x128 | 224x224
    batch_size: int = 64
    precision: str = "fp16"    # fp32 | fp16 | bf16


def run(hyperparams: Hyperparams) -> dict:
    cfg = _DATASET_CFG[hyperparams.dataset]
    device = torch.device("cuda")

    csv_path = Path(__file__).parent / f"{gpu_info.get_gpu_name()}_metrics.csv"
    monitor = CUDAMonitor(interval_ms=20)

    try:
        try:
            model = timm.create_model(hyperparams.model, pretrained=False, num_classes=16, img_size=cfg["img_size"])
        except TypeError:
            model = timm.create_model(hyperparams.model, pretrained=False, num_classes=16)
    except Exception as e:
        record_run(csv_path, hyperparams, "setup", error=f"model_creation_failed: {e}")
        return {"error": "model_creation_failed", "message": str(e)}
    model = model.to(device).train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    amp_dtype = _PRECISION[hyperparams.precision]
    use_autocast = hyperparams.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda") if hyperparams.precision == "fp16" else None

    def make_batch():
        x = torch.randn(hyperparams.batch_size, 3, cfg["img_size"], cfg["img_size"], device=device)
        y = torch.randint(0, 16, (hyperparams.batch_size,), device=device)
        return x, y

    def step(x, y):
        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = criterion(model(x), y)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    try:
        for _ in range(WARMUP_BATCHES):
            step(*make_batch())
        torch.cuda.synchronize()
    except RuntimeError as e:
        record_run(csv_path, hyperparams, "train", error=f"forward_failed: {e}")
        return {"error": "forward_failed", "message": str(e)}

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
        oom = True
        raise
    finally:
        train_metrics = monitor.stop(oom=oom)
        train_avg_ms = sum(times_ms) / len(times_ms) if times_ms else None
        record_run(csv_path, hyperparams, "train", metrics=train_metrics, avg_ms=train_avg_ms)

    model.eval()
    x_inf = torch.randn(128, 3, cfg["img_size"], cfg["img_size"], device=device)

    def infer(x):
        with torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=amp_dtype):
                    return model(x).argmax(dim=1)
            else:
                return model(x).argmax(dim=1)

    for _ in range(WARMUP_BATCHES):
        infer(x_inf)
    torch.cuda.synchronize()

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
        raise
    finally:
        infer_metrics = monitor.stop(oom=oom)
        inf_avg_ms = sum(inf_times_ms) / len(inf_times_ms) if inf_times_ms else None
        record_run(csv_path, hyperparams, "infer", metrics=infer_metrics, avg_ms=inf_avg_ms)

    del model, optimizer, x_inf
    print(f"train: {train_avg_ms / hyperparams.batch_size:.3f} ms/sample | inference: {inf_avg_ms / 128:.3f} ms/sample  ({hyperparams.model} | {hyperparams.dataset} | bs={hyperparams.batch_size} | {hyperparams.precision})")
    return {
        "avg_batch_time_ms": train_avg_ms,
        "batch_times_ms": times_ms,
        "avg_inference_time_ms": inf_avg_ms,
        "inference_times_ms": inf_times_ms,
    }


ONLY_MODELS = None  # set to a set/list of names to filter; None = run all configs

if __name__ == "__main__":
    config_path = Path(__file__).parent / "classification_train_config.json"
    for params in json.loads(config_path.read_text()):
        if ONLY_MODELS is not None and params["model"] not in ONLY_MODELS:
            continue
        try:
            metrics = run(Hyperparams(**params))
            if isinstance(metrics, dict) and "error" in metrics:
                print(f"SKIP ({metrics['error']}): {params}")
        except torch.cuda.OutOfMemoryError:
            print(f"OOM: {params}")
        except Exception as e:
            # Catches cuBLAS/cuDNN errors, plain RuntimeErrors from OOM-adjacent failures,
            # NaN crashes, etc. Don't let one bad config kill the whole sweep.
            print(f"FAIL ({type(e).__name__}): {params} - {e}")
        finally:
            torch.cuda.empty_cache()
