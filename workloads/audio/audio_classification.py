import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row

import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import AutoConfig, AutoModelForAudioClassification


WARMUP_BATCHES = 5
TIMED_BATCHES = 20

SAMPLE_RATE = 16000  # every model expects 16 kHz waveform input


@dataclass
class Hyperparams:
    model: str = "facebook/wav2vec2-base"   # HuggingFace repo id
    seconds: int = 4                        # waveform length in seconds (sampled at 16 kHz)
    batch_size: int = 16
    precision: str = "fp16"                 # fp32 | fp16


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config (train + infer). Returns one or more rows, one per recorded phase."""
    rows: list[dict] = []
    num_samples = hyperparams.seconds * SAMPLE_RATE
    device = torch.device("cuda")
    monitor = CUDAMonitor(interval_ms=10)

    try:
        model_cfg = AutoConfig.from_pretrained(hyperparams.model, num_labels=16)
        model = AutoModelForAudioClassification.from_config(model_cfg)
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}"))
        return rows
    model = model.to(device).train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    use_fp16 = hyperparams.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    def make_batch():
        x = torch.randn(hyperparams.batch_size, num_samples, device=device)
        y = torch.randint(0, 16, (hyperparams.batch_size,), device=device)
        return x, y

    def step(x, y):
        optimizer.zero_grad()
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_values=x, labels=y).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(input_values=x, labels=y).loss
            loss.backward()
            optimizer.step()

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
    x_inf = torch.randn(hyperparams.batch_size, num_samples, device=device)

    def infer(x):
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    return model(input_values=x).logits.argmax(dim=1)
            else:
                return model(input_values=x).logits.argmax(dim=1)

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
    short = hyperparams.model.split("/")[-1]
    def _fmt(avg_ms, batch):
        return "OOM" if avg_ms is None else f"{avg_ms / batch:.3f} ms/sample"
    print(f"train: {_fmt(train_avg_ms, hyperparams.batch_size)} | inference: {_fmt(inf_avg_ms, hyperparams.batch_size)}  ({short} | {hyperparams.seconds}s | bs={hyperparams.batch_size} | {hyperparams.precision})")
    return rows


MODELS = [
    'facebook/wav2vec2-base',
    'facebook/wav2vec2-large',
    'facebook/wav2vec2-xls-r-300m',
    'facebook/hubert-base-ls960',
    'facebook/hubert-large-ls960-ft',
    'microsoft/wavlm-base',
    'microsoft/wavlm-base-plus',
    'microsoft/wavlm-large',
    'facebook/data2vec-audio-base',
    'facebook/data2vec-audio-large',
    'microsoft/unispeech-sat-base',
]
SECONDS = [1, 4, 8]
BATCH_SIZES = [16, 32, 64]
PRECISIONS = ['fp32', 'fp16']


def run_all(upload_fn: Optional[Callable[[pd.DataFrame], None]] = None,
            only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config; after each model is finished, call upload_fn(current DataFrame)."""
    all_rows: list[dict] = []
    for model in MODELS:
        if only_models is not None and model not in only_models:
            continue
        for seconds, batch_size, precision in product(SECONDS, BATCH_SIZES, PRECISIONS):
            params = {"model": model, "seconds": seconds, "batch_size": batch_size, "precision": precision}
            try:
                all_rows.extend(run(Hyperparams(**params)))
            except Exception as e:
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
