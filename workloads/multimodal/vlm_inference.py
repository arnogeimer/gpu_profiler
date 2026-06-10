import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row

import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import AutoProcessor, AutoModelForImageTextToText


WARMUP_BATCHES = 1
TIMED_BATCHES = 3
PROMPT = "Describe this image in one short sentence."


def _make_fake_image(size: int = 384) -> Image.Image:
    """Random RGB PIL image — VLM doesn't care about content for timing."""
    arr = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    return Image.fromarray(arr)


@dataclass
class Hyperparams:
    model: str = "HuggingFaceTB/SmolVLM-256M-Instruct"
    batch_size: int = 1                         # num image+prompt pairs per generate call
    max_new_tokens: int = 64
    precision: str = "fp16"                     # fp32 | fp16


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config (inference only — VLM training is impractical on consumer GPUs).
    Returns one or more rows, one per recorded phase."""
    rows: list[dict] = []
    device = torch.device("cuda")
    monitor = CUDAMonitor(interval_ms=10)
    dtype = torch.float16 if hyperparams.precision == "fp16" else torch.float32

    try:
        processor = AutoProcessor.from_pretrained(hyperparams.model)
        model = AutoModelForImageTextToText.from_pretrained(
            hyperparams.model, torch_dtype=dtype,
        ).to(device).eval()
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}"))
        return rows

    images = [_make_fake_image() for _ in range(hyperparams.batch_size)]
    # Apply the model's chat template so the image placeholder token is inserted correctly.
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}
    ]
    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        prompts = [text] * hyperparams.batch_size
        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(device)
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"processor_failed: {e}"))
        return rows

    def generate():
        with torch.no_grad():
            return model.generate(
                **inputs,
                max_new_tokens=hyperparams.max_new_tokens,
                min_new_tokens=hyperparams.max_new_tokens,   # force fixed length for reproducible timing
                do_sample=False,
            )

    try:
        for _ in range(WARMUP_BATCHES):
            generate()
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, "infer", error=f"forward_failed: {e}"))
        return rows

    monitor.start()
    times_ms = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            generate()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        infer_metrics = monitor.stop(oom=oom)
        inf_avg_ms = sum(times_ms) / len(times_ms) if times_ms else None
        rows.append(build_row(hyperparams, "infer", metrics=infer_metrics, avg_ms=inf_avg_ms))

    del model, processor, inputs
    short = hyperparams.model.split("/")[-1]
    def _fmt(avg_ms, batch, tokens):
        if avg_ms is None:
            return "OOM"
        return f"{avg_ms / batch:.0f} ms/sample ({avg_ms / batch / tokens:.1f} ms/token)"
    print(f"inference: {_fmt(inf_avg_ms, hyperparams.batch_size, hyperparams.max_new_tokens)}  ({short} | bs={hyperparams.batch_size} | tokens={hyperparams.max_new_tokens} | {hyperparams.precision})")
    return rows


MODELS = [
    # tiny
    'HuggingFaceTB/SmolVLM-256M-Instruct',
    # small (~0.5B)
    'HuggingFaceTB/SmolVLM-500M-Instruct',
    'llava-hf/llava-onevision-qwen2-0.5b-ov-hf',
    'llava-hf/llava-interleave-qwen-0.5b-hf',
    # medium (~2B)
    'Qwen/Qwen2-VL-2B-Instruct',
    'HuggingFaceTB/SmolVLM-Instruct',
]
BATCH_SIZES = [1, 2]
MAX_NEW_TOKENS = [32, 64]
PRECISIONS = ['fp32', 'fp16']


UPLOAD_EVERY_N_MODELS = 20      # only 6 models → upload only at the end


def run_all(upload_fn: Optional[Callable[[pd.DataFrame], None]] = None,
            only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config; upload after every N models and once at the end."""
    all_rows: list[dict] = []
    selected = [m for m in MODELS if only_models is None or m in only_models]
    for i, model in enumerate(selected):
        for batch_size, max_new_tokens, precision in product(BATCH_SIZES, MAX_NEW_TOKENS, PRECISIONS):
            params = {
                "model": model,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "precision": precision,
            }
            try:
                all_rows.extend(run(Hyperparams(**params)))
            except Exception as e:
                print(f"FAIL ({type(e).__name__}): {params} - {e}")
                all_rows.append({**params, "phase": "config_failed", "error": f"{type(e).__name__}: {e}"})
            finally:
                torch.cuda.empty_cache()

        is_milestone = (i + 1) % UPLOAD_EVERY_N_MODELS == 0
        is_last = i == len(selected) - 1
        if upload_fn and (is_milestone or is_last):
            try:
                upload_fn(pd.DataFrame(all_rows))
            except Exception as e:
                print(f"  upload after {model} failed: {e}")

    return pd.DataFrame(all_rows)
