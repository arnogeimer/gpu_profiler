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
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


WARMUP_BATCHES = 5
TIMED_BATCHES = 20

INFER_WARMUP_BATCHES = 1
INFER_TIMED_BATCHES = 3
INFER_MAX_NEW_TOKENS = 64       # fixed across all inference configs for comparability


@dataclass
class Hyperparams:
    model: str = "HuggingFaceTB/SmolLM2-360M"
    sequence_length: int = 512   # training context length AND inference prompt length
    batch_size: int = 2
    lora_rank: int = 16
    precision: str = "fp16"      # fp32 | fp16


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config: LoRA fine-tune (train phase) + autoregressive generation (infer phase)."""
    rows: list[dict] = []
    device = torch.device("cuda")
    monitor = CUDAMonitor(interval_ms=10)
    dtype = torch.float16 if hyperparams.precision == "fp16" else torch.float32

    try:
        base = AutoModelForCausalLM.from_pretrained(hyperparams.model, torch_dtype=dtype)
        lora_cfg = LoraConfig(
            r=hyperparams.lora_rank,
            lora_alpha=hyperparams.lora_rank * 2,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base, lora_cfg)
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}"))
        return rows

    model = model.to(device).train()
    vocab_size = model.config.vocab_size

    # Optimizer only sees the (tiny) LoRA adapter params.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )
    use_fp16 = hyperparams.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    def make_batch():
        return torch.randint(0, vocab_size, (hyperparams.batch_size, hyperparams.sequence_length), device=device)

    def step(input_ids):
        optimizer.zero_grad()
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(input_ids=input_ids, labels=input_ids).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = model(input_ids=input_ids, labels=input_ids).loss
            loss.backward()
            optimizer.step()

    # === Train warmup ===
    try:
        for _ in range(WARMUP_BATCHES):
            step(make_batch())
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, "train", error=f"forward_failed: {e}"))
        return rows

    # === Train timed ===
    monitor.start()
    times_ms = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            ids = make_batch()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step(ids)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True   # record below, then continue to inference
    finally:
        train_metrics = monitor.stop(oom=oom)
        train_avg_ms = sum(times_ms) / len(times_ms) if times_ms else None
        rows.append(build_row(hyperparams, "train", metrics=train_metrics, avg_ms=train_avg_ms))
    torch.cuda.empty_cache()

    # === Inference ===
    model.eval()
    input_ids = torch.randint(0, vocab_size, (hyperparams.batch_size, hyperparams.sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    pad_id = model.config.eos_token_id if model.config.eos_token_id is not None else 0

    def generate():
        with torch.no_grad():
            return model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=INFER_MAX_NEW_TOKENS,
                min_new_tokens=INFER_MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=pad_id,
            )

    # Infer warmup
    try:
        for _ in range(INFER_WARMUP_BATCHES):
            generate()
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, "infer", error=f"forward_failed: {e}"))
        return rows

    # Infer timed
    monitor.start()
    inf_times_ms = []
    oom = False
    try:
        for _ in range(INFER_TIMED_BATCHES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            generate()
            end.record()
            torch.cuda.synchronize()
            inf_times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        infer_metrics = monitor.stop(oom=oom)
        inf_avg_ms = sum(inf_times_ms) / len(inf_times_ms) if inf_times_ms else None
        rows.append(build_row(hyperparams, "infer", metrics=infer_metrics, avg_ms=inf_avg_ms))

    del model, base, optimizer, input_ids, attention_mask
    short = hyperparams.model.split("/")[-1]
    def _fmt(avg_ms, batch):
        return "OOM" if avg_ms is None else f"{avg_ms / batch:.2f} ms/sample"
    print(f"train: {_fmt(train_avg_ms, hyperparams.batch_size)} | inference: {_fmt(inf_avg_ms, hyperparams.batch_size)}  ({short} | seq={hyperparams.sequence_length} | bs={hyperparams.batch_size} | r={hyperparams.lora_rank} | {hyperparams.precision})")
    return rows


MODELS = [
    # tier 1 (≤500M)
    'HuggingFaceTB/SmolLM2-135M',
    'HuggingFaceTB/SmolLM2-360M',
    'Qwen/Qwen2.5-0.5B',
    # tier 2 (~1-1.7B)
    'EleutherAI/pythia-1b',
    'bigcode/starcoderbase-1b',
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'bigscience/bloom-1b1',
    'microsoft/phi-1_5',
    'EleutherAI/pythia-1.4b',
    'Qwen/Qwen2.5-1.5B',
    'Qwen/Qwen2.5-Coder-1.5B',
    'stabilityai/stablelm-2-1_6b',
    'HuggingFaceTB/SmolLM2-1.7B',
    # tier 3 (~2.7-3.8B)
    'microsoft/phi-2',
    'Qwen/Qwen2.5-3B',
    'microsoft/Phi-3-mini-4k-instruct',
]
SEQUENCE_LENGTHS = [256, 512]
BATCH_SIZES = [1, 2, 4]
LORA_RANKS = [16]               # fixed — adapter is tiny, rank rarely changes timing meaningfully
PRECISIONS = ['fp32', 'fp16']


UPLOAD_EVERY_N_MODELS = 5


def run_all(upload_fn: Optional[Callable[[pd.DataFrame], None]] = None,
            only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config; upload after every N models and once at the end."""
    all_rows: list[dict] = []
    selected = [m for m in MODELS if only_models is None or m in only_models]
    for i, model in enumerate(selected):
        for seq, bs, rank, precision in product(SEQUENCE_LENGTHS, BATCH_SIZES, LORA_RANKS, PRECISIONS):
            params = {
                "model": model,
                "sequence_length": seq,
                "batch_size": bs,
                "lora_rank": rank,
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
