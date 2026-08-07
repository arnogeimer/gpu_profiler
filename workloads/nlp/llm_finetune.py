import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row
from profiler.profiler import time_fn

import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.initialization import no_init_weights
from peft import LoraConfig, get_peft_model


# (warmup, repeats, iters) handed to time_fn. The step is measured inside a captured CUDA graph,
# so no Python dispatch lands in the timed window.
TRAIN_TIMING = (3, 5, 2)
TIMING_METHOD = "cuda_graph"

# fp32 runs without autocast; the other two run under it.
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


@dataclass
class Hyperparams:
    model: str = "HuggingFaceTB/SmolLM2-360M"
    sequence_length: int = 512   # training context length AND inference prompt length
    batch_size: int = 2
    lora_rank: int = 16
    precision: str = "fp16"      # fp32 | fp16 | bf16


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config: one LoRA fine-tune step. Inference lives in inference.py."""
    rows: list[dict] = []
    device = torch.device("cuda")
    monitor = CUDAMonitor(interval_ms=10)
    # Weights in the config's own precision. fp32 masters existed only to make GradScaler
    # meaningful, and the scaler was dropped so the step could be CUDA-graph captured -- so
    # holding fp32 weights now just doubles the footprint, which put the whole 7B tier beyond
    # every card in host_info.STOCK_TDP_W (29-30GB of weights against a 32GB ceiling).
    dtype = AMP_DTYPES.get(hyperparams.precision, torch.float32)

    try:
        # We profile compute, not accuracy: step timings depend only on the architecture
        # (shapes/dtype), never on weight values. So skip the checkpoint entirely — build from
        # config.json alone (no multi-GB download) and skip CPU random-init (no_init_weights).
        # These models are all dense (no value-dependent routing), so uninitialized weights leave
        # every measured step unchanged -- nothing here branches on a value or validates one.
        cfg = AutoConfig.from_pretrained(hyperparams.model)
        with no_init_weights():
            base = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
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
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}",
                              timing_method=TIMING_METHOD))
        return rows

    model = model.to(device).train()
    vocab_size = model.config.vocab_size

    # Optimizer only sees the (tiny) LoRA adapter params. capturable=True keeps AdamW's step
    # counter on-device; the default reads a CPU scalar, which fails under graph capture.
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        capturable=True,
    )
    amp_dtype = AMP_DTYPES.get(hyperparams.precision)

    train_ids = torch.randint(0, vocab_size, (hyperparams.batch_size, hyperparams.sequence_length), device=device)

    def step():
        # set_to_none=False keeps the gradient buffers at fixed addresses across replays; the
        # default frees them, which would move allocations under the graph.
        optimizer.zero_grad(set_to_none=False)
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(input_ids=train_ids, labels=train_ids).loss
        else:
            loss = model(input_ids=train_ids, labels=train_ids).loss
        loss.backward()
        optimizer.step()

    monitor.start()
    train_avg_ms, oom, err = None, False, ""
    try:
        step()      # materialise gradients and AdamW's lazy state before the capture
        torch.cuda.synchronize()
        train_avg_ms = time_fn(step, *TRAIN_TIMING)
    except torch.cuda.OutOfMemoryError:
        oom = True   # record below, then continue to inference
    except RuntimeError as e:
        err = f"train_failed: {e}"
    finally:
        rows.append(build_row(hyperparams, "train", metrics=monitor.stop(oom=oom),
                              error=err, avg_ms=train_avg_ms, timing_method=TIMING_METHOD))
    torch.cuda.empty_cache()

    del model, base, optimizer, train_ids
    short = hyperparams.model.split("/")[-1]
    def _fmt(avg_ms, batch):
        return "OOM" if avg_ms is None else f"{avg_ms / batch:.2f} ms/sample"
    print(f"train: {_fmt(train_avg_ms, hyperparams.batch_size)}"
          f"  ({short} | seq={hyperparams.sequence_length} | bs={hyperparams.batch_size} | r={hyperparams.lora_rank} | {hyperparams.precision})")
    return rows


MODELS = [
    # tier 1 (≤500M). gpt2/distilgpt2 are the only non-RoPE entries: learned absolute position
    # embeddings, LayerNorm and Conv1D projections instead of RoPE/RMSNorm/nn.Linear, so they
    # exercise a decoder stack the other 22 do not.
    'openai-community/gpt2',
    'distilbert/distilgpt2',
    'HuggingFaceTB/SmolLM2-135M',
    'HuggingFaceTB/SmolLM2-360M',
    'Qwen/Qwen2.5-0.5B',
    'Qwen/Qwen3-0.6B',
    # tier 2 (~1-1.7B)
    'EleutherAI/pythia-1b',
    'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    'bigscience/bloom-1b1',
    'microsoft/phi-1_5',
    'EleutherAI/pythia-1.4b',
    'Qwen/Qwen2.5-1.5B',
    'Qwen/Qwen2.5-Coder-1.5B',
    'stabilityai/stablelm-2-1_6b',
    'HuggingFaceTB/SmolLM2-1.7B',
    'Qwen/Qwen3-1.7B',
    # tier 3 (~2.8-3.8B)
    'EleutherAI/pythia-2.8b',
    'microsoft/phi-2',
    'Qwen/Qwen2.5-3B',
    'microsoft/Phi-3-mini-4k-instruct',
    'microsoft/Phi-3.5-mini-instruct',
    'Qwen/Qwen3-4B',
    # tier 4 (~7B)
    'HuggingFaceH4/zephyr-7b-beta',
    'mistralai/Mistral-7B-v0.3',
    'Qwen/Qwen2.5-7B',
    'Qwen/Qwen2.5-Coder-7B',
    'Qwen/Qwen3-8B',
]
# Extended past 512: the whole workload suite used to top out there while device_probe sweeps
# attention to 4096, so the predictor was extrapolating. Real LoRA finetuning runs at 2k-8k.
SEQUENCE_LENGTHS = [256, 512, 1024, 2048]
BATCH_SIZES = [1, 2, 4]
LORA_RANKS = [16]               # fixed — adapter is tiny, rank rarely changes timing meaningfully
PRECISIONS = ['fp32', 'fp16', 'bf16']



def run_all(only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config. Returns one row per recorded phase; main.py uploads the result."""
    all_rows: list[dict] = []
    selected = [m for m in MODELS if only_models is None or m in only_models]
    for model in selected:
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

    return pd.DataFrame(all_rows)
