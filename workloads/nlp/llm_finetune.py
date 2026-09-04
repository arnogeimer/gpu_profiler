import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row, is_oom, progress_line
from profiler.profiler import time_fn

import transformers
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.initialization import no_init_weights
from peft import LoraConfig, get_peft_model


# (warmup, repeats, iters) handed to time_fn. The step is measured inside a captured CUDA graph,
# so no Python dispatch lands in the timed window.
TRAIN_TIMING = (3, 10, 2)
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

        # A sequence longer than the model's learned position table indexes past the end of it,
        # and the embedding gather fails as a DEVICE-SIDE ASSERT rather than a Python exception:
        # the CUDA context is poisoned, every later call fails, and the process is unrecoverable.
        # So this has to be prevented, not caught. gpt2 and distilgpt2 are the only two of the 27
        # with a fixed table (n_positions=1024) -- everything else is RoPE or ALiBi and has no
        # limit worth checking -- and gpt2 is first in MODELS, so the assert killed llm_finetune
        # at config 1 on every node and object_detection never ran at all.
        max_pos = getattr(cfg, "max_position_embeddings", None) or getattr(cfg, "n_positions", None)
        if max_pos and hyperparams.sequence_length > max_pos:
            rows.append(build_row(
                hyperparams, "setup", timing_method=TIMING_METHOD,
                error=f"sequence_length {hyperparams.sequence_length} exceeds the model's "
                      f"{max_pos} position embeddings"))
            return rows

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

    # Guarded, unlike the other workloads: this is the only one whose weights alone can exceed
    # the card. A 3.8B model in fp32 is ~15 GB before a single activation, so .to(device) is
    # where the largest tier OOMs -- and an OOM here used to escape run() entirely and be
    # recorded by run_all as `config_failed` with the OOM text buried in `error`. That mislabelled
    # 4 046 rows as failures when they are the expected, informative answer: it does not fit.
    #
    # The monitor starts before the move rather than before the step, so a config that dies here
    # still reports the memory it reached. Everything downstream reads peak memory to decide
    # whether an OOM sits at the capacity boundary, and a row with no memory at all cannot be
    # placed.
    monitor.start()
    try:
        model = model.to(device).train()
        vocab_size = model.config.vocab_size

        # Optimizer only sees the (tiny) LoRA adapter params. capturable=True keeps AdamW's step
        # counter on-device; the default reads a CPU scalar, which fails under graph capture.
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-4,
            capturable=True,
        )
        train_ids = torch.randint(0, vocab_size,
                                  (hyperparams.batch_size, hyperparams.sequence_length),
                                  device=device)
    except Exception as e:
        metrics = monitor.stop()
        # is_oom covers all three routes an out-of-VRAM config arrives by; see cuda_monitor.is_oom.
        if not is_oom(e, metrics.get("max_memory_used_pct")):
            raise
        metrics["oom"] = True
        rows.append(build_row(hyperparams, "train", metrics=metrics,
                              timing_method=TIMING_METHOD))
        del model, base
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass    # a context this broken cannot be helped by clearing its cache anyway
        return rows
    amp_dtype = AMP_DTYPES.get(hyperparams.precision)

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

    # No second monitor.start(): it was started before the move above, and starting again would
    # leave the first polling thread running alongside the second.
    train_avg_ms, err, exc = None, "", None
    try:
        step()      # materialise gradients and AdamW's lazy state before the capture
        torch.cuda.synchronize()
        train_avg_ms = time_fn(step, *TRAIN_TIMING)
    except Exception as e:
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

    del model, base, optimizer, train_ids
    return rows


# bloom-1b1, Phi-3-mini-4k-instruct and Phi-3.5-mini-instruct are absent on purpose. All three
# build a host tensor inside their forward -- BLOOM its alibi slopes, the Phi-3 pair their rotary
# inverse frequencies -- so CUDA graph capture rejects the copy exactly as it did for WavLM and
# SEW-D in audio_classification. Across 17 cards and 36 configs each they produced 1 836 rows and
# not one timing: every config that had memory to run at all failed to capture, and the rest OOM'd.
#
# Phi-3.5 was checked alone in a fresh process rather than assumed guilty by association: it fails
# on its own, so this is architectural and not fallout from Phi-3-mini running before it. Nothing
# is recoverable by re-running them, and patching two families for zero rows is not worth the
# measurement caveat a patch carries (see the WavLM note in audio_classification for what that
# costs). Their absence is why tier 2 holds 9 and tier 3 holds 4.
#
# They were not free to keep. Each burned a slot's worth of capture attempts per card, and the
# Phi-3 failures poisoned the CUDA RNG generator process-wide, which cost object_detection every
# row it ever collected -- see profiler._clear_capture_state, which now contains that damage.
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
        prev = progress_line(model, i, len(selected), "llm finetune", prev)
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
