"""Audio classification across both input modalities. 15 waveform models (wav2vec2, HuBERT,
WavLM, SEW, data2vec, UniSpeech-SAT) take raw 16 kHz samples through a 1D conv stack -- the only
place in the suite that emits conv1d. 6 spectrogram models (AST, Whisper encoders) take mel
frames instead and treat them as a 2D image. Swept over 2/4/8/16/30s x batch 2/4/8 x
fp32/fp16/bf16 = 945 configs, each timed for one SGD training step inside a captured CUDA
graph, so the measurement is GPU kernel time with no host dispatch in it."""

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
from transformers import AutoConfig, AutoModelForAudioClassification
from transformers.initialization import no_init_weights
from transformers.models.wavlm import modeling_wavlm


def _wavlm_compute_bias(self, query_length: int, key_length: int) -> torch.Tensor:
    """WavLM's relative-position bias, built on the GPU instead of on the host.

    Stock transformers builds the position buckets with two device-less torch.arange calls and
    copies the result to the GPU at the end of compute_bias. That copy is illegal inside a CUDA
    graph capture ("Cannot copy between CPU and CUDA tensors ... unless the CPU tensor is
    pinned"), so every WavLM config failed -- 45/45 rows on all three nodes. Nothing about the
    model placement causes it: all parameters and buffers are already on the GPU, and the tensor
    is constructed fresh inside the forward pass on every call, so no amount of .to() or pinning
    upstream reaches it.

    Building the buckets on-device removes the copy and the model captures normally.

    Note what this changes about the measurement. The bucket arithmetic now runs as GPU kernels
    over a (query_length x key_length) index tensor -- 1499**2 at 30 s -- where stock WavLM runs
    it on the host, invisible to a GPU-time measurement. It is not a distortion of an existing
    number, since stock WavLM cannot be captured at all and so has no number; but WavLM rows do
    carry per-layer work that no other model in the suite carries, which matters when comparing
    across models rather than across cards. SEW-D fails the same way for its own reasons and is
    left out rather than patched twice."""
    device = self.rel_attn_embed.weight.device
    context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
    memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
    bucket = self._relative_positions_bucket(memory_position - context_position)
    return self.rel_attn_embed(bucket).permute([2, 0, 1])


modeling_wavlm.WavLMAttention.compute_bias = _wavlm_compute_bias


# (warmup, repeats, iters) handed to time_fn. The step is measured inside a captured CUDA
# graph, so no Python dispatch lands in the timed window.
TRAIN_TIMING = (3, 10, 2)
TIMING_METHOD = "cuda_graph"

SAMPLE_RATE = 16000       # waveform models expect 16 kHz samples
FRAMES_PER_SECOND = 100   # 25ms window / 10ms hop, the mel framing both AST and Whisper assume

# fp32 runs without autocast; the other two run under it.
AMP_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


def _input_spec(cfg, seconds: int, batch_size: int) -> tuple[str, tuple]:
    """Argument name and tensor shape for one config, resizing the config where that is what
    controls clip length. Must run before the model is built: both spectrogram families size a
    position embedding from these fields.

    The three families disagree on all of it. Waveform models take raw samples as `input_values`.
    AST takes mel frames as `input_values` too but shaped (batch, frames, mels). Whisper takes
    (batch, mels, frames) as `input_features` and pads every clip to 30s unless
    max_source_positions is resized -- left alone, its duration axis measures nothing."""
    if not hasattr(cfg, "num_mel_bins"):
        return "input_values", (batch_size, seconds * SAMPLE_RATE)
    frames = seconds * FRAMES_PER_SECOND
    if cfg.model_type == "whisper":
        cfg.max_source_positions = frames // 2      # the conv frontend halves the frame count
        return "input_features", (batch_size, cfg.num_mel_bins, frames)
    cfg.max_length = frames
    return "input_values", (batch_size, frames, cfg.num_mel_bins)


@dataclass
class Hyperparams:
    model: str = "facebook/wav2vec2-base"   # HuggingFace repo id
    seconds: int = 4                        # waveform length in seconds (sampled at 16 kHz)
    batch_size: int = 4
    precision: str = "fp16"                 # fp32 | fp16 | bf16


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config (train). Returns one or more rows, one per recorded phase."""
    rows: list[dict] = []
    device = torch.device("cuda")
    monitor = CUDAMonitor(interval_ms=10)

    try:
        model_cfg = AutoConfig.from_pretrained(hyperparams.model, num_labels=16)
        # spec_augment runs on the CPU
        model_cfg.apply_spec_augment = False
        input_key, input_shape = _input_spec(model_cfg, hyperparams.seconds, hyperparams.batch_size)
        with no_init_weights():
            model = AutoModelForAudioClassification.from_config(model_cfg)
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}",
                              timing_method=TIMING_METHOD))
        return rows
    model = model.to(device).train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    amp_dtype = AMP_DTYPES.get(hyperparams.precision)

    # One fixed batch rather than a fresh one per step: a captured graph replays the recorded
    # kernels against the addresses they were captured with, so the input cannot move.
    x = torch.randn(*input_shape, device=device)
    y = torch.randint(0, 16, (hyperparams.batch_size,), device=device)

    def step():
        # set_to_none=False keeps the gradient buffers at fixed addresses across replays.
        optimizer.zero_grad(set_to_none=False)
        if amp_dtype is not None:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(**{input_key: x}, labels=y).loss
        else:
            loss = model(**{input_key: x}, labels=y).loss
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


# The SEW-D family is absent on purpose. Like WavLM it builds position buckets on the host
# inside the attention path, so CUDA graph capture rejects the copy and all 45 configs fail --
# sew-d-tiny and sew-d-mid each failed 45/45 rows on all three nodes, costing ~5 min per node to
# produce nothing. WavLM is back because _wavlm_compute_bias above fixes it for that family;
# SEW-D would need its own patch against different code, and is two small models.
MODELS = [
    # small (<50M)
    'ntu-spml/distilhubert',
    'asapp/sew-tiny-100k',
    # base (~94M)
    'facebook/wav2vec2-base',
    'facebook/hubert-base-ls960',
    'microsoft/wavlm-base',
    'microsoft/wavlm-base-plus',
    'facebook/data2vec-audio-base',
    'microsoft/unispeech-sat-base',
    # large (~315M)
    'facebook/wav2vec2-large',
    'facebook/wav2vec2-xls-r-300m',
    'facebook/hubert-large-ls960-ft',
    'microsoft/wavlm-large',
    'facebook/data2vec-audio-large',
    # --- spectrogram input: mel frames rather than raw samples, so no conv1d frontend at all.
    # AST is the most-downloaded audio classification model on HF by an order of magnitude; the
    # two variants differ in patch stride (10 vs 14), which changes the token count for a fixed
    # clip. Whisper encoders are dispatch-bound below 30s at tiny/base, so their duration axis
    # only bites at the long end. ---
    'MIT/ast-finetuned-audioset-10-10-0.4593',
    'MIT/ast-finetuned-audioset-14-14-0.443',
    'openai/whisper-tiny',
    'openai/whisper-base',
    'openai/whisper-small',
    'openai/whisper-medium',
]

SECONDS = [2, 4, 8, 16, 30]
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
        prev = progress_line(model, i, len(selected), "audio classification", prev)
        for seconds, batch_size, precision in product(SECONDS, BATCH_SIZES, PRECISIONS):
            params = {"model": model, "seconds": seconds, "batch_size": batch_size, "precision": precision}
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
