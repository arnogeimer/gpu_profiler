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

import diffusers
import transformers
diffusers.logging.set_verbosity_error()
transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore")
from transformers import AutoConfig
from transformers.initialization import no_init_weights


WARMUP_BATCHES = 1     # one full denoising loop to JIT-compile kernels
TIMED_BATCHES = 5
PROMPT = "a photograph of an astronaut riding a horse"  # fixed arbitrary prompt


@dataclass
class Hyperparams:
    model: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 20
    image_size: int = 512                       # square H=W
    batch_size: int = 1                         # num_images_per_prompt
    precision: str = "fp16"                     # fp32 | fp16


def _build_pipeline_random(model_id: str, dtype: torch.dtype, device):
    """Assemble the text2image pipeline from each component's *config* only — no checkpoint
    download. We profile compute, not accuracy: timing depends on architecture (shapes/dtype),
    never on weight values, and the denoising loop runs a fixed num_inference_steps, so random
    weights leave every measured step unchanged (mirrors the config-only builds elsewhere).
    Weight-bearing modules (unet, vae, text encoder(s)) get random/uninitialised weights; the
    weightless bits (tokenizer, scheduler) load normally from their small config/vocab files."""
    index = diffusers.DiffusionPipeline.load_config(model_id)
    pipe_cls = getattr(diffusers, index["_class_name"])
    components = {}
    for name, spec in index.items():
        # component entries look like [library, class_name]; skip _class_name / scalars.
        if not (isinstance(spec, list) and len(spec) == 2):
            continue
        lib, cls_name = spec
        if lib is None:                              # optional component absent (safety_checker, ...)
            components[name] = None
        elif lib == "diffusers":
            cls = getattr(diffusers, cls_name)
            if "Scheduler" in cls_name:              # scheduler: config only, no weights
                components[name] = cls.from_pretrained(model_id, subfolder=name)
            else:                                    # unet / vae: random-init from config
                components[name] = cls.from_config(cls.load_config(model_id, subfolder=name))
        elif lib == "transformers":
            cls = getattr(transformers, cls_name)
            if "Tokenizer" in cls_name:              # tokenizer: vocab files, no weights
                components[name] = cls.from_pretrained(model_id, subfolder=name)
            else:                                    # text encoder: random-init, skip the weights
                text_cfg = AutoConfig.from_pretrained(model_id, subfolder=name)
                with no_init_weights():
                    components[name] = cls(text_cfg)
    return pipe_cls(**components).to(device=device, dtype=dtype)


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run a single config (inference only — diffusion training is impractical on consumer GPUs).
    Returns one or more rows, one per recorded phase."""
    rows: list[dict] = []
    device = torch.device("cuda")
    monitor = CUDAMonitor(interval_ms=10)
    dtype = torch.float16 if hyperparams.precision == "fp16" else torch.float32

    try:
        pipe = _build_pipeline_random(hyperparams.model, dtype, device)
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}"))
        return rows

    def generate():
        return pipe(
            PROMPT,
            num_inference_steps=hyperparams.num_inference_steps,
            height=hyperparams.image_size,
            width=hyperparams.image_size,
            num_images_per_prompt=hyperparams.batch_size,
        ).images

    # Warmup (one full denoising loop kicks off cuDNN/SDPA kernel selection)
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

    del pipe
    short = hyperparams.model.split("/")[-1]
    def _fmt(avg_ms, batch):
        return "OOM" if avg_ms is None else f"{avg_ms / batch:.0f} ms/image"
    print(f"inference: {_fmt(inf_avg_ms, hyperparams.batch_size)}  ({short} | {hyperparams.image_size}px | bs={hyperparams.batch_size} | steps={hyperparams.num_inference_steps} | {hyperparams.precision})")
    return rows


MODELS = [
    # small (distilled / pruned SD 1.5)
    'segmind/tiny-sd',
    # medium-small
    'stabilityai/sd-turbo',
    # SD 1.x / 2.x family
    'runwayml/stable-diffusion-v1-5',
    'stabilityai/stable-diffusion-2-1',
    # distilled SDXL (smaller than full SDXL)
    'segmind/SSD-1B',
    # SDXL family
    'stabilityai/stable-diffusion-xl-base-1.0',
    'stabilityai/sdxl-turbo',
]
IMAGE_SIZES = [512, 1024]
BATCH_SIZES = [1, 2, 4]
NUM_INFERENCE_STEPS = [20]      # fixed — per-step latency is what matters; multiplying steps just scales timing
PRECISIONS = ['fp32', 'fp16']


UPLOAD_EVERY_N_MODELS = 20      # only ~4 models → effectively upload only at the end


def run_all(upload_fn: Optional[Callable[[pd.DataFrame], None]] = None,
            only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config; upload after every N models and once at the end."""
    all_rows: list[dict] = []
    selected = [m for m in MODELS if only_models is None or m in only_models]
    for i, model in enumerate(selected):
        for image_size, batch_size, num_steps, precision in product(IMAGE_SIZES, BATCH_SIZES, NUM_INFERENCE_STEPS, PRECISIONS):
            params = {
                "model": model,
                "num_inference_steps": num_steps,
                "image_size": image_size,
                "batch_size": batch_size,
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
