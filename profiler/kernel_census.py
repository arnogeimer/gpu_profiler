#!/usr/bin/env python3
"""Enumerate every CUDA kernel the workloads/ suite actually dispatches.

Why this exists
---------------
device_probe.py measures synthetic primitives (GEMM, triad). Those pin the machine's
peak throughput and its memory hierarchy, but they are not what the models run: a
resnet50 training step dispatches ~840 kernels, of which the GEMMs are a minority and
the rest are cuDNN implicit-GEMM convolutions, normalisations, and hundreds of tiny
elementwise/reduce kernels. This walks every model in every workload, profiles one
representative step, and collects the kernel names.

The output is the raw name set plus the label(s) that produced each name. Normalisation
(collapsing tile sizes, so one kernel type counts once) is deliberately NOT baked in
here — it is applied offline by kernel_normalize.py, so the rules can be changed
without re-running the sweep, which takes tens of minutes.

    uv run profiler/kernel_census.py --out kernels.json
    uv run profiler/kernel_census.py --out kernels.json --only vision,audio
"""

import argparse
import gc
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

PRECISIONS = ("fp16", "fp32")


def kernels_of(step, warmup: int = 1) -> list[str]:
    """Demangled CUDA kernel names dispatched by one call of `step`.

    Warmup first so cuDNN/cuBLAS algorithm selection and lazy init do not land in the
    trace; those emit kernels that no steady-state step would run."""
    from torch.profiler import ProfilerActivity, profile

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        step()
        torch.cuda.synchronize()
    return [torch._C._demangle(k.name) for e in p.events() for k in e.kernels]


def _autocast_step(fn, precision: str):
    """Wrap fn in the same autocast the workloads use — fp16 vs fp32 select different
    kernels entirely (tensor-core hgemm vs sgemm), so both are swept."""
    if precision != "fp16":
        return fn

    def step():
        with torch.autocast("cuda", dtype=torch.float16):
            return fn()

    return step


# --------------------------------------------------------------------------- vision
def jobs_vision():
    """workloads/computer_vision/image_classification.py — timm, train step."""
    import timm
    import torch.nn as nn
    from workloads.computer_vision import image_classification as W

    for name in W.MODELS:
        try:
            try:
                model = timm.create_model(name, pretrained=False, num_classes=16, img_size=224)
            except TypeError:
                model = timm.create_model(name, pretrained=False, num_classes=16)
            model = model.cuda().train()
        except Exception as e:
            yield f"vision/{name}", None, f"create_failed: {e}"
            continue

        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        crit = nn.CrossEntropyLoss()
        x = torch.randn(16, 3, 224, 224, device="cuda")
        y = torch.randint(0, 16, (16,), device="cuda")

        def train(model=model, opt=opt, crit=crit, x=x, y=y):
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()

        for p in PRECISIONS:
            yield f"vision/{name}/{p}", _autocast_step(train, p), None
        del model, opt, crit, x, y


# ---------------------------------------------------------------------------- audio
def jobs_audio():
    """workloads/audio/audio_classification.py — HF audio encoders, train step."""
    from transformers import AutoConfig, AutoModelForAudioClassification
    from transformers.initialization import no_init_weights
    from workloads.audio import audio_classification as W

    for name in W.MODELS:
        try:
            cfg = AutoConfig.from_pretrained(name, num_labels=16)
            with no_init_weights():
                model = AutoModelForAudioClassification.from_config(cfg)
            model = model.cuda().train()
        except Exception as e:
            yield f"audio/{name}", None, f"create_failed: {e}"
            continue

        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        x = torch.randn(2, 2 * W.SAMPLE_RATE, device="cuda")
        y = torch.randint(0, 16, (2,), device="cuda")

        def train(model=model, opt=opt, x=x, y=y):
            opt.zero_grad()
            loss = model(input_values=x, labels=y).loss
            loss.backward()
            opt.step()

        for p in PRECISIONS:
            yield f"audio/{name}/{p}", _autocast_step(train, p), None
        del model, opt, x, y


# ------------------------------------------------------------------------------ nlp
def _param_estimate(cfg) -> float:
    """Rough parameter count in billions, from config alone. Used to skip models whose
    from_config allocation would not fit — the sweep needs the architecture, and the
    7B tier cannot be built on a 12 GB card either way."""
    h = getattr(cfg, "hidden_size", 0) or getattr(cfg, "n_embd", 0) or 0
    layers = getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "n_layer", 0) or 0
    vocab = getattr(cfg, "vocab_size", 0) or 0
    inter = getattr(cfg, "intermediate_size", 4 * h) or 4 * h
    return (layers * (4 * h * h + 3 * h * inter) + 2 * vocab * h) / 1e9


def jobs_nlp(max_params_b: float):
    """workloads/nlp/llm_finetune.py — LoRA fine-tune step + prefill + single decode step."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.initialization import no_init_weights
    from workloads.nlp import llm_finetune as W

    for name in W.MODELS:
        for p in PRECISIONS:
            dtype = torch.float16 if p == "fp16" else torch.float32
            try:
                cfg = AutoConfig.from_pretrained(name)
                est = _param_estimate(cfg)
                if est > max_params_b:
                    yield f"nlp/{name}/{p}", None, f"skipped: ~{est:.1f}B params"
                    continue
                with no_init_weights():
                    base = AutoModelForCausalLM.from_config(cfg, dtype=dtype)
                model = get_peft_model(base, LoraConfig(
                    r=16, lora_alpha=32, target_modules="all-linear",
                    lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
                model = model.cuda().train()
            except Exception as e:
                yield f"nlp/{name}/{p}", None, f"create_failed: {type(e).__name__}: {e}"
                continue

            opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=1e-4)
            ids = torch.randint(0, cfg.vocab_size, (1, 256), device="cuda")

            def train(model=model, opt=opt, ids=ids):
                opt.zero_grad()
                model(input_ids=ids, labels=ids).loss.backward()
                opt.step()

            yield f"nlp/{name}/{p}/train", _autocast_step(train, p), None

            # generation is a different kernel regime: prefill then single-token decode,
            # where the GEMMs degenerate to GEMV and the elementwise tail dominates
            def gen(model=model, ids=ids):
                with torch.no_grad():
                    model.eval()
                    out = model.generate(input_ids=ids, max_new_tokens=4, min_new_tokens=4,
                                         do_sample=False, pad_token_id=0)
                    model.train()
                    return out

            yield f"nlp/{name}/{p}/generate", gen, None
            del model, base, opt, ids


# ----------------------------------------------------------------------- multimodal
def jobs_vlm():
    """workloads/multimodal/vlm_inference.py — image+text generate."""
    import numpy as np
    from PIL import Image
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor
    from transformers.initialization import no_init_weights
    from workloads.multimodal import vlm_inference as W

    for name in W.MODELS:
        for p in PRECISIONS:
            dtype = torch.float16 if p == "fp16" else torch.float32
            try:
                proc = AutoProcessor.from_pretrained(name)
                cfg = AutoConfig.from_pretrained(name)
                with no_init_weights():
                    model = AutoModelForImageTextToText.from_config(cfg, dtype=dtype)
                model = model.cuda().eval()
                img = Image.fromarray((np.random.rand(384, 384, 3) * 255).astype(np.uint8))
                text = proc.apply_chat_template(
                    [{"role": "user", "content": [{"type": "image"},
                                                  {"type": "text", "text": W.PROMPT}]}],
                    add_generation_prompt=True)
                inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
            except Exception as e:
                yield f"vlm/{name}/{p}", None, f"create_failed: {type(e).__name__}: {e}"
                continue

            def gen(model=model, inputs=inputs):
                with torch.no_grad():
                    return model.generate(**inputs, max_new_tokens=4, min_new_tokens=4,
                                          do_sample=False)

            yield f"vlm/{name}/{p}", gen, None
            del model, inputs


# ----------------------------------------------------------------------- generative
def jobs_diffusion():
    """workloads/generative/diffusion_inference.py — 2-step denoise (kernels repeat per step)."""
    from workloads.generative import diffusion_inference as W

    for name in W.MODELS:
        for p in PRECISIONS:
            dtype = torch.float16 if p == "fp16" else torch.float32
            try:
                pipe = W._build_pipeline_random(name, dtype, torch.device("cuda"))
                pipe.set_progress_bar_config(disable=True)
            except Exception as e:
                yield f"diffusion/{name}/{p}", None, f"create_failed: {type(e).__name__}: {e}"
                continue

            def gen(pipe=pipe):
                return pipe(W.PROMPT, num_inference_steps=2, height=512, width=512,
                            num_images_per_prompt=1).images

            yield f"diffusion/{name}/{p}", gen, None
            del pipe


# ------------------------------------------------------------------------- robotics
def jobs_robotics():
    """workloads/robotics/le_world_model.py — JEPA train step (encoder + predictor + EMA)."""
    import timm
    from workloads.robotics import le_world_model as W

    for name in W.ENCODERS:
        for p in PRECISIONS:
            try:
                online = timm.create_model(name, pretrained=False, num_classes=0, img_size=224)
                target = timm.create_model(name, pretrained=False, num_classes=0, img_size=224)
                target.load_state_dict(online.state_dict())
                for q in target.parameters():
                    q.requires_grad_(False)
                with torch.no_grad():
                    emb = W._encode(online, torch.zeros(1, 3, 224, 224)).shape[-1]
                pred = W.Predictor(emb_dim=emb, action_dim=6, num_predictions=4)
                online, target, pred = online.cuda().train(), target.cuda().eval(), pred.cuda().train()
            except Exception as e:
                yield f"robotics/{name}/{p}", None, f"create_failed: {type(e).__name__}: {e}"
                continue

            opt = torch.optim.AdamW(list(online.parameters()) + list(pred.parameters()), lr=1e-4)
            bs, ctx, npred = 2, 8, 4
            frames = torch.randn(bs, ctx + npred, 3, 224, 224, device="cuda")
            actions = torch.randn(bs, npred, 6, device="cuda")

            def train(online=online, target=target, pred=pred, opt=opt,
                      frames=frames, actions=actions):
                opt.zero_grad()
                flat = frames.flatten(0, 1)
                embs = W._encode(online, flat).view(bs, ctx + npred, -1)
                with torch.no_grad():
                    tgt = W._encode(target, flat).view(bs, ctx + npred, -1)[:, ctx:]
                out = pred(embs[:, :ctx], actions)
                loss = torch.nn.functional.mse_loss(out, tgt)
                loss.backward()
                opt.step()
                W._ema_update(target, online)

            yield f"robotics/{name}/{p}", _autocast_step(train, p), None
            del online, target, pred, opt, frames, actions


SUITES = {
    "vision": jobs_vision, "audio": jobs_audio, "nlp": jobs_nlp,
    "vlm": jobs_vlm, "diffusion": jobs_diffusion, "robotics": jobs_robotics,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="kernels.json")
    ap.add_argument("--only", help="comma-separated suite names (default: all)")
    ap.add_argument("--max-params-b", type=float, default=3.5,
                    help="skip causal LMs above this many billion params (12 GB card)")
    args = ap.parse_args()

    want = args.only.split(",") if args.only else list(SUITES)
    seen: dict[str, list[str]] = {}     # raw kernel name -> labels that dispatched it
    skipped: dict[str, str] = {}
    growth: list[dict] = []            # set size after each job, to show saturation
    t0 = time.time()

    for suite in want:
        gen = SUITES[suite](args.max_params_b) if suite == "nlp" else SUITES[suite]()
        while True:
            try:
                label, step, err = next(gen)
            except StopIteration:
                break
            except Exception as e:
                print(f"  !! {suite} generator died: {type(e).__name__}: {e}", flush=True)
                traceback.print_exc()
                break
            if step is None:
                skipped[label] = err
                print(f"  -- {label}: {err}", flush=True)
                continue
            try:
                names = kernels_of(step)
                new = 0
                for n in names:
                    if n not in seen:
                        seen[n] = []
                        new += 1
                    if len(seen[n]) < 8:
                        seen[n].append(label)
                growth.append({"label": label, "launches": len(names),
                               "unique_here": len(set(names)), "total": len(seen)})
                print(f"  {label}: {len(names)} launches, {len(set(names))} unique, "
                      f"+{new} new, total {len(seen)}  [{time.time()-t0:.0f}s]", flush=True)
            except Exception as e:
                skipped[label] = f"profile_failed: {type(e).__name__}: {e}"
                print(f"  -- {label}: profile_failed: {type(e).__name__}: {e}", flush=True)
            finally:
                gc.collect()
                torch.cuda.empty_cache()

            with open(args.out, "w") as f:      # incremental: the sweep is long
                json.dump({"kernels": seen, "skipped": skipped, "growth": growth}, f, indent=1)

    print(f"\n{len(seen)} distinct raw kernel names from {len(growth)} profiled steps "
          f"({len(skipped)} skipped) in {time.time()-t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
