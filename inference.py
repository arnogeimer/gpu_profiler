"""Parked inference code.

The workloads now record training only. The inference phases that used to sit inside them are
kept here verbatim-in-behaviour so they can be restored without being rewritten. Nothing in this
file is wired into main.py.

The two inference-only workloads, workloads/multimodal/vlm_inference.py and
workloads/generative/diffusion_inference.py, are NOT copied here -- they are already standalone
files and were simply dropped from main.WORKLOADS. Re-adding them there brings them back.

Each function returns rows in the shape build_row produces, so restoring a phase means calling it
where the original block sat and extending the caller's row list with the result. Each takes the
already-built model, because the originals ran on the same model instance the training phase had
just finished with.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from profiler.cuda_monitor import build_row
from profiler.profiler import time_fn

WARMUP_BATCHES = 3
TIMED_BATCHES = 6

# (warmup, repeats, iters) for device_probe.time_fn, used by the llm phases below.
PREFILL_TIMING = (3, 5, 3)
DECODE_TIMING = (2, 5, 4)


def _decode_cache_len(sequence_length: int) -> int:
    """Cache slots one decode measurement consumes, on top of the prefilled sequence.

    StaticCache advances its own write position on every execution and ignores the
    cache_position we pass -- CUDA graph replays included. So the cache has to hold one slot
    per execution time_fn performs: warmup, capture, the priming replay, and repeats x iters
    timed replays. Overrunning it trips a device-side index_copy_ assert that poisons the
    CUDA context for the rest of the process, so the +8 is deliberate slack."""
    warm, repeats, iters = DECODE_TIMING
    return sequence_length + warm + iters * (2 + repeats) + 8


def _timed_loop(fn, monitor, hyperparams, phase: str) -> tuple[list[dict], float | None]:
    """Warmup then TIMED_BATCHES cuda-event-timed calls of fn. Shared by the two eager phases."""
    rows: list[dict] = []
    try:
        for _ in range(WARMUP_BATCHES):
            fn()
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, phase, error=f"forward_failed: {e}"))
        return rows, None

    monitor.start()
    times_ms: list[float] = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        metrics = monitor.stop(oom=oom)
        avg_ms = sum(times_ms) / len(times_ms) if times_ms else None
        rows.append(build_row(hyperparams, phase, metrics=metrics, avg_ms=avg_ms))
    return rows, avg_ms


def image_classification_infer(hyperparams, model, monitor, device) -> tuple[list[dict], float | None]:
    """Was the second half of workloads/computer_vision/image_classification.py run().

    Runs at hyperparams.batch_size. The original hardcoded 128 here while build_row stamped the
    config's batch size on the row, so three identical runs were labelled 16/32/64."""
    use_fp16 = hyperparams.precision == "fp16"
    model.eval()
    x_inf = torch.randn(hyperparams.batch_size, 3, hyperparams.img_size, hyperparams.img_size,
                        device=device)

    def infer():
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    return model(x_inf).argmax(dim=1)
            return model(x_inf).argmax(dim=1)

    return _timed_loop(infer, monitor, hyperparams, "infer")


def audio_classification_infer(hyperparams, model, monitor, device, sample_rate: int = 16000):
    """Was the second half of workloads/audio/audio_classification.py run()."""
    use_fp16 = hyperparams.precision == "fp16"
    model.eval()
    x_inf = torch.randn(hyperparams.batch_size, hyperparams.seconds * sample_rate, device=device)

    def infer():
        with torch.no_grad():
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    return model(input_values=x_inf).logits.argmax(dim=1)
            return model(input_values=x_inf).logits.argmax(dim=1)

    return _timed_loop(infer, monitor, hyperparams, "infer")


def llm_prefill_decode(hyperparams, model, monitor, device, vocab_size: int):
    """Was the inference half of workloads/nlp/llm_finetune.py run().

    Two measurements rather than one model.generate() call: a prefill forward over the whole
    sequence (compute-bound) and a single decode step against a prefilled KV cache
    (weight-bandwidth-bound). generate() ran its decode loop in Python at roughly 30ms of host
    dispatch per token, which swamped the GPU work entirely -- a 1.5B model decoded no slower than
    a 135M one, so those rows tracked CPU speed rather than the card. time_fn captures a CUDA
    graph, leaving no Python inside the timed window.

    Inference runs on cast weights rather than autocast over fp32 ones. transformers 5.10 sizes
    StaticCache lazily from the first key tensor it sees (fp32, because RoPE promotes) and then
    tries to write autocast's fp16 keys into it, which raises a dtype mismatch. Casting sidesteps
    that, and costs nothing here: inference needs no fp32 master weights.
    """
    from transformers import StaticCache

    rows: list[dict] = []
    use_fp16 = hyperparams.precision == "fp16"
    model.eval()
    if use_fp16:
        model = model.half()
    seq = hyperparams.sequence_length
    input_ids = torch.randint(0, vocab_size, (hyperparams.batch_size, seq), device=device)

    def prefill():
        with torch.no_grad():
            model(input_ids=input_ids)

    monitor.start()
    prefill_ms, oom, err = None, False, ""
    try:
        prefill_ms = time_fn(prefill, *PREFILL_TIMING)
    except torch.cuda.OutOfMemoryError:
        oom = True
    except RuntimeError as e:
        err = f"prefill_failed: {e}"
    finally:
        rows.append(build_row(hyperparams, "prefill", metrics=monitor.stop(oom=oom),
                              error=err, avg_ms=prefill_ms))
    torch.cuda.empty_cache()

    # Started before the try so a failure during cache setup cannot report the prefill's
    # samples as the decode row's metrics.
    monitor.start()
    decode_ms, oom, err = None, False, ""
    try:
        cache = StaticCache(config=model.config, max_cache_len=_decode_cache_len(seq))
        with torch.no_grad():
            model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(seq, device=device))
        torch.cuda.synchronize()
        next_id = torch.randint(0, vocab_size, (hyperparams.batch_size, 1), device=device)

        def decode():
            with torch.no_grad():
                model(input_ids=next_id, past_key_values=cache, use_cache=True)

        decode_ms = time_fn(decode, *DECODE_TIMING)
    except torch.cuda.OutOfMemoryError:
        oom = True
    except RuntimeError as e:
        err = f"decode_failed: {e}"
    finally:
        rows.append(build_row(hyperparams, "decode", metrics=monitor.stop(oom=oom),
                              error=err, avg_ms=decode_ms))

    return rows, prefill_ms, decode_ms
