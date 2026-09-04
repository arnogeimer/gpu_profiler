"""The v2 probe extension: the shapes the workloads dispatch that device_probe never samples.

device_probe was sized to run on every card we might rent, which capped it well below what the
workloads actually do. Measured over 1089 ms of aten-level device time across the eight configs,
18.8% of that time sits in shapes ABOVE the probe's grid:

    gemm    28.8% of step time    in-grid 14.0%   below 2.2%   ABOVE 12.6%
    conv    17.2% of step time    in-grid  0.6%   below 5.6%   ABOVE  6.2%
    attn     6.9% of step time    in-grid  0.0%   below 6.9%   above  0.0%

Only gemm and conv are extended here. attn's shortfall is entirely BELOW the grid, which is the
launch-bound regime where cost is roughly constant and a count x floor model already covers it,
and bmm is 0.0% of workload time because the aten::matmul calls dispatch as plain GEMMs.

Two specific gaps:
  gemm    GEMM_M tops out at 8192 and the random draws at 2**13, while convnext dispatches
          addmm at M=100352 -- twelve times the ceiling. N and K need nothing: they are model
          hidden dims and the existing 128-4096 span already covers them.
  conv    there is no conv1d arm at all, and wav2vec2's feature extractor is seven conv1d
          layers over 480k samples. The 2D arm is separately capped at bs*cin*hw**2 <= 6e7.

This is deliberately NOT merged into device_probe. Every card screens against the frozen v1
reference, which was built from probes that do not contain these rows; adding them to the probe
would strand the 253 existing probes and change what every published ratio means. The extension
is written to its own file per card and merged into a v2 reference offline, once collected.

A second, unplanned benefit: the current grid does not saturate the big dies. A 5090 runs the
whole 10-minute probe at 50% of its power cap and never leaves 67C, so sustained-load defects
cannot show up there. These shapes do fill those cards.
"""
import random

import torch

from .device_probe import (BWD_ITER_DIV, DIRECTIONS, DTYPES, SHAPE_SEED, _conv_point,
                           _gemm_point, _iters_for, _point_result, _rng, _logdraw, device_meta,
                           _ClockSampler)
from .profiler import time_fn

EXT_VERSION = "v2"

# A static fp32 byte estimate, applied identically on every card: the row set is a property of
# the grid, not of the device, so a 5090 and a 4060 attempt exactly the same shapes and a shape
# too big for 8 GB is recorded as {"oom": true} rather than skipped silently or crashed on. That
# makes the small cards' ceiling visible in the data instead of hidden in the grid definition.
# The budget is therefore not sized to the smallest card -- it only bounds runtime and keeps
# absurd shapes out. In practice it is not the binding constraint at all: the shape ladder is.
# Measured on a 4070 SUPER, the full arm peaks at 4.66 GB allocated and produces zero OOM rows
# even with the process capped at the 7.2 GB an 8 GB card leaves after its context, so every
# eligible model returns the same 381 rows. Raising the cap from 4 GB to 8 GB added 21 rows;
# widening coverage further means adding shapes, not budget.
MEM_BUDGET_BYTES = 8.0e9
FP32 = 4
# Operands + their grads + grad_output. Calibrated, not derived: a first pass at 2.2 predicted
# 3.41 GB for the widest shape where the run actually peaked at 4.42 GB allocated / 4.93 GB
# reserved, because autograd retains buffers the naive count misses and the CUDA graph holds a
# private pool. 2.9 makes the estimate track the measurement, so the budget above means what it
# says. Re-measure if the arm gains a shape family.
GRAD_FACTOR = 2.9

# Heavy shapes run for milliseconds, so the timing loop is much shorter than device_probe's
# (3, 5, 20) -- twenty iterations of a 100 ms GEMM would be 2 s per capture.
EXT_TIMING = (2, 3, 2)

# (N, K) hidden dims, as in device_probe. (512, 128) is convnext's own: its addmm is
# [100352, 128] @ [128, 512], the single largest M the workloads emit.
GEMM_EXT_HIDDEN = [(768, 768), (3072, 768), (768, 3072), (512, 128), (1536, 576)]
# Picks up where GEMM_M stops. 100352 is not a round number because it is not a synthetic
# choice -- it is the exact M convnext_base dispatches at batch 16.
GEMM_EXT_M = [16384, 32768, 65536, 100352, 131072]

# wav2vec2-base's feature extractor at 30 s of 16 kHz audio: conv1d(1->512, k=10, s=5) then
# 4x (512->512, k=3, s=2) then 2x (512->512, k=2, s=2), with the length each layer actually
# sees. Batch 4 rather than 8 keeps the widest activation under budget in fp32.
# (bs, cin, cout, length, k, stride)
CONV1D_SHAPES = [(4, 1, 512, 480000, 10, 5),
                 (4, 512, 512, 95999, 3, 2),
                 (4, 512, 512, 47999, 3, 2),
                 (4, 512, 512, 23999, 3, 2),
                 (4, 512, 512, 11999, 3, 2),
                 (4, 512, 512, 5999, 2, 2),
                 (4, 512, 512, 2999, 2, 2),
                 (8, 1, 512, 160000, 10, 5),      # 10 s clip, larger batch
                 (8, 512, 512, 31999, 3, 2)]

# Activation volumes of 1e8 elements and up, against the 2D arm's 6e7 ceiling.
# (bs, cin, cout, hw, k, stride, pad, groups)
CONV2D_EXT_SHAPES = [(32, 64, 64, 224, 3, 1, 1, 1),
                     (64, 128, 128, 112, 3, 1, 1, 1),
                     (128, 256, 256, 56, 3, 1, 1, 1),
                     (32, 128, 256, 224, 1, 1, 0, 1),
                     (64, 384, 384, 56, 3, 1, 1, 384),
                     (32, 3, 96, 448, 7, 2, 3, 1)]
CONV_EXT_DIRECTIONS = ("fprop", "dgrad", "wgrad")

N_RANDOM_EXT = 16


def _random_gemm_ext_shapes() -> list:
    """Large-M draws off the power-of-two grid.

    The ladder above is all round numbers, and a GEMM's time is not smooth in M: it steps as the
    tile grid gains a row and as the last wave goes partial. Sampling M log-uniformly between
    2**13 and 2**17 -- snapped to 16, not to a power of two -- is what makes the extension able
    to say anything about the M=100352-shaped shapes the workloads actually dispatch, rather
    than only about the ones a benchmark would choose."""
    r, out = _rng("gemm_ext"), []
    while len(out) < N_RANDOM_EXT:
        m = _logdraw(r, 13, 17, 16)                        # 8192 .. 131072
        n, k = (_logdraw(r, 7, 12, 64) for _ in range(2))   # 128 .. 4096
        if _gemm_bytes(m, n, k, grad=True) <= MEM_BUDGET_BYTES:
            out.append((m, n, k))
    return out


def _gemm_bytes(m: int, n: int, k: int, grad: bool) -> float:
    elems = m * k + k * n + m * n
    return elems * FP32 * (GRAD_FACTOR if grad else 1.0)


# The conv arms never build an autograd graph: fprop, dgrad and wgrad are three separate forward
# calls, each given x, w and gy up front (see _conv_point). So GRAD_FACTOR does not apply to them
# -- only cuDNN's algorithm workspace, which is small beside the tensors themselves.
CONV_WORKSPACE_FACTOR = 1.3


def _conv1d_bytes(bs: int, cin: int, cout: int, length: int, k: int, stride: int) -> float:
    out_len = (length - k) // stride + 1
    elems = bs * cin * length + bs * cout * out_len + cout * cin * k
    return elems * FP32 * CONV_WORKSPACE_FACTOR


def _conv2d_bytes(bs: int, cin: int, cout: int, hw: int, k: int, stride: int, pad: int,
                  groups: int) -> float:
    out_hw = (hw + 2 * pad - k) // stride + 1
    elems = bs * cin * hw * hw + bs * cout * out_hw * out_hw + cout * (cin // groups) * k * k
    return elems * FP32 * CONV_WORKSPACE_FACTOR


GEMM_EXT_RANDOM = _random_gemm_ext_shapes()


def _conv1d_point(bs: int, cin: int, cout: int, length: int, k: int, stride: int,
                  direction: str, dt: torch.dtype, warm: int, rep: int, iters: int) -> float:
    """Time one 1D convolution, mirroring _conv_point's fprop/dgrad/wgrad split."""
    x = torch.randn(bs, cin, length, device="cuda", dtype=dt)
    w = torch.randn(cout, cin, k, device="cuda", dtype=dt)
    out_len = (length - k) // stride + 1
    gy = torch.randn(bs, cout, out_len, device="cuda", dtype=dt)
    torch.cuda.synchronize()

    if direction == "fprop":
        fn = lambda: torch.nn.functional.conv1d(x, w, stride=stride)
    elif direction == "dgrad":
        fn = lambda: torch.nn.grad.conv1d_input(x.shape, w, gy, stride=stride)
    elif direction == "wgrad":
        fn = lambda: torch.nn.grad.conv1d_weight(x, w.shape, gy, stride=stride)
    else:
        raise ValueError(f"direction must be fprop|dgrad|wgrad, got {direction!r}")
    return time_fn(fn, warm, rep, iters)


def probe_gemm_ext(rows: list, dt: torch.dtype, name: str) -> None:
    """Matmul above device_probe's M ceiling."""
    warm, rep, iters = EXT_TIMING
    shapes = [(m, n, k) for n, k in GEMM_EXT_HIDDEN for m in GEMM_EXT_M] + GEMM_EXT_RANDOM
    for m, n, k in shapes:
        size = f"m{m}n{n}k{k}"
        for d in DIRECTIONS:
            if _gemm_bytes(m, n, k, grad=(d == "fwd_bwd")) > MEM_BUDGET_BYTES:
                continue                      # skipped identically on every card
            rows.append(_point_result(
                {"probe": "gemm_ext", "dtype": name, "size": size, "direction": d},
                lambda m=m, n=n, k=k, d=d: _gemm_point(m, n, k, dt, d, warm, rep, _iters_for(d, iters))))


def probe_conv1d(rows: list, dt: torch.dtype, name: str) -> None:
    """1D convolution -- an op family device_probe does not measure at all."""
    warm, rep, iters = EXT_TIMING
    for bs, cin, cout, length, k, stride in CONV1D_SHAPES:
        size = f"b{bs}c{cin}o{cout}l{length}k{k}s{stride}"
        for d in CONV_EXT_DIRECTIONS:
            if _conv1d_bytes(bs, cin, cout, length, k, stride) > MEM_BUDGET_BYTES:
                continue
            rows.append(_point_result(
                {"probe": "conv1d", "dtype": name, "size": size, "direction": d},
                lambda bs=bs, cin=cin, cout=cout, length=length, k=k, stride=stride, d=d:
                    _conv1d_point(bs, cin, cout, length, k, stride, d, dt, warm, rep, iters)))


def probe_conv_ext(rows: list, dt: torch.dtype, name: str) -> None:
    """2D convolution above the activation-volume cap device_probe stops at."""
    warm, rep, iters = EXT_TIMING
    for bs, cin, cout, hw, k, stride, pad, groups in CONV2D_EXT_SHAPES:
        size = f"b{bs}c{cin}o{cout}hw{hw}k{k}s{stride}p{pad}g{groups}"
        for d in CONV_EXT_DIRECTIONS:
            if _conv2d_bytes(bs, cin, cout, hw, k, stride, pad, groups) > MEM_BUDGET_BYTES:
                continue
            rows.append(_point_result(
                {"probe": "conv_ext", "dtype": name, "size": size, "direction": d},
                lambda bs=bs, cin=cin, cout=cout, hw=hw, k=k, stride=stride, pad=pad, groups=groups, d=d:
                    _conv_point(bs, cin, cout, hw, k, stride, pad, groups, d, dt, warm, rep, iters)))


def run_extension(dtypes: dict | None = None) -> dict:
    """Measure the v2 extension and return it as a dict, shaped like a device_probe signature.

    Same {"device": ..., "probes": [...]} envelope so the offline merge into a v2 reference can
    treat extension rows and probe rows identically -- the row key already carries the arm name,
    so gemm_ext rows cannot collide with gemm rows."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")

    sampler = _ClockSampler()
    sampler.start()
    rows: list = []
    try:
        for name, dt in (dtypes or DTYPES).items():
            probe_gemm_ext(rows, dt, name)
            probe_conv1d(rows, dt, name)
            probe_conv_ext(rows, dt, name)
    finally:
        clocks = sampler.stop()

    return {"extension_version": EXT_VERSION,
            "device": {**device_meta(), **clocks},
            "probes": rows}


def main() -> None:
    import sys
    import time
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")
    t0 = time.time()
    ext = run_extension()
    print(f"\n  {'probe':10s}{'dt':>6s}{'size':>34s}{'dir':>9s}{'ms':>11s}")
    for r in ext["probes"]:
        head = f"  {r['probe']:10s}{r['dtype']:>6s}{r['size']:>34}{r.get('direction',''):>9s}"
        print(f"{head}{'OOM':>11s}" if r.get("oom") else f"{head}{r['ms']:>11.4f}")
    n_oom = sum(1 for r in ext["probes"] if r.get("oom"))
    print(f"\n{len(ext['probes'])} rows, {n_oom} OOM, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
