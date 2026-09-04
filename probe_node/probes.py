#!/usr/bin/env python3

"""The hardware signature: every synthetic kernel this node times, in one file.

Sixteen families, swept in fp32, identified by (probe, dtype, size, direction, kind). Four
groups, by what they are for rather than when they were added:

  core          gemm, bmm, conv, attn, elementwise, pool, rnn
                The shapes a step is mostly made of, sized to run on any card worth renting.

  large shape   gemm_ext, conv_ext, conv1d
                Where the core grid stops but the workloads do not. Measured over 1089 ms of
                aten-level device time across eight configs, 18.8% of it sits ABOVE the core grid:

                    gemm    28.8% of step time    in-grid 14.0%   below 2.2%   ABOVE 12.6%
                    conv    17.2% of step time    in-grid  0.6%   below 5.6%   ABOVE  6.2%
                    attn     6.9% of step time    in-grid  0.0%   below 6.9%   above  0.0%

                Only gemm and conv needed extending. attn's shortfall is entirely BELOW the grid,
                the launch-bound regime a count x floor model already covers, and bmm is 0.0% of
                workload time because the aten::matmul calls dispatch as plain GEMMs. Two gaps
                specifically: the core M ladder tops out at 8192 while convnext dispatches addmm
                at M=100352, and there was no conv1d at all though wav2vec2's feature extractor
                is seven conv1d layers over 480k samples.

  per layer     activation, norm, dropout, reduction
                The costs between the matmuls -- individually small, collectively the part of a
                step that gemm and conv timings cannot account for.

  per step      loss, optimizer
                Once per step rather than once per layer, and cheap enough that their floor
                matters more than their scaling.

ADDING A FAMILY. Three steps, all local to this file:

  1. shape constants and, if the shapes are drawn rather than literal, a `_random_*_shapes()`
     helper seeded through `_rng("<family>")` -- its own stream, so adding a family cannot shift
     another family's draws.
  2. a `_<family>_point(...)` that builds the tensors and returns `time_fn(...)` ms, and a
     `probe_<family>(rows, dt, name)` that walks the shapes and appends `_point_result(...)`.
  3. the probe function into PROBES.

  Adding a family only adds rows. Changing an EXISTING family's shapes, size strings or kinds
  changes what its rows mean, and anything comparing against previously collected rows for that
  family has to be rebuilt -- so prefer a new family over widening an old one.

Every row is one of three outcomes and never more: a `ms` timing, `oom` true, or an `error`
string. See _point_result for why a failure is recorded rather than raised.
"""
import json
import os
import platform
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.grad          # submodule; `import torch` alone does not pull it in
from profiler import time_fn

SEP = "# " + "-" * 117

WARMUP, REPEATS = 5, 10

# CUDA cores per SM by compute capability; neither torch nor NVML reports it.
CORES_PER_SM = {(3, 0): 192, (3, 5): 192, (3, 7): 192,
                (5, 0): 128, (5, 2): 128, (5, 3): 128,
                (6, 0): 64, (6, 1): 128, (6, 2): 128,
                (7, 0): 64, (7, 2): 64, (7, 5): 64,
                (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
                (9, 0): 128, (10, 0): 128, (12, 0): 128}

# repeats stays at 5 here while the workloads use 10: the probe is one atomic artefact
# competing against container lifetime, and doubling it measurably cut how many nodes
# produced one at all. The workloads can afford 10 because they checkpoint.
# fp32 only. The mixed-precision dtypes were dropped deliberately, not for cost: on a probe
# they measure the tensor cores and the cast machinery as much as the kernel, and which of the
# two dominates moves with the shape -- so an fp16 row is a blend whose mixture is not a property
# of the hardware. fp32 is one path through the SM for every family here, which is what makes a
# row comparable between shapes and between cards.
#
# Widening this back out is a one-line change and every family already sweeps whatever is in it;
# nothing below assumes a single dtype. But rows collected under a widened DTYPES are not
# comparable with rows collected under this one at the same (probe, size) -- the dtype is part of
# the row key precisely so that mixing them is visible rather than silent.
DTYPES = {"fp32": torch.float32}

# (N, K) hidden dims x an M ladder. The workloads never dispatch a square matmul: N and K
# are model hidden dims while M is the token count (batch x seq, or batch x H x W), and

# ---- core shapes ------------------------------------------------------------------------------------------------
GEMM_HIDDEN = [(768, 768), (3072, 768), (768, 3072), (1536, 576), (49152, 576)]
GEMM_M = [80, 256, 512, 2048, 3152, 8192]
GEMM_SQUARE = [1024, 2048, 4096]
GEMM_TIMING = (3, 5, 20)

# (batch, M, N, K). Not recoverable from the gemm family: at equal total FLOPs the time
# swings ~3x with how the work splits between batch and matrix dims. The first three rows
# are exactly that split held at 1.07 GFLOP; then both extremes of it; then the QK^T and
# AV shapes attention takes when it does not route through SDPA.
BMM_SHAPES = [(256, 128, 128, 128),
              (32, 256, 256, 256),
              (4, 512, 512, 512),
              (1024, 64, 64, 64),
              (2, 1024, 1024, 1024),
              (192, 197, 197, 64),
              (192, 197, 64, 197),
              (36, 512, 512, 64),
              (36, 512, 64, 512)]
BMM_TIMING = (3, 5, 20)

ELEMENTWISE_NUMEL = [2 ** 19, 2 ** 21, 2 ** 23, 2 ** 25, 2 ** 26, 2 ** 27, 2 ** 28]
ELEMENTWISE_TIERS = [(2 ** 23, 3, 5, 200), (2 ** 26, 3, 5, 20), (2 ** 63, 3, 5, 3)]

# (bs, cin, cout, hw, k, stride, pad, groups); pad is explicit because patch embedding
# needs pad=0 at k=16, which k//2 cannot express.
CONV_SHAPES = [(16, 3, 64, 224, 7, 2, 3, 1),
               (16, 3, 768, 224, 16, 16, 0, 1),
               (16, 64, 64, 56, 3, 1, 1, 1),
               (16, 64, 256, 56, 1, 1, 0, 1),
               (16, 128, 256, 28, 3, 2, 1, 1),
               (16, 384, 384, 28, 3, 1, 1, 384),
               (16, 256, 256, 14, 3, 1, 1, 1),
               (16, 512, 512, 7, 3, 1, 1, 1),
               (16, 512, 2048, 7, 1, 1, 0, 1)]
CONV_DIRECTIONS = ("fprop", "dgrad", "wgrad")
CONV_TIMING = (3, 5, 20)

# (bs, q_heads, kv_heads, seq_q, seq_kv, head_dim). q_heads != kv_heads is GQA and
# seq_q != seq_kv is cross-attention; both are dispatched by the workloads and neither
# is reachable from a single (bs, heads, seq, dim) tuple.
ATTN_SHAPES = [(16, 12, 12, 197, 197, 64),
               (16, 12, 12, 577, 577, 64),
               (4, 9, 3, 256, 256, 64),
               (4, 9, 3, 512, 512, 64),
               (8, 12, 12, 1024, 1024, 64),
               (8, 12, 12, 2048, 2048, 64),
               (2, 32, 8, 2048, 2048, 128),
               (2, 8, 8, 4096, 4096, 40),
               (2, 8, 8, 4096, 77, 40)]
ATTN_TIMING = (3, 5, 10)

# (bs, c, hw, k, stride) x max/avg/adaptive_avg. 64c56 and 256c28 are gone: their times
# did not move with dtype, so they were reporting a launch floor rather than the op.
POOL_SHAPES = [(16, 64, 112, 3, 2), (16, 2048, 7, 3, 1)]
POOL_KINDS = ("max", "avg", "adaptive_avg")
POOL_TIMING = (3, 5, 50)

# (bs, seq, input, hidden, layers, bidirectional) x lstm/gru/rnn. One axis moves per row
# off a fixed baseline. No workload uses a recurrent layer, so unlike the other families these
# are conventional shapes rather than traced ones. seq is a sequential dependency chain,
# not parallel work -- the one axis in the probe that rewards clock over SM count.
RNN_SHAPES = [(32, 128, 512, 512, 1, False),
              (32, 128, 512, 512, 3, False),
              (32, 512, 512, 512, 1, False),
              (32, 128, 512, 1024, 1, False),
              (32, 128, 512, 512, 1, True),
              (8, 128, 512, 512, 1, False),
              (128, 128, 512, 512, 1, False)]
RNN_KINDS = ("lstm", "gru", "rnn")
RNN_TIMING = (3, 5, 10)

DIRECTIONS = ("fwd", "fwd_bwd")
# a captured fwd_bwd iteration allocates fresh activations and grads that the plain forward
# does not, so it gets a shorter loop to keep peak memory near the forward's
BWD_ITER_DIV = 4

SHAPE_SEED = 420
N_RANDOM = 50

# ---- large shapes -----------------------------------------------------------------------------------------------
# A static fp32 byte estimate, applied identically on every card: the row set is a property of
# the grid, not of the device, so a 5090 and a 4060 attempt exactly the same shapes and a shape
# too big for 8 GB is recorded as {"oom": true} rather than skipped silently or crashed on. That
# makes the small cards' ceiling visible in the data instead of hidden in the grid definition.
# The budget is therefore not sized to the smallest card -- it only bounds runtime and keeps
# absurd shapes out. In practice it is not the binding constraint at all: the shape ladder is.
# Measured on a 4070 SUPER, these peak at 4.66 GB allocated and produce zero OOM rows
# even with the process capped at the 7.2 GB an 8 GB card leaves after its context, so every
# eligible model returns the same 381 rows. Raising the cap from 4 GB to 8 GB added 21 rows;
# widening coverage further means adding shapes, not budget.
MEM_BUDGET_BYTES = 8.0e9
FP32 = 4
# Operands + their grads + grad_output. Calibrated, not derived: a first pass at 2.2 predicted
# 3.41 GB for the widest shape where the run actually peaked at 4.42 GB allocated / 4.93 GB
# reserved, because autograd retains buffers the naive count misses and the CUDA graph holds a
# private pool. 2.9 makes the estimate track the measurement, so the budget above means what it
# says. Re-measure if these gain a shape family.
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

# Activation volumes of 1e8 elements and up, against the core conv grid's 6e7 ceiling.
# (bs, cin, cout, hw, k, stride, pad, groups)
CONV2D_EXT_SHAPES = [(32, 64, 64, 224, 3, 1, 1, 1),
                     (64, 128, 128, 112, 3, 1, 1, 1),
                     (128, 256, 256, 56, 3, 1, 1, 1),
                     (32, 128, 256, 224, 1, 1, 0, 1),
                     (64, 384, 384, 56, 3, 1, 1, 384),
                     (32, 3, 96, 448, 7, 2, 3, 1)]
CONV_EXT_DIRECTIONS = ("fprop", "dgrad", "wgrad")

N_RANDOM_EXT = 16


# ---- per-layer and per-step shapes --------------------------------------------------------------------------------
# The costs a step is built from that the matmul and conv families do not measure: the activation
# and normalisation between every conv, the regularisation around them, the reductions FX finds
# outside SDPA, the loss at the end, and the optimizer that follows. Individually small, and
# collectively the part of a step that gemm/conv timings cannot account for.

# Activation and dropout share this ladder, so the two are directly comparable at every point on
# it. It reaches further down than ELEMENTWISE_NUMEL (256 against 2**19) because the small end is
# where the launch floor dominates, and that floor is what a per-kind cost has to be read against.
POINTWISE_NUMEL = [256, 1024, 4096, 16384, 65536, 262144, 1048576,
                   4194304, 16777216, 67108864]

# Deliberately separate from the elementwise family's sigmoid rather than proxied by it. Sigmoid is
# one transcendental with one cost; gelu is an erf, silu a sigmoid-multiply, mish a
# softplus-tanh, and hardswish pure arithmetic with no transcendental at all. Their backward
# passes diverge further still. One of them cannot stand in for the others.
ACTIVATION_KINDS = {
    "relu": torch.nn.functional.relu,
    "relu6": torch.nn.functional.relu6,
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
    "hardswish": torch.nn.functional.hardswish,
    "mish": torch.nn.functional.mish,
}

# Both dropout kinds are shaped (DROPOUT_BATCH, numel // DROPOUT_BATCH) rather than flat. A flat
# tensor has no sample axis for drop_path to broadcast a mask over, and giving the two kinds
# different layouts would leave them measuring different things. Every value in POINTWISE_NUMEL
# divides by 16.
DROPOUT_BATCH = 16
DROPOUT_P = 0.1
DROPOUT_KINDS = ("dropout", "drop_path")

# (batch, C, H, W) -- the resnet/convnext stage widths, at the batch the vision sweep runs.
BATCHNORM_SHAPES = [(16, 32, 112, 112), (16, 64, 112, 112),
                    (16, 64, 56, 56),   (16, 128, 56, 56),
                    (16, 128, 28, 28),  (16, 256, 28, 28),
                    (16, 256, 14, 14),  (16, 512, 14, 14),
                    (16, 512, 7, 7),    (16, 768, 14, 14),
                    (16, 1024, 7, 7),   (16, 2048, 7, 7)]

# (rows, hidden). rows is batch x tokens already flattened, which is how a norm sees a
# transformer activation; hidden is the model dim.
LAYERNORM_SHAPES = [(200704, 96), (50176, 128), (50176, 192),
                    (12544, 256), (12544, 384), (3152, 768),
                    (3152, 1024), (1040, 768), (272, 768),
                    (1040, 1024), (272, 512), (3152, 1536)]

# (batch, C, H, W, groups). 32 groups is the common default; 1 group is normalisation over all
# channels at once, which is what convnext's "LayerNorm2d" actually dispatches.
GROUPNORM_SHAPES = [(16, 64, 56, 56, 32), (16, 128, 28, 28, 32),
                    (16, 256, 14, 14, 32), (16, 512, 7, 7, 32),
                    (16, 256, 14, 14, 1),  (16, 512, 7, 7, 1)]

# rmsnorm shares layernorm's grid and layout -- one fewer reduction, and a weight but no bias --
# but keeps its own kind, so the difference between them is measured rather than assumed.
NORM_KINDS = ("batchnorm2d", "layernorm", "groupnorm", "rmsnorm")
NORM_TIMING = (3, 5, 20)

# (rows, width). The standalone softmax and reduction paths FX finds OUTSIDE SDPA; the attn family
# already covers the fused kernel, and these are what it does not reach.
REDUCTION_SHAPES = [(16, 16), (256, 64), (768, 197), (768, 256),
                    (192, 512), (192, 1024), (64, 2048), (32, 4096)]
REDUCTION_KINDS = ("softmax", "log_softmax", "mean_last")
REDUCTION_TIMING = (3, 5, 50)

# (batch, classes). Small and launch-bound on purpose: the loss is a rounding error in step time,
# and a floor is worth knowing exactly rather than estimating.
LOSS_SHAPES = [(16, 16), (32, 16), (64, 16)]
LOSS_TIMING = (3, 5, 50)

# Optimizer cost scales with parameter COUNT and with total size independently -- 512 small
# tensors and 32 large ones at the same element count are entirely different launch profiles --
# so the two are swept as a grid rather than along one axis.
OPTIMIZER_TENSORS = [32, 128, 512]
OPTIMIZER_NUMEL = [1048576, 4194304, 16777216, 67108864, 268435456]
OPTIMIZER_KINDS = ("zero_grad", "sgd_momentum")
OPTIMIZER_TIMING = (3, 5, 5)

# ---- shape draws ------------------------------------------------------------------------------------------------
def _rng(family: str) -> random.Random:
    """One stream per family, so adding or reordering families cannot shift another's draws.
    Seeding from a str goes through sha512, which is stable across runs and platforms
    (unlike hash(), which PYTHONHASHSEED randomises)."""
    return random.Random(f"{SHAPE_SEED}:{family}")


def _logdraw(r: random.Random, lo: float, hi: float, mult: int) -> int:
    """Draw uniformly in the exponent, snapped to a multiple of mult."""
    return max(mult, int(round(2 ** r.uniform(lo, hi) / mult)) * mult)


def _random_gemm_shapes() -> list:
    r, out = _rng("gemm"), []
    while len(out) < N_RANDOM:
        m = _logdraw(r, 6, 13, 16)                        # 64 .. 8192
        n, k = (_logdraw(r, 7, 12, 64) for _ in range(2))  # 128 .. 4096
        if 2 * m * n * k <= 40e9:
            out.append((m, n, k))
    return out


def _random_bmm_shapes() -> list:
    r, out = _rng("bmm"), []
    while len(out) < N_RANDOM:
        b = r.choice([2, 4, 8, 16, 32, 64, 128, 256, 512])
        m, n, k = (_logdraw(r, 5, 10, 32) for _ in range(3))   # 32 .. 1024
        if 2 * b * m * n * k <= 20e9:
            out.append((b, m, n, k))
    return out


def _random_conv_shapes() -> list:
    r, out = _rng("conv"), []
    while len(out) < N_RANDOM:
        k, stride = r.choice([1, 3, 5, 7]), r.choice([1, 1, 2])
        hw = r.choice([7, 14, 28, 56, 112, 224])
        cin = _logdraw(r, 5, 10, 32)           # 32 .. 1024
        depthwise = r.random() < 0.25          # groups must divide cin and cout
        cout, groups = (cin, cin) if depthwise else (_logdraw(r, 5, 10, 32), 1)
        bs = r.choice([8, 16, 32])
        if bs * cin * hw * hw <= 6e7:
            out.append((bs, cin, cout, hw, k, stride, k // 2, groups))
    return out


def _random_attn_shapes() -> list:
    r, out = _rng("attn"), []
    while len(out) < N_RANDOM:
        d = r.choice([32, 64, 80, 128])
        qh = r.choice([4, 8, 12, 16, 32])
        kvh = r.choice([h for h in (1, 2, 4, 8, qh) if qh % h == 0])
        seq = r.choice([128, 197, 256, 384, 512, 1024, 2048, 4096])
        bs = r.choice([1, 2, 4, 8, 16])
        if bs * qh * seq * seq * d <= 4e9:     # seq_q == seq_kv, so causal stays meaningful
            out.append((bs, qh, kvh, seq, seq, d))
    return out


def _random_rnn_shapes() -> list:
    r, out = _rng("rnn"), []
    while len(out) < N_RANDOM:
        inp, hidden = (_logdraw(r, 7, 10.58, 128) for _ in range(2))   # 128 .. 1536
        seq = r.choice([32, 64, 128, 256, 512])
        bs = r.choice([8, 16, 32, 64, 128])
        layers, bidir = r.choice([1, 1, 2, 3]), r.random() < 0.3
        if bs * seq * hidden * layers <= 4e7:
            out.append((bs, seq, inp, hidden, layers, bidir))
    return out

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


# The conv families never build an autograd graph: fprop, dgrad and wgrad are three separate forward
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

GEMM_RANDOM = _random_gemm_shapes()
BMM_SHAPES += _random_bmm_shapes()
CONV_SHAPES += _random_conv_shapes()
ATTN_SHAPES += _random_attn_shapes()
RNN_SHAPES += _random_rnn_shapes()

GEMM_EXT_RANDOM = _random_gemm_ext_shapes()

# -------------------------------------------------------------------------------------------------------------------


def fwd_bwd_fn(fwd, inputs: list, warm: int, rep: int, iters: int) -> float:
    """Median per-iteration ms for one forward plus its backward for function fn."""
    def step():
        out = fwd()
        if isinstance(out, tuple):
            out = out[0]
        torch.autograd.grad(out, inputs, torch.ones_like(out))
    return time_fn(step, warm, rep, iters)


def _iters_for(direction: str, iters: int) -> int:
    return iters if direction == "fwd" else max(1, iters // BWD_ITER_DIV)

# -------------------------------------------------------------------------------------------------------------------

def _gemm_point(m: int, n: int, k: int, dt: torch.dtype, direction: str, warm: int,
                rep: int, iters: int) -> float:
    """Allocate, time and release one (m x k) @ (k x n) GEMM."""
    grad = direction == "fwd_bwd"
    a = torch.randn(m, k, device="cuda", dtype=dt, requires_grad=grad)
    b = torch.randn(k, n, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: torch.mm(a, b), [a, b], warm, rep, iters)
    c = torch.empty(m, n, device="cuda", dtype=dt)   # out= is unavailable under autograd
    return time_fn(lambda: torch.mm(a, b, out=c), warm, rep, iters)


def _bmm_point(batch: int, m: int, n: int, k: int, dt: torch.dtype, direction: str,
               warm: int, rep: int, iters: int) -> float:
    """Allocate, time and release one batched (m x k) @ (k x n) matmul."""
    grad = direction == "fwd_bwd"
    a = torch.randn(batch, m, k, device="cuda", dtype=dt, requires_grad=grad)
    b = torch.randn(batch, k, n, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: torch.bmm(a, b), [a, b], warm, rep, iters)
    c = torch.empty(batch, m, n, device="cuda", dtype=dt)
    return time_fn(lambda: torch.bmm(a, b, out=c), warm, rep, iters)


def _conv_point(bs: int, cin: int, cout: int, hw: int, k: int, stride: int, pad: int,
                groups: int, direction: str, dt: torch.dtype, warm: int, rep: int,
                iters: int) -> float:
    """Time one 2D convolution."""
    x = torch.randn(bs, cin, hw, hw, device="cuda", dtype=dt)
    w = torch.randn(cout, cin // groups, k, k, device="cuda", dtype=dt)
    out_hw = (hw + 2 * pad - k) // stride + 1
    gy = torch.randn(bs, cout, out_hw, out_hw, device="cuda", dtype=dt)
    torch.cuda.synchronize()

    if direction == "fprop":
        fn = lambda: torch.nn.functional.conv2d(x, w, stride=stride, padding=pad, groups=groups)
    elif direction == "dgrad":
        fn = lambda: torch.nn.grad.conv2d_input(x.shape, w, gy, stride=stride,
                                                padding=pad, groups=groups)
    elif direction == "wgrad":
        fn = lambda: torch.nn.grad.conv2d_weight(x, w.shape, gy, stride=stride,
                                                 padding=pad, groups=groups)
    else:
        raise ValueError(f"direction must be fprop|dgrad|wgrad, got {direction!r}")
    return time_fn(fn, warm, rep, iters)


def _attn_point(bs: int, q_heads: int, kv_heads: int, seq_q: int, seq_kv: int,
                head_dim: int, causal: bool, dt: torch.dtype, direction: str, warm: int,
                rep: int, iters: int) -> float:
    """Time one scaled-dot-product attention, on whichever backend SDPA picks."""
    grad = direction == "fwd_bwd"
    q = torch.randn(bs, q_heads, seq_q, head_dim, device="cuda", dtype=dt, requires_grad=grad)
    k = torch.randn(bs, kv_heads, seq_kv, head_dim, device="cuda", dtype=dt, requires_grad=grad)
    v = torch.randn(bs, kv_heads, seq_kv, head_dim, device="cuda", dtype=dt, requires_grad=grad)
    gqa = q_heads != kv_heads
    fn = lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=causal, enable_gqa=gqa)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(fn, [q, k, v], warm, rep, iters)
    return time_fn(fn, warm, rep, iters)


def _pool_point(bs: int, c: int, hw: int, k: int, stride: int, kind: str,
                dt: torch.dtype, direction: str, warm: int, rep: int,
                iters: int) -> float:
    """Time one 2D pooling op."""
    grad = direction == "fwd_bwd"
    x = torch.randn(bs, c, hw, hw, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if kind == "max":
        fn = lambda: torch.nn.functional.max_pool2d(x, k, stride, k // 2)
    elif kind == "avg":
        fn = lambda: torch.nn.functional.avg_pool2d(x, k, stride, k // 2)
    elif kind == "adaptive_avg":
        fn = lambda: torch.nn.functional.adaptive_avg_pool2d(x, 1)
    else:
        raise ValueError(f"kind must be max|avg|adaptive_avg, got {kind!r}")
    if grad:
        return fwd_bwd_fn(fn, [x], warm, rep, iters)
    return time_fn(fn, warm, rep, iters)


def _elementwise_point(numel: int, dt: torch.dtype, direction: str,
                       warm: int, rep: int, iters: int) -> float:
    """Time one activation. Two tensors live, so the working set is 2 x numel."""
    grad = direction == "fwd_bwd"
    a = torch.randn(numel, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: torch.sigmoid(a), [a], warm, rep, iters)
    c = torch.empty(numel, device="cuda", dtype=dt)
    return time_fn(lambda: torch.sigmoid(a, out=c), warm, rep, iters)

def _rnn_point(bs: int, seq: int, inp: int, hidden: int, layers: int, bidir: bool,
               kind: str, dt: torch.dtype, direction: str, warm: int, rep: int,
               iters: int) -> float:
    """Time one recurrent stack on cuDNN's fused kernel."""
    cls = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU, "rnn": torch.nn.RNN}.get(kind)
    if cls is None:
        raise ValueError(f"kind must be lstm|gru|rnn, got {kind!r}")
    grad = direction == "fwd_bwd"
    m = cls(inp, hidden, num_layers=layers, batch_first=True,
            bidirectional=bidir, dropout=0.0).cuda().to(dt)
    m.flatten_parameters()      # cuDNN needs the weights in one contiguous buffer
    x = torch.randn(bs, seq, inp, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: m(x), [x] + list(m.parameters()), warm, rep, iters)
    m.eval()
    with torch.no_grad():
        return time_fn(lambda: m(x), warm, rep, iters)


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


def _activation_point(numel: int, kind: str, dt: torch.dtype,
                      warm: int, rep: int, iters: int) -> float:
    """Time one activation's forward and its backward."""
    fn = ACTIVATION_KINDS.get(kind)
    if fn is None:
        raise ValueError(f"kind must be one of {sorted(ACTIVATION_KINDS)}, got {kind!r}")
    a = torch.randn(numel, device="cuda", dtype=dt, requires_grad=True)
    torch.cuda.synchronize()
    return fwd_bwd_fn(lambda: fn(a), [a], warm, rep, iters)


def _dropout_point(numel: int, kind: str, dt: torch.dtype,
                   warm: int, rep: int, iters: int) -> float:
    """Time one dropout variant at DROPOUT_P, in training mode.

    Shaped (DROPOUT_BATCH, -1) rather than flat because drop_path needs a sample axis; see
    DROPOUT_BATCH. Both kinds get the same shape so the only difference between them is the
    granularity of the mask, which is the thing being measured."""
    x = torch.randn(DROPOUT_BATCH, numel // DROPOUT_BATCH, device="cuda", dtype=dt,
                    requires_grad=True)
    keep = 1.0 - DROPOUT_P

    if kind == "dropout":
        fwd = lambda: torch.nn.functional.dropout(x, p=DROPOUT_P, training=True)
    elif kind == "drop_path":
        # Stochastic depth as timm implements it: ONE Bernoulli draw per sample, broadcast over
        # every other dimension, scaled by keep_prob. Aliasing this to ordinary dropout would
        # draw a mask numel/DROPOUT_BATCH times larger and measure the wrong kernel entirely.
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        def fwd():
            mask = x.new_empty(mask_shape).bernoulli_(keep).div_(keep)
            return x * mask
    else:
        raise ValueError(f"kind must be dropout|drop_path, got {kind!r}")

    torch.cuda.synchronize()
    return fwd_bwd_fn(fwd, [x], warm, rep, iters)


def _norm_point(shape: tuple, kind: str, dt: torch.dtype,
                warm: int, rep: int, iters: int) -> float:
    """Time one normalisation layer in train() mode, with gradients for input and affine params.

    train() rather than eval() because that is what the workloads run and the two are different
    kernels -- batchnorm in particular computes batch statistics and updates its running buffers
    in training mode, and does neither in eval."""
    if kind == "batchnorm2d":
        bs, c, h, w = shape
        module = torch.nn.BatchNorm2d(c)
        x = torch.randn(bs, c, h, w, device="cuda", dtype=dt, requires_grad=True)
    elif kind == "groupnorm":
        bs, c, h, w, groups = shape
        module = torch.nn.GroupNorm(groups, c)
        x = torch.randn(bs, c, h, w, device="cuda", dtype=dt, requires_grad=True)
    elif kind in ("layernorm", "rmsnorm"):
        n_rows, hidden = shape
        module = torch.nn.LayerNorm(hidden) if kind == "layernorm" else torch.nn.RMSNorm(hidden)
        x = torch.randn(n_rows, hidden, device="cuda", dtype=dt, requires_grad=True)
    else:
        raise ValueError(f"kind must be one of {NORM_KINDS}, got {kind!r}")

    module = module.cuda().to(dt).train()
    torch.cuda.synchronize()
    # The gradient set is read off the module rather than assumed, which is what "affine bias
    # where applicable" means in practice: layernorm and the conv norms carry weight and bias,
    # rmsnorm carries weight alone.
    return fwd_bwd_fn(lambda: module(x), [x] + list(module.parameters()), warm, rep, iters)


def _reduction_point(n_rows: int, width: int, kind: str, dt: torch.dtype,
                     warm: int, rep: int, iters: int) -> float:
    """Time one standalone reduction over the last dimension, forward and backward."""
    x = torch.randn(n_rows, width, device="cuda", dtype=dt, requires_grad=True)
    if kind == "softmax":
        fwd = lambda: torch.nn.functional.softmax(x, dim=-1)
    elif kind == "log_softmax":
        fwd = lambda: torch.nn.functional.log_softmax(x, dim=-1)
    elif kind == "mean_last":
        fwd = lambda: x.mean(dim=-1)
    else:
        raise ValueError(f"kind must be one of {REDUCTION_KINDS}, got {kind!r}")
    torch.cuda.synchronize()
    return fwd_bwd_fn(fwd, [x], warm, rep, iters)


def _loss_point(batch: int, classes: int, dt: torch.dtype,
                warm: int, rep: int, iters: int) -> float:
    """Time cross-entropy over integer targets, with the logits carrying the gradient."""
    logits = torch.randn(batch, classes, device="cuda", dtype=dt, requires_grad=True)
    # int64 class indices, not one-hot -- the same thing every workload's criterion is handed.
    target = torch.randint(0, classes, (batch,), device="cuda")
    torch.cuda.synchronize()
    return fwd_bwd_fn(lambda: torch.nn.functional.cross_entropy(logits, target),
                      [logits], warm, rep, iters)


def _split_evenly(total: int, parts: int) -> list[int]:
    """total split across parts, the remainder spread one element at a time over the first few."""
    base, extra = divmod(total, parts)
    return [base + (1 if i < extra else 0) for i in range(parts)]


def _optimizer_point(tensors: int, total_numel: int, kind: str,
                     warm: int, rep: int, iters: int) -> float:
    """Time one optimizer operation over a persistent fp32 parameter set.

    Only the operation itself is inside the timed window. The parameters, their gradients and --
    for sgd_momentum -- the momentum buffers are all materialised first, because a captured graph
    replays against the addresses it recorded: an allocation on the first step would either be
    captured as one-off work that never happens again, or move a buffer under the graph.

    zero_grad uses set_to_none=False deliberately. set_to_none=True frees the gradient tensors,
    which is a deallocation rather than a kernel and would measure nothing on the device."""
    sizes = _split_evenly(total_numel, tensors)
    params = [torch.nn.Parameter(torch.randn(n, device="cuda", dtype=torch.float32))
              for n in sizes]
    for p in params:
        p.grad = torch.randn_like(p)

    if kind == "zero_grad":
        # No momentum: this kind never calls step(), and the buffers would be a third of the
        # working set allocated to be ignored.
        opt = torch.optim.SGD(params, lr=0.01)
        torch.cuda.synchronize()
        return time_fn(lambda: opt.zero_grad(set_to_none=False), warm, rep, iters)

    if kind == "sgd_momentum":
        opt = torch.optim.SGD(params, lr=0.01, momentum=0.9, foreach=None)
        opt.step()                  # materialises the momentum buffers before capture
        torch.cuda.synchronize()
        return time_fn(opt.step, warm, rep, iters)

    raise ValueError(f"kind must be one of {OPTIMIZER_KINDS}, got {kind!r}")


# -------------------------------------------------------------------------------------------------------------------

def _point_result(base: dict, fn) -> dict:
    """Run one probe point; base plus a timing, or a failure recorded instead of raised.

    A single row's exception must never end the sweep. OutOfMemoryError got its own catch
    everywhere already, but that is not the only way a point can fail: a card pushed hard enough
    can leave the CUDA context in a state where the NEXT op raises a plain RuntimeError (a
    corrupted allocator, "device not ready", "an illegal memory access was encountered") rather
    than OutOfMemoryError, and an uncaught one of those crashes run_probe/run_extension outright.
    OOM still gets its own boolean, because "does not fit" is meaningful signal; anything else is
    recorded as text and is not expected to fire in normal operation.

    empty_cache() is called here rather than by each caller, and inside its own try -- not as
    tidiness, but because it is itself a CUDA call. On an RTX 2070/2080 Ti it was observed sitting
    right after this function returns in every probe_* loop, unguarded, and once a point's own
    context-corrupting failure had already been caught and recorded above, THIS call raised the
    same corruption straight past every remaining row and out of run_probe() entirely -- so the
    per-row catch above was doing its job and losing the credit for it one line later. A device
    already this broken cannot be helped by emptying its cache anyway, so the failure is
    swallowed rather than left to escape a second time."""
    try:
        out = {**base, "ms": fn()}
    except torch.cuda.OutOfMemoryError:
        out = {**base, "oom": True}
    except Exception as e:
        out = {**base, "error": f"{type(e).__name__}: {e}"}
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out


def probe_gemm(rows: list, dt: torch.dtype, name: str) -> None:
    """Matmul: an M ladder at each hidden (N, K), plus square anchors."""
    warm, rep, iters = GEMM_TIMING
    shapes = [(m, n, k) for n, k in GEMM_HIDDEN for m in GEMM_M]
    shapes += [(s, s, s) for s in GEMM_SQUARE] + GEMM_RANDOM
    for m, n, k in shapes:
        size = f"m{m}n{n}k{k}"
        for d in DIRECTIONS:
            rows.append(_point_result(
                {"probe": "gemm", "dtype": name, "size": size, "direction": d},
                lambda m=m, n=n, k=k, d=d: _gemm_point(m, n, k, dt, d, warm, rep, _iters_for(d, iters))))


def probe_bmm(rows: list, dt: torch.dtype, name: str) -> None:
    """Batched matmul across the batch-vs-matrix-size split."""
    warm, rep, iters = BMM_TIMING
    for batch, m, n, k in BMM_SHAPES:
        size = f"b{batch}m{m}n{n}k{k}"
        for d in DIRECTIONS:
            rows.append(_point_result(
                {"probe": "bmm", "dtype": name, "size": size, "direction": d},
                lambda batch=batch, m=m, n=n, k=k, d=d:
                    _bmm_point(batch, m, n, k, dt, d, warm, rep, _iters_for(d, iters))))


def probe_conv(rows: list, dt: torch.dtype, name: str) -> None:
    """2D convolution, every shape x fprop/dgrad/wgrad."""
    warm, rep, iters = CONV_TIMING
    for bs, cin, cout, hw, k, stride, pad, groups in CONV_SHAPES:
        for d in CONV_DIRECTIONS:
            size = f"{cin}x{cout}c{hw}k{k}s{stride}g{groups}"
            rows.append(_point_result(
                {"probe": "conv", "dtype": name, "size": size, "direction": d},
                lambda bs=bs, cin=cin, cout=cout, hw=hw, k=k, stride=stride, pad=pad,
                       groups=groups, d=d:
                    _conv_point(bs, cin, cout, hw, k, stride, pad, groups, d, dt, warm, rep, iters)))


def probe_attn(rows: list, dt: torch.dtype, name: str) -> None:
    """Scaled-dot-product attention; causal only where seq_q == seq_kv."""
    warm, rep, iters = ATTN_TIMING
    for bs, qh, kvh, seq_q, seq_kv, head_dim in ATTN_SHAPES:
        for causal in ((False, True) if seq_q == seq_kv else (False,)):
            size = f"q{seq_q}k{seq_kv}h{qh}/{kvh}d{head_dim}"
            for d in DIRECTIONS:
                rows.append(_point_result(
                    {"probe": "attn", "dtype": name, "size": size, "causal": causal,
                     "direction": d},
                    lambda bs=bs, qh=qh, kvh=kvh, seq_q=seq_q, seq_kv=seq_kv, head_dim=head_dim,
                           causal=causal, d=d:
                        _attn_point(bs, qh, kvh, seq_q, seq_kv, head_dim, causal, dt, d, warm,
                                   rep, _iters_for(d, iters))))


def probe_elementwise(rows: list, dt: torch.dtype, name: str) -> None:
    """Activation across an element-count ladder."""
    for numel in ELEMENTWISE_NUMEL:
        warm, rep, iters = next((w, r, i) for cap, w, r, i in ELEMENTWISE_TIERS
                                if numel <= cap)
        for d in DIRECTIONS:
            rows.append(_point_result(
                {"probe": "elementwise", "dtype": name, "size": numel, "direction": d},
                lambda numel=numel, d=d:
                    _elementwise_point(numel, dt, d, warm, rep, _iters_for(d, iters))))


def probe_pool(rows: list, dt: torch.dtype, name: str) -> None:
    """2D pooling, every shape x max/avg/adaptive_avg."""
    warm, rep, iters = POOL_TIMING
    for bs, c, hw, k, stride in POOL_SHAPES:
        for kind in POOL_KINDS:
            size = f"{c}c{hw}k{k}s{stride}"
            for d in DIRECTIONS:
                rows.append(_point_result(
                    {"probe": "pool", "dtype": name, "size": size, "kind": kind, "direction": d},
                    lambda bs=bs, c=c, hw=hw, k=k, stride=stride, kind=kind, d=d:
                        _pool_point(bs, c, hw, k, stride, kind, dt, d, warm, rep,
                                   _iters_for(d, iters))))


def probe_rnn(rows: list, dt: torch.dtype, name: str) -> None:
    """Recurrent stack, every shape x lstm/gru/rnn."""
    warm, rep, iters = RNN_TIMING
    for bs, seq, inp, hidden, layers, bidir in RNN_SHAPES:
        for kind in RNN_KINDS:
            size = f"b{bs}s{seq}i{inp}h{hidden}l{layers}d{2 if bidir else 1}"
            for d in DIRECTIONS:
                rows.append(_point_result(
                    {"probe": "rnn", "dtype": name, "size": size, "kind": kind, "direction": d},
                    lambda bs=bs, seq=seq, inp=inp, hidden=hidden, layers=layers, bidir=bidir,
                           kind=kind, d=d:
                        _rnn_point(bs, seq, inp, hidden, layers, bidir, kind, dt, d, warm, rep,
                                  _iters_for(d, iters))))


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


def probe_activation(rows: list, dt: torch.dtype, name: str) -> None:
    """Every activation the workloads dispatch, across the pointwise ladder. fwd_bwd only --
    an activation's backward is a kernel of its own and is where the kinds differ most."""
    for numel in POINTWISE_NUMEL:
        warm, rep, iters = next((w, r, i) for cap, w, r, i in ELEMENTWISE_TIERS if numel <= cap)
        for kind in ACTIVATION_KINDS:
            rows.append(_point_result(
                {"probe": "activation", "dtype": name, "size": f"n{numel}", "kind": kind,
                 "direction": "fwd_bwd"},
                lambda numel=numel, kind=kind, warm=warm, rep=rep, iters=iters:
                    _activation_point(numel, kind, dt, warm, rep,
                                      _iters_for("fwd_bwd", iters))))


def probe_norm(rows: list, dt: torch.dtype, name: str) -> None:
    """Each normalisation layer over the shapes its own architecture family produces."""
    warm, rep, iters = NORM_TIMING
    it = _iters_for("fwd_bwd", iters)
    for kind in NORM_KINDS:
        if kind == "batchnorm2d":
            grid = [(s, f"b{s[0]}c{s[1]}h{s[2]}w{s[3]}") for s in BATCHNORM_SHAPES]
        elif kind == "groupnorm":
            grid = [(s, f"b{s[0]}c{s[1]}h{s[2]}w{s[3]}g{s[4]}") for s in GROUPNORM_SHAPES]
        else:                                   # layernorm and rmsnorm share one grid
            grid = [(s, f"r{s[0]}c{s[1]}") for s in LAYERNORM_SHAPES]
        for shape, size in grid:
            rows.append(_point_result(
                {"probe": "norm", "dtype": name, "size": size, "kind": kind,
                 "direction": "fwd_bwd"},
                lambda shape=shape, kind=kind: _norm_point(shape, kind, dt, warm, rep, it)))


def probe_dropout(rows: list, dt: torch.dtype, name: str) -> None:
    """Per-element dropout against per-sample stochastic depth, on the same ladder and shape."""
    for numel in POINTWISE_NUMEL:
        warm, rep, iters = next((w, r, i) for cap, w, r, i in ELEMENTWISE_TIERS if numel <= cap)
        for kind in DROPOUT_KINDS:
            rows.append(_point_result(
                {"probe": "dropout", "dtype": name,
                 "size": f"n{numel}p{int(DROPOUT_P * 100)}", "kind": kind,
                 "direction": "fwd_bwd"},
                lambda numel=numel, kind=kind, warm=warm, rep=rep, iters=iters:
                    _dropout_point(numel, kind, dt, warm, rep,
                                   _iters_for("fwd_bwd", iters))))


def probe_reduction(rows: list, dt: torch.dtype, name: str) -> None:
    """Standalone softmax and mean paths -- the ones outside SDPA, which attn does not reach."""
    warm, rep, iters = REDUCTION_TIMING
    it = _iters_for("fwd_bwd", iters)
    for n_rows, width in REDUCTION_SHAPES:
        for kind in REDUCTION_KINDS:
            rows.append(_point_result(
                {"probe": "reduction", "dtype": name, "size": f"r{n_rows}c{width}",
                 "kind": kind, "direction": "fwd_bwd"},
                lambda n_rows=n_rows, width=width, kind=kind:
                    _reduction_point(n_rows, width, kind, dt, warm, rep, it)))


def probe_loss(rows: list, dt: torch.dtype, name: str) -> None:
    """Cross-entropy at the batch sizes the classification sweeps run."""
    warm, rep, iters = LOSS_TIMING
    it = _iters_for("fwd_bwd", iters)
    for batch, classes in LOSS_SHAPES:
        rows.append(_point_result(
            {"probe": "loss", "dtype": name, "size": f"b{batch}c{classes}",
             "kind": "cross_entropy", "direction": "fwd_bwd"},
            lambda batch=batch, classes=classes:
                _loss_point(batch, classes, dt, warm, rep, it)))


def probe_optimizer(rows: list, dt: torch.dtype, name: str) -> None:
    """SGD's two per-step costs over a persistent parameter set. No direction -- neither is a
    forward or a backward, so both are recorded as fwd.

    fp32 regardless of what DTYPES holds: the parameters an optimizer owns stay fp32 whatever the
    autocast dtype of the step that produced their gradients, so a row in any other dtype would
    name a configuration nothing runs. The guard is redundant while DTYPES is fp32-only and is
    kept because it is a property of the optimizer, not of the current sweep. Adam is
    deliberately absent: the workloads use SGD with momentum."""
    if dt is not torch.float32:
        return
    warm, rep, iters = OPTIMIZER_TIMING
    for tensors in OPTIMIZER_TENSORS:
        for total in OPTIMIZER_NUMEL:
            for kind in OPTIMIZER_KINDS:
                rows.append(_point_result(
                    {"probe": "optimizer", "dtype": name, "size": f"t{tensors}e{total}",
                     "kind": kind, "direction": "fwd"},
                    lambda tensors=tensors, total=total, kind=kind:
                        _optimizer_point(tensors, total, kind, warm, rep, iters)))


# ---- the sweep --------------------------------------------------------------------------------------------------

# Every family, in the order they are swept. This is the only place the sweep's membership is
# written down -- run_probes walks it, so adding a family here is the whole wiring change.
#
# Ordered heaviest-first within each group, so a card that dies partway through has still
# produced the rows that say the most about it.
PROBES = (
    # core
    probe_gemm, probe_bmm, probe_conv, probe_attn, probe_elementwise, probe_pool, probe_rnn,
    # large shape
    probe_gemm_ext, probe_conv_ext, probe_conv1d,
    # per layer
    probe_activation, probe_norm, probe_dropout, probe_reduction,
    # per step
    probe_loss, probe_optimizer,
)

# -------------------------------------------------------------------------------------------------------------------

def device_meta() -> dict:
    p = torch.cuda.get_device_properties(0)
    free, total = torch.cuda.mem_get_info()
    cores = CORES_PER_SM.get((p.major, p.minor))
    meta = {
        "gpu": p.name.replace("NVIDIA ", ""),
        "compute_capability": f"{p.major}.{p.minor}",
        "sm_count": p.multi_processor_count,
        "cores_per_sm": cores,
        "cuda_cores": p.multi_processor_count * cores if cores else None,
        "total_memory_gb": round(total / 1024 ** 3, 2),
        "free_memory_gb": round(free / 1024 ** 3, 2),
        "l2_cache_mb": round(getattr(p, "L2_cache_size", 0) / 1024 ** 2, 2),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver_host": platform.node(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for key, fn in (
            ("uuid", lambda: pynvml.nvmlDeviceGetUUID(h)),
            ("driver_version", lambda: pynvml.nvmlSystemGetDriverVersion()),
            ("power_limit_w", lambda: pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0),
            ("max_sm_clock_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
            ("max_graphics_clock_mhz",
             lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)),
            ("max_memory_clock_mhz",
             lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)),
            # A board can enforce below its configured limit -- insufficient power connectors,
            # for instance -- which would explain a card hitting its ceiling at a low clock
            # while power_limit_w still reads stock.
            ("enforced_power_limit_w", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0),
            ("power_limit_range_w",
             lambda: [x / 1000.0 for x in pynvml.nvmlDeviceGetPowerManagementLimitConstraints(h)]),
            # Reference point for max_temp_c: without it, a die temperature means nothing.
            ("thermal_slowdown_c", lambda: pynvml.nvmlDeviceGetTemperatureThreshold(
                h, pynvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)),
            ("max_pcie_link_width", lambda: pynvml.nvmlDeviceGetMaxPcieLinkWidth(h)),
        ):
            try:
                v = fn()
                meta[key] = v.decode() if isinstance(v, bytes) else v
            except Exception:
                pass
        pynvml.nvmlShutdown()
    except Exception:
        pass

    return meta


def save(sig: dict, path: str | Path | None = None) -> Path:
    """Write device meta and every probe row as one JSON."""
    d = sig["device"]
    tag = str(d.get("uuid", "unknownuuid")).removeprefix("GPU-").split("-")[0]
    out = Path(path or f"{d['gpu'].replace(' ', '_')}/device_probe_{tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sig, indent=2))
    return out


# NVML throttle-reason bits worth recording. GpuIdle (0x1) is excluded on purpose: it fires
# whenever the card is between measurements and says nothing about capability.
THROTTLE_BITS = {
    "applications_clocks": 0x2,
    "sw_power_cap": 0x4,
    "hw_slowdown": 0x8,
    "sync_boost": 0x10,
    "sw_thermal": 0x20,
    "hw_thermal": 0x40,
    "hw_power_brake": 0x80,
    "display_clock": 0x100,
}
CLOCK_SAMPLE_MS = 100

# Cumulative nanoseconds the card was held down by each cause. Strictly better than the
# throttle bitmask, which collapsed to ['sw_power_cap'] on all 11 nodes of one fleet and
# separated nothing: these are time-weighted and attribute the loss to a specific policy.
# Read at probe start and end; only the delta over the run is meaningful.
VIOLATION_POLICIES = ("POWER", "THERMAL", "SYNC_BOOST", "BOARD_LIMIT", "RELIABILITY")


def _violations(pynvml, h) -> dict:
    """Cumulative throttle nanoseconds per policy, or {} where unsupported."""
    out = {}
    for name in VIOLATION_POLICIES:
        pol = getattr(pynvml, f"NVML_PERF_POLICY_{name}", None)
        if pol is None:
            continue
        try:
            out[name] = pynvml.nvmlDeviceGetViolationStatus(h, pol).violationTime
        except Exception:
            pass
    return out


class _ClockSampler:
    """Polls the clock actually sustained under load, plus why it is being held down.

    device_meta records max_sm_clock_mhz, the card's advertised ceiling, which reads identical
    on a healthy and a throttled card. One RTX 5090 node measured ~38% slower than three
    siblings across every probe family while reporting the same 3090 MHz ceiling, a stock 575 W
    power limit, and the highest startup TFLOPS of the group -- nothing in the metadata
    separated it. What does is the clock it actually holds while working, and NVML's reason for
    the difference. Sampling runs through the whole probe, which is minutes of sustained load."""

    def __init__(self, interval_ms: int = CLOCK_SAMPLE_MS):
        self.interval_s = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread = None
        self._sm, self._mem, self._temp, self._reasons = [], [], [], 0
        self._power, self._free = [], []
        self._pstate, self._fan, self._pcie_w, self._pcie_g, self._others = [], [], [], [], []
        self._viol_start = {}
        self._busy_samples = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            return
        reasons_fn = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons",
                             getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
        self._viol_start = _violations(pynvml, h)
        me = os.getpid()
        while True:
            try:
                sm = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                self._sm.append(sm)
                self._mem.append(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
                self._temp.append(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
                # Separates "power-limited" from "clock capped for some other reason", and a
                # dip in device-wide free VRAM is a co-tenant arriving mid-probe. Both are
                # reliable here but not in cuda_monitor: these counters need a window of
                # seconds, and the probe gives minutes where a workload config gives ~0.85s.
                self._power.append(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
                mi = pynvml.nvmlDeviceGetMemoryInfo(h)
                self._free.append((mi.total - mi.used) / 1e9)
                self._pstate.append(pynvml.nvmlDeviceGetPerformanceState(h))
                self._fan.append(pynvml.nvmlDeviceGetFanSpeed(h))
                self._pcie_w.append(pynvml.nvmlDeviceGetCurrPcieLinkWidth(h))
                self._pcie_g.append(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h))
                # Definitive co-tenancy answer; free VRAM is not, since our own probe
                # allocations dominate the dip.
                self._others.append(sum(1 for x in pynvml.nvmlDeviceGetComputeRunningProcesses(h)
                                        if x.pid != me))
                if reasons_fn is not None:
                    bits = reasons_fn(h)
                    # 0x1 is GpuIdle; only count reasons seen while the card is actually working
                    if not bits & 0x1:
                        self._busy_samples += 1
                        self._reasons |= bits
            except Exception:
                pass
            if self._stop.wait(self.interval_s):
                break

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        viol = {}
        try:
            import pynvml
            end = _violations(pynvml, pynvml.nvmlDeviceGetHandleByIndex(0))
            viol = {f"violation_{k.lower()}_ms": round((end[k] - self._viol_start.get(k, 0)) / 1e6, 1)
                    for k in end if k in self._viol_start}
        except Exception:
            pass
        if not self._sm:
            return viol
        busy = [c for c in self._sm if c > 0]
        return {
            "achieved_sm_clock_mhz_median": statistics.median(busy),
            "achieved_sm_clock_mhz_max": max(busy),
            "achieved_mem_clock_mhz_median": statistics.median(self._mem),
            "max_temp_c": max(self._temp) if self._temp else None,
            "clock_samples": len(self._sm),
            "busy_samples": self._busy_samples,
            "throttle_reasons": sorted(n for n, b in THROTTLE_BITS.items() if self._reasons & b),
            "power_w_median": statistics.median(self._power) if self._power else None,
            "power_w_max": max(self._power) if self._power else None,
            "free_memory_gb_min": round(min(self._free), 2) if self._free else None,
            "perf_state_max": max(self._pstate) if self._pstate else None,
            "fan_speed_pct_max": max(self._fan) if self._fan else None,
            "pcie_link_width_max": max(self._pcie_w) if self._pcie_w else None,
            "pcie_link_gen_max": max(self._pcie_g) if self._pcie_g else None,
            "other_processes_max": max(self._others) if self._others else None,
            **viol,
        }


def run_probes(dtypes: dict | None = None, families: tuple | None = None) -> dict:
    """Sweep every family and return {"device": ..., "probes": [...]}.

    One clock sampler spans the whole sweep: throttling is a property of the card under sustained
    load, not of whichever family happened to be running, and one continuous window is what makes
    the violation counters comparable between cards.

    Each family is contained. A row's own failure never escapes _point_result, but a context
    corrupted badly enough can raise from outside it, and a family that dies that way must not
    take the families after it down as well -- a partial signature is worth collecting, and the
    families that still ran are the only evidence of what the card could do."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")

    sampler = _ClockSampler()
    sampler.start()
    rows: list = []
    try:
        for name, dt in (dtypes or DTYPES).items():
            for family in (families or PROBES):
                try:
                    family(rows, dt, name)
                except Exception as e:
                    print(f"  {family.__name__} [{name}] ended early -- "
                          f"{type(e).__name__}: {e}", flush=True)
    finally:
        clocks = sampler.stop()

    return {"device": {**device_meta(), **clocks}, "probes": rows}


def main() -> None:
    if not torch.cuda.is_available():
        sys.exit("no CUDA device")

    meta = device_meta()
    print(f"{meta['gpu']}  cc{meta['compute_capability']}  "
          f"{meta['sm_count']} SMs  {meta['total_memory_gb']} GB  "
          f"L2 {meta['l2_cache_mb']} MB")

    sig = run_probes()

    print(f"\n  {'probe':11s}{'dt':>6s}{'size':>24s}{'variant':>13s}{'dir':>9s}"
          f"{'us':>13s}{'ms':>11s}")
    for r in sig["probes"]:
        # kind/causal and direction are separate columns: they vary independently, and
        # folding them would print the three pool kinds as identical lines
        variant = str(r.get("kind", r.get("causal", "")))
        head = (f"  {r['probe']:11s}{r['dtype']:>6s}{r['size']:>24}"
                f"{variant:>13s}{r.get('direction', ''):>9s}")
        if r.get("oom"):
            print(f"{head}{'OOM':>13s}{'':>11s}")
            continue
        if r.get("error"):
            print(f"{head}{'ERR':>13s}{'':>11s}   {r['error'][:60]}")
            continue
        # both units: one timed configuration spans 1.9 us to 115 ms across the sweep
        print(f"{head}{r['ms'] * 1000:>13.2f}{r['ms']:>11.4f}")

    print(f"\nsignature written to {save(sig)}")


if __name__ == "__main__":
    main()
