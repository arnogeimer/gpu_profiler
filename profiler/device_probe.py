#!/usr/bin/env python3

'''
CUDA kernel timing probes for different operations.
'''

import json
import platform
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import torch
import torch.nn.grad          # submodule; `import torch` alone does not pull it in
from profiler.profiler import time_fn

WARMUP, REPEATS = 5, 10

# CUDA cores per SM by compute capability; neither torch nor NVML reports it.
CORES_PER_SM = {(3, 0): 192, (3, 5): 192, (3, 7): 192,
                (5, 0): 128, (5, 2): 128, (5, 3): 128,
                (6, 0): 64, (6, 1): 128, (6, 2): 128,
                (7, 0): 64, (7, 2): 64, (7, 5): 64,
                (8, 0): 64, (8, 6): 128, (8, 7): 128, (8, 9): 128,
                (9, 0): 128, (10, 0): 128, (12, 0): 128}

DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}

# (N, K) hidden dims x an M ladder. The workloads never dispatch a square matmul: N and K
# are model hidden dims while M is the token count (batch x seq, or batch x H x W), and
# every sweep axis they vary -- batch_size, sequence_length, img_size -- moves only M.
GEMM_HIDDEN = [(768, 768), (3072, 768), (768, 3072), (1536, 576), (49152, 576)]
GEMM_M = [80, 256, 512, 2048, 3152, 8192]
GEMM_SQUARE = [1024, 2048, 4096]
GEMM_TIMING = (3, 10, 20)

# (batch, M, N, K). Not recoverable from the gemm arm: at equal total FLOPs the time
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
BMM_TIMING = (3, 10, 20)

ELEMENTWISE_NUMEL = [2 ** 19, 2 ** 21, 2 ** 23, 2 ** 25, 2 ** 26, 2 ** 27, 2 ** 28]
ELEMENTWISE_TIERS = [(2 ** 23, 3, 10, 200), (2 ** 26, 3, 10, 20), (2 ** 63, 3, 10, 3)]

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
CONV_TIMING = (3, 10, 20)

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
ATTN_TIMING = (3, 10, 10)

# (bs, c, hw, k, stride) x max/avg/adaptive_avg. 64c56 and 256c28 are gone: their times
# did not move with dtype, so they were reporting a launch floor rather than the op.
POOL_SHAPES = [(16, 64, 112, 3, 2), (16, 2048, 7, 3, 1)]
POOL_KINDS = ("max", "avg", "adaptive_avg")
POOL_TIMING = (3, 10, 50)

# (bs, seq, input, hidden, layers, bidirectional) x lstm/gru/rnn. One axis moves per row
# off a fixed baseline. No workload uses a recurrent layer, so unlike the other arms these
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
RNN_TIMING = (3, 10, 10)

DIRECTIONS = ("fwd", "fwd_bwd")
# a captured fwd_bwd iteration allocates fresh activations and grads that the plain forward
# does not, so it gets a shorter loop to keep peak memory near the forward's
BWD_ITER_DIV = 4

SHAPE_SEED = 420
N_RANDOM = 50


def _rng(arm: str) -> random.Random:
    """One stream per arm, so adding or reordering arms cannot shift another's draws.
    Seeding from a str goes through sha512, which is stable across runs and platforms
    (unlike hash(), which PYTHONHASHSEED randomises)."""
    return random.Random(f"{SHAPE_SEED}:{arm}")


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


GEMM_RANDOM = _random_gemm_shapes()
BMM_SHAPES += _random_bmm_shapes()
CONV_SHAPES += _random_conv_shapes()
ATTN_SHAPES += _random_attn_shapes()
RNN_SHAPES += _random_rnn_shapes()

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


# -------------------------------------------------------------------------------------------------------------------

def probe_gemm(rows: list, dt: torch.dtype, name: str) -> None:
    """Matmul: an M ladder at each hidden (N, K), plus square anchors."""
    warm, rep, iters = GEMM_TIMING
    shapes = [(m, n, k) for n, k in GEMM_HIDDEN for m in GEMM_M]
    shapes += [(s, s, s) for s in GEMM_SQUARE] + GEMM_RANDOM
    for m, n, k in shapes:
        size = f"m{m}n{n}k{k}"
        for d in DIRECTIONS:
            try:
                ms = _gemm_point(m, n, k, dt, d, warm, rep, _iters_for(d, iters))
                rows.append({
                    "probe": "gemm",
                    "dtype": name,
                    "size": size,
                    "direction": d,
                    "ms": ms,
                })
            except torch.cuda.OutOfMemoryError:
                rows.append({"probe": "gemm", "dtype": name, "size": size,
                             "direction": d, "oom": True})
            torch.cuda.empty_cache()


def probe_bmm(rows: list, dt: torch.dtype, name: str) -> None:
    """Batched matmul across the batch-vs-matrix-size split."""
    warm, rep, iters = BMM_TIMING
    for batch, m, n, k in BMM_SHAPES:
        size = f"b{batch}m{m}n{n}k{k}"
        for d in DIRECTIONS:
            try:
                ms = _bmm_point(batch, m, n, k, dt, d, warm, rep, _iters_for(d, iters))
                rows.append({
                    "probe": "bmm", "dtype": name, "size": size,
                    "direction": d, "ms": ms,
                })
            except torch.cuda.OutOfMemoryError:
                rows.append({"probe": "bmm", "dtype": name, "size": size,
                             "direction": d, "oom": True})
            torch.cuda.empty_cache()


def probe_conv(rows: list, dt: torch.dtype, name: str) -> None:
    """2D convolution, every shape x fprop/dgrad/wgrad."""
    warm, rep, iters = CONV_TIMING
    for bs, cin, cout, hw, k, stride, pad, groups in CONV_SHAPES:
        for d in CONV_DIRECTIONS:
            size = f"{cin}x{cout}c{hw}k{k}s{stride}g{groups}"
            try:
                ms = _conv_point(bs, cin, cout, hw, k, stride, pad, groups, d,
                                 dt, warm, rep, iters)
                rows.append({
                    "probe": "conv", "dtype": name, "size": size,
                    "direction": d, "ms": ms,
                })
            except torch.cuda.OutOfMemoryError:
                rows.append({"probe": "conv", "dtype": name, "size": size,
                             "direction": d, "oom": True})
            torch.cuda.empty_cache()


def probe_attn(rows: list, dt: torch.dtype, name: str) -> None:
    """Scaled-dot-product attention; causal only where seq_q == seq_kv."""
    warm, rep, iters = ATTN_TIMING
    for bs, qh, kvh, seq_q, seq_kv, head_dim in ATTN_SHAPES:
        for causal in ((False, True) if seq_q == seq_kv else (False,)):
            size = f"q{seq_q}k{seq_kv}h{qh}/{kvh}d{head_dim}"
            for d in DIRECTIONS:
                try:
                    ms = _attn_point(bs, qh, kvh, seq_q, seq_kv, head_dim, causal,
                                     dt, d, warm, rep, _iters_for(d, iters))
                    rows.append({
                        "probe": "attn", "dtype": name, "size": size,
                        "causal": causal, "direction": d, "ms": ms,
                    })
                except torch.cuda.OutOfMemoryError:
                    rows.append({"probe": "attn", "dtype": name, "size": size,
                                 "causal": causal, "direction": d, "oom": True})
                torch.cuda.empty_cache()


def probe_elementwise(rows: list, dt: torch.dtype, name: str) -> None:
    """Activation across an element-count ladder."""
    for numel in ELEMENTWISE_NUMEL:
        warm, rep, iters = next((w, r, i) for cap, w, r, i in ELEMENTWISE_TIERS
                                if numel <= cap)
        for d in DIRECTIONS:
            try:
                ms = _elementwise_point(numel, dt, d, warm, rep, _iters_for(d, iters))
                rows.append({
                    "probe": "elementwise", "dtype": name, "size": numel,
                    "direction": d, "ms": ms,
                })
            except torch.cuda.OutOfMemoryError:
                rows.append({"probe": "elementwise", "dtype": name, "size": numel,
                             "direction": d, "oom": True})
            torch.cuda.empty_cache()


def probe_pool(rows: list, dt: torch.dtype, name: str) -> None:
    """2D pooling, every shape x max/avg/adaptive_avg."""
    warm, rep, iters = POOL_TIMING
    for bs, c, hw, k, stride in POOL_SHAPES:
        for kind in POOL_KINDS:
            size = f"{c}c{hw}k{k}s{stride}"
            for d in DIRECTIONS:
                try:
                    ms = _pool_point(bs, c, hw, k, stride, kind, dt, d,
                                     warm, rep, _iters_for(d, iters))
                    rows.append({
                        "probe": "pool", "dtype": name, "size": size, "kind": kind,
                        "direction": d, "ms": ms
                    })
                except torch.cuda.OutOfMemoryError:
                    rows.append({"probe": "pool", "dtype": name, "size": size,
                                 "kind": kind, "direction": d, "oom": True})
                torch.cuda.empty_cache()


def probe_rnn(rows: list, dt: torch.dtype, name: str) -> None:
    """Recurrent stack, every shape x lstm/gru/rnn."""
    warm, rep, iters = RNN_TIMING
    for bs, seq, inp, hidden, layers, bidir in RNN_SHAPES:
        for kind in RNN_KINDS:
            size = f"b{bs}s{seq}i{inp}h{hidden}l{layers}d{2 if bidir else 1}"
            for d in DIRECTIONS:
                try:
                    ms = _rnn_point(bs, seq, inp, hidden, layers, bidir, kind,
                                    dt, d, warm, rep, _iters_for(d, iters))
                    rows.append({
                        "probe": "rnn", "dtype": name, "size": size, "kind": kind,
                        "direction": d, "ms": ms
                    })
                except torch.cuda.OutOfMemoryError:
                    rows.append({"probe": "rnn", "dtype": name, "size": size,
                                 "kind": kind, "direction": d, "oom": True})
                torch.cuda.empty_cache()

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


class _ClockSampler:
    """Polls the clock actually sustained under load, plus why it is being held down.

    device_meta records max_sm_clock_mhz, the card's advertised ceiling, which reads identical
    on a healthy and a throttled card. One RTX 5090 node measured ~38% slower than three
    siblings across every probe arm while reporting the same 3090 MHz ceiling, a stock 575 W
    power limit, and the highest startup TFLOPS of the group -- nothing in the metadata
    separated it. What does is the clock it actually holds while working, and NVML's reason for
    the difference. Sampling runs through the whole probe, which is minutes of sustained load."""

    def __init__(self, interval_ms: int = CLOCK_SAMPLE_MS):
        self.interval_s = interval_ms / 1000.0
        self._stop = threading.Event()
        self._thread = None
        self._sm, self._mem, self._temp, self._reasons = [], [], [], 0
        self._power, self._free = [], []
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
        if not self._sm:
            return {}
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
        }


def run_probe(dtypes: dict | None = None) -> dict:
    """Measure the full signature and return it as a dict."""
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")

    sampler = _ClockSampler()
    sampler.start()
    rows: list = []
    try:
        for name, dt in (dtypes or DTYPES).items():
            probe_gemm(rows, dt, name)
            probe_bmm(rows, dt, name)
            probe_conv(rows, dt, name)
            probe_attn(rows, dt, name)
            probe_elementwise(rows, dt, name)
            probe_pool(rows, dt, name)
            probe_rnn(rows, dt, name)
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

    sig = run_probe()

    print(f"\n  {'probe':11s}{'dt':>6s}{'size':>24s}{'variant':>13s}{'dir':>9s}"
          f"{'us':>13s}{'ms':>11s}")
    for r in sig["probes"]:
        # kind/causal and direction are separate columns: they vary independently, and
        # folding them would print the three pool kinds as identical lines
        variant = str(r.get("kind", r.get("causal", "")))
        head = (f"  {r['probe']:11s}{r['dtype']:>6s}{r['size']:>24}{variant:>13s}"
                f"{r.get('direction', ''):>9s}")
        if r.get("oom"):
            print(f"{head}{'OOM':>13s}{'':>11s}")
            continue
        # both units: one timed configuration spans 1.9 us to 115 ms across the sweep
        print(f"{head}{r['ms'] * 1000:>13.2f}{r['ms']:>11.4f}")

    print(f"\nsignature written to {save(sig)}")


if __name__ == "__main__":
    main()
