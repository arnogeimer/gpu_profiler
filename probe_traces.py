"""Standalone device probe: runs exactly the configurations in probe_traces.csv.

Returns one DataFrame, joinable to that file on (probe, dtype, size, direction, kind, causal):
    gpu, probe, dtype, size, direction, kind, causal, ms, oom
ms is the fastest of `repeats` CUDA-graph replays, per iteration. A configuration that runs out
of memory keeps its row, with ms=NaN and oom=True.

    from probe_traces import run
    df = run()

The configurations come from probe_traces.csv itself: that file lists every experiment as
(probe, dtype, size, direction, kind, causal), and this runs those and no others. The shape
grids below exist only to turn a `size` string back into the tensor dimensions to allocate.
If the CSV names a configuration this grid cannot build, run() raises rather than silently
measuring less.

Shapes, seeds and timing loops are copied from profiler/device_probe.py and
profiler/probe_extension.py; keep them in step if those change.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import math
import random
from pathlib import Path

import pandas as pd
import torch

MEMORY_FRACTION = 0.995
MIN_GRAPH_MS = 2.0
MAX_ITERS = 200
BWD_ITER_DIV = 4
SHAPE_SEED = 420
N_RANDOM = 50
N_RANDOM_EXT = 16
DTYPES = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
DIRECTIONS = ("fwd", "fwd_bwd")
CONV_DIRECTIONS = ("fprop", "dgrad", "wgrad")
COLS = ["gpu", "probe", "dtype", "size", "direction", "kind", "causal", "ms", "oom"]
# Beside this file, not the caller's working directory -- run() is meant to be imported.
REFERENCE_CSV = Path(__file__).resolve().parent / "probe_traces.csv"

GEMM_HIDDEN = [(768, 768), (3072, 768), (768, 3072), (1536, 576), (49152, 576)]
GEMM_M = [80, 256, 512, 2048, 3152, 8192]
GEMM_SQUARE = [1024, 2048, 4096]
GEMM_TIMING = (3, 5, 20)

BMM_SHAPES = [(256, 128, 128, 128), (32, 256, 256, 256), (4, 512, 512, 512),
              (1024, 64, 64, 64), (2, 1024, 1024, 1024), (192, 197, 197, 64),
              (192, 197, 64, 197), (36, 512, 512, 64), (36, 512, 64, 512)]
BMM_TIMING = (3, 5, 20)

ELEMENTWISE_NUMEL = [2 ** 19, 2 ** 21, 2 ** 23, 2 ** 25, 2 ** 26, 2 ** 27, 2 ** 28]
ELEMENTWISE_TIERS = [(2 ** 23, 3, 5, 200), (2 ** 26, 3, 5, 20), (2 ** 63, 3, 5, 3)]

CONV_SHAPES = [(16, 3, 64, 224, 7, 2, 3, 1), (16, 3, 768, 224, 16, 16, 0, 1),
               (16, 64, 64, 56, 3, 1, 1, 1), (16, 64, 256, 56, 1, 1, 0, 1),
               (16, 128, 256, 28, 3, 2, 1, 1), (16, 384, 384, 28, 3, 1, 1, 384),
               (16, 256, 256, 14, 3, 1, 1, 1), (16, 512, 512, 7, 3, 1, 1, 1),
               (16, 512, 2048, 7, 1, 1, 0, 1)]
CONV_TIMING = (3, 5, 20)

ATTN_SHAPES = [(16, 12, 12, 197, 197, 64), (16, 12, 12, 577, 577, 64),
               (4, 9, 3, 256, 256, 64), (4, 9, 3, 512, 512, 64),
               (8, 12, 12, 1024, 1024, 64), (8, 12, 12, 2048, 2048, 64),
               (2, 32, 8, 2048, 2048, 128), (2, 8, 8, 4096, 4096, 40),
               (2, 8, 8, 4096, 77, 40)]
ATTN_TIMING = (3, 5, 10)

POOL_SHAPES = [(16, 64, 112, 3, 2), (16, 2048, 7, 3, 1)]
POOL_KINDS = ("max", "avg", "adaptive_avg")
POOL_TIMING = (3, 5, 50)

RNN_SHAPES = [(32, 128, 512, 512, 1, False), (32, 128, 512, 512, 3, False),
              (32, 512, 512, 512, 1, False), (32, 128, 512, 1024, 1, False),
              (32, 128, 512, 512, 1, True), (8, 128, 512, 512, 1, False),
              (128, 128, 512, 512, 1, False)]
RNN_KINDS = ("lstm", "gru", "rnn")
RNN_TIMING = (3, 5, 10)

EXT_TIMING = (2, 3, 2)
GEMM_EXT_HIDDEN = [(768, 768), (3072, 768), (768, 3072), (512, 128), (1536, 576)]
GEMM_EXT_M = [16384, 32768, 65536, 100352, 131072]
CONV1D_SHAPES = [(4, 1, 512, 480000, 10, 5), (4, 512, 512, 95999, 3, 2),
                 (4, 512, 512, 47999, 3, 2), (4, 512, 512, 23999, 3, 2),
                 (4, 512, 512, 11999, 3, 2), (4, 512, 512, 5999, 2, 2),
                 (4, 512, 512, 2999, 2, 2), (8, 1, 512, 160000, 10, 5),
                 (8, 512, 512, 31999, 3, 2)]
CONV2D_EXT_SHAPES = [(32, 64, 64, 224, 3, 1, 1, 1), (64, 128, 128, 112, 3, 1, 1, 1),
                     (128, 256, 256, 56, 3, 1, 1, 1), (32, 128, 256, 224, 1, 1, 0, 1),
                     (64, 384, 384, 56, 3, 1, 1, 384), (32, 3, 96, 448, 7, 2, 3, 1)]
MEM_BUDGET_BYTES, FP32, GRAD_FACTOR = 8.0e9, 4, 2.9

# --- random shapes -------------------------------------------------------------------------

def _rng(arm):
    return random.Random(f"{SHAPE_SEED}:{arm}")


def _logdraw(r, lo, hi, mult):
    return max(mult, int(round(2 ** r.uniform(lo, hi) / mult)) * mult)


def _random_gemm_shapes():
    r, out = _rng("gemm"), []
    while len(out) < N_RANDOM:
        m = _logdraw(r, 6, 13, 16)
        n, k = (_logdraw(r, 7, 12, 64) for _ in range(2))
        if 2 * m * n * k <= 40e9:
            out.append((m, n, k))
    return out


def _random_bmm_shapes():
    r, out = _rng("bmm"), []
    while len(out) < N_RANDOM:
        b = r.choice([2, 4, 8, 16, 32, 64, 128, 256, 512])
        m, n, k = (_logdraw(r, 5, 10, 32) for _ in range(3))
        if 2 * b * m * n * k <= 20e9:
            out.append((b, m, n, k))
    return out


def _random_conv_shapes():
    r, out = _rng("conv"), []
    while len(out) < N_RANDOM:
        k, stride = r.choice([1, 3, 5, 7]), r.choice([1, 1, 2])
        hw = r.choice([7, 14, 28, 56, 112, 224])
        cin = _logdraw(r, 5, 10, 32)
        depthwise = r.random() < 0.25
        cout, groups = (cin, cin) if depthwise else (_logdraw(r, 5, 10, 32), 1)
        bs = r.choice([8, 16, 32])
        if bs * cin * hw * hw <= 6e7:
            out.append((bs, cin, cout, hw, k, stride, k // 2, groups))
    return out


def _random_attn_shapes():
    r, out = _rng("attn"), []
    while len(out) < N_RANDOM:
        d = r.choice([32, 64, 80, 128])
        qh = r.choice([4, 8, 12, 16, 32])
        kvh = r.choice([h for h in (1, 2, 4, 8, qh) if qh % h == 0])
        seq = r.choice([128, 197, 256, 384, 512, 1024, 2048, 4096])
        bs = r.choice([1, 2, 4, 8, 16])
        if bs * qh * seq * seq * d <= 4e9:
            out.append((bs, qh, kvh, seq, seq, d))
    return out


def _random_rnn_shapes():
    r, out = _rng("rnn"), []
    while len(out) < N_RANDOM:
        inp, hidden = (_logdraw(r, 7, 10.58, 128) for _ in range(2))
        seq = r.choice([32, 64, 128, 256, 512])
        bs = r.choice([8, 16, 32, 64, 128])
        layers, bidir = r.choice([1, 1, 2, 3]), r.random() < 0.3
        if bs * seq * hidden * layers <= 4e7:
            out.append((bs, seq, inp, hidden, layers, bidir))
    return out


def _random_gemm_ext_shapes():
    r, out = _rng("gemm_ext"), []
    while len(out) < N_RANDOM_EXT:
        m = _logdraw(r, 13, 17, 16)
        n, k = (_logdraw(r, 7, 12, 64) for _ in range(2))
        if (m * k + k * n + m * n) * FP32 * GRAD_FACTOR <= MEM_BUDGET_BYTES:
            out.append((m, n, k))
    return out


GEMM_RANDOM = _random_gemm_shapes()
BMM_SHAPES += _random_bmm_shapes()
CONV_SHAPES += _random_conv_shapes()
ATTN_SHAPES += _random_attn_shapes()
RNN_SHAPES += _random_rnn_shapes()
GEMM_EXT_RANDOM = _random_gemm_ext_shapes()

# One entry per family: (shape list, timing). Family names and size-string encodings are
# exactly those the fleet emitted, so a local run joins probe_traces.csv on (probe, size).
# gemm_ext/conv_ext are the same operations at larger shapes; they stay separate families
# because that is how they were recorded.
GEMM_GRID = ([(sh, GEMM_TIMING) for sh in
              [(m, n, k) for n, k in GEMM_HIDDEN for m in GEMM_M]
              + [(s, s, s) for s in GEMM_SQUARE] + GEMM_RANDOM])
GEMM_EXT_GRID = [(sh, EXT_TIMING) for sh in
                 [(m, n, k) for n, k in GEMM_EXT_HIDDEN for m in GEMM_EXT_M] + GEMM_EXT_RANDOM]
BMM_GRID = [(sh, BMM_TIMING) for sh in BMM_SHAPES]
CONV_GRID = [(sh, CONV_TIMING) for sh in CONV_SHAPES]
CONV_EXT_GRID = [(sh, EXT_TIMING) for sh in CONV2D_EXT_SHAPES]
CONV1D_GRID = [(sh, EXT_TIMING) for sh in CONV1D_SHAPES]
ATTN_GRID = [(sh, ATTN_TIMING) for sh in ATTN_SHAPES]
POOL_GRID = [(sh, POOL_TIMING) for sh in POOL_SHAPES]
RNN_GRID = [(sh, RNN_TIMING) for sh in RNN_SHAPES]

# --- timing --------------------------------------------------------------------------------

def _clear_capture_state():
    """A capture that raised leaves the RNG generator capture-bound; only a clean capture
    clears it, and every later RNG op in the process fails until one runs."""
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.zeros(1, device="cuda").add_(1.0)
        g.replay()
        torch.cuda.synchronize()
    except Exception:
        pass


def time_fn(fn, warmup, repeats, iters):
    """Fastest per-iteration CUDA graph ms. iters is raised until the graph spans
    MIN_GRAPH_MS, so one graph launch is amortised over enough work."""
    try:
        return _time_fn(fn, warmup, repeats, iters)
    except Exception:
        _clear_capture_state()
        raise


def _time_fn(fn, warmup, repeats, iters):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()
    g.replay()
    torch.cuda.synchronize()

    s0, e0 = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    s0.record(); g.replay(); e0.record()
    torch.cuda.synchronize()
    per_iter = s0.elapsed_time(e0) / iters
    if per_iter > 0 and per_iter * iters < MIN_GRAPH_MS:
        iters = min(MAX_ITERS, math.ceil(MIN_GRAPH_MS / per_iter))
        del g
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        g.replay()
        torch.cuda.synchronize()

    out = []
    for _ in range(repeats):
        s, e = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        s.record(); g.replay(); e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return min(out) / iters


def fwd_bwd_fn(fwd, inputs, warm, rep, iters):
    def step():
        out = fwd()
        if isinstance(out, tuple):
            out = out[0]
        torch.autograd.grad(out, inputs, torch.ones_like(out))
    return time_fn(step, warm, rep, iters)


def _iters_for(direction, iters):
    return iters if direction == "fwd" else max(1, iters // BWD_ITER_DIV)

# --- one measurement per kernel -------------------------------------------------------------

def _gemm_point(m, n, k, dt, direction, warm, rep, iters):
    grad = direction == "fwd_bwd"
    a = torch.randn(m, k, device="cuda", dtype=dt, requires_grad=grad)
    b = torch.randn(k, n, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: torch.mm(a, b), [a, b], warm, rep, iters)
    c = torch.empty(m, n, device="cuda", dtype=dt)
    return time_fn(lambda: torch.mm(a, b, out=c), warm, rep, iters)


def _bmm_point(batch, m, n, k, dt, direction, warm, rep, iters):
    grad = direction == "fwd_bwd"
    a = torch.randn(batch, m, k, device="cuda", dtype=dt, requires_grad=grad)
    b = torch.randn(batch, k, n, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: torch.bmm(a, b), [a, b], warm, rep, iters)
    c = torch.empty(batch, m, n, device="cuda", dtype=dt)
    return time_fn(lambda: torch.bmm(a, b, out=c), warm, rep, iters)


def _conv_point(bs, cin, cout, hw, k, stride, pad, groups, direction, dt, warm, rep, iters):
    x = torch.randn(bs, cin, hw, hw, device="cuda", dtype=dt)
    w = torch.randn(cout, cin // groups, k, k, device="cuda", dtype=dt)
    out_hw = (hw + 2 * pad - k) // stride + 1
    gy = torch.randn(bs, cout, out_hw, out_hw, device="cuda", dtype=dt)
    torch.cuda.synchronize()
    if direction == "fprop":
        fn = lambda: torch.nn.functional.conv2d(x, w, stride=stride, padding=pad, groups=groups)
    elif direction == "dgrad":
        fn = lambda: torch.nn.grad.conv2d_input(x.shape, w, gy, stride=stride, padding=pad,
                                                groups=groups)
    else:
        fn = lambda: torch.nn.grad.conv2d_weight(x, w.shape, gy, stride=stride, padding=pad,
                                                 groups=groups)
    return time_fn(fn, warm, rep, iters)


def _conv1d_point(bs, cin, cout, length, k, stride, direction, dt, warm, rep, iters):
    x = torch.randn(bs, cin, length, device="cuda", dtype=dt)
    w = torch.randn(cout, cin, k, device="cuda", dtype=dt)
    out_len = (length - k) // stride + 1
    gy = torch.randn(bs, cout, out_len, device="cuda", dtype=dt)
    torch.cuda.synchronize()
    if direction == "fprop":
        fn = lambda: torch.nn.functional.conv1d(x, w, stride=stride)
    elif direction == "dgrad":
        fn = lambda: torch.nn.grad.conv1d_input(x.shape, w, gy, stride=stride)
    else:
        fn = lambda: torch.nn.grad.conv1d_weight(x, w.shape, gy, stride=stride)
    return time_fn(fn, warm, rep, iters)


def _attn_point(bs, qh, kvh, seq_q, seq_kv, head_dim, causal, dt, direction, warm, rep, iters):
    grad = direction == "fwd_bwd"
    q = torch.randn(bs, qh, seq_q, head_dim, device="cuda", dtype=dt, requires_grad=grad)
    k = torch.randn(bs, kvh, seq_kv, head_dim, device="cuda", dtype=dt, requires_grad=grad)
    v = torch.randn(bs, kvh, seq_kv, head_dim, device="cuda", dtype=dt, requires_grad=grad)
    fn = lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k, v, dropout_p=0.0, is_causal=causal, enable_gqa=qh != kvh)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(fn, [q, k, v], warm, rep, iters)
    return time_fn(fn, warm, rep, iters)


def _pool_point(bs, c, hw, k, stride, kind, dt, direction, warm, rep, iters):
    grad = direction == "fwd_bwd"
    x = torch.randn(bs, c, hw, hw, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if kind == "max":
        fn = lambda: torch.nn.functional.max_pool2d(x, k, stride, k // 2)
    elif kind == "avg":
        fn = lambda: torch.nn.functional.avg_pool2d(x, k, stride, k // 2)
    else:
        fn = lambda: torch.nn.functional.adaptive_avg_pool2d(x, 1)
    if grad:
        return fwd_bwd_fn(fn, [x], warm, rep, iters)
    return time_fn(fn, warm, rep, iters)


def _elementwise_point(numel, dt, direction, warm, rep, iters):
    grad = direction == "fwd_bwd"
    a = torch.randn(numel, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: torch.sigmoid(a), [a], warm, rep, iters)
    c = torch.empty(numel, device="cuda", dtype=dt)
    return time_fn(lambda: torch.sigmoid(a, out=c), warm, rep, iters)


def _rnn_point(bs, seq, inp, hidden, layers, bidir, kind, dt, direction, warm, rep, iters):
    cls = {"lstm": torch.nn.LSTM, "gru": torch.nn.GRU, "rnn": torch.nn.RNN}[kind]
    grad = direction == "fwd_bwd"
    m = cls(inp, hidden, num_layers=layers, batch_first=True,
            bidirectional=bidir, dropout=0.0).cuda().to(dt)
    m.flatten_parameters()
    x = torch.randn(bs, seq, inp, device="cuda", dtype=dt, requires_grad=grad)
    torch.cuda.synchronize()
    if grad:
        return fwd_bwd_fn(lambda: m(x), [x] + list(m.parameters()), warm, rep, iters)
    m.eval()
    with torch.no_grad():
        return time_fn(lambda: m(x), warm, rep, iters)

# --- driver ---------------------------------------------------------------------------------

def configure():
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)


def _emit(rows, probe, dtype, size, direction, kind, causal, fn):
    try:
        ms, oom = fn(), False
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and "out of memory" not in str(e).lower():
            raise
        ms, oom = math.nan, True
    torch.cuda.empty_cache()
    rows.append((torch.cuda.get_device_name(0).replace(" ", "_"), probe, dtype, size,
                 direction, kind, causal, ms, oom))


def _collect(sink, emit=None):
    emit = emit or (lambda s_, *k: s_.append(k[:6]))   # drop the thunk, keep the key
    for name, dt in DTYPES.items():
        for grid, fam in ((GEMM_GRID, "gemm"), (GEMM_EXT_GRID, "gemm_ext")):
            for (m, n, k), (w, r, i) in grid:
                for d in DIRECTIONS:
                    emit(sink, fam, name, f"m{m}n{n}k{k}", d, None, None,
                          lambda m=m, n=n, k=k, d=d, dt=dt, w=w, r=r, i=i:
                              _gemm_point(m, n, k, dt, d, w, r, _iters_for(d, i)))

        for (b, m, n, k), (w, r, i) in BMM_GRID:
            for d in DIRECTIONS:
                emit(sink, "bmm", name, f"b{b}m{m}n{n}k{k}", d, None, None,
                      lambda b=b, m=m, n=n, k=k, d=d, dt=dt, w=w, r=r, i=i:
                          _bmm_point(b, m, n, k, dt, d, w, r, _iters_for(d, i)))

        for (bs, cin, cout, hw, k, st, pad, gr), (w, r, i) in CONV_GRID:
            for d in CONV_DIRECTIONS:
                emit(sink, "conv", name, f"{cin}x{cout}c{hw}k{k}s{st}g{gr}", d, None, None,
                      lambda bs=bs, cin=cin, cout=cout, hw=hw, k=k, st=st, pad=pad, gr=gr,
                             d=d, dt=dt, w=w, r=r, i=i:
                          _conv_point(bs, cin, cout, hw, k, st, pad, gr, d, dt, w, r, i))

        for (bs, cin, cout, hw, k, st, pad, gr), (w, r, i) in CONV_EXT_GRID:
            for d in CONV_DIRECTIONS:
                emit(sink, "conv_ext", name,
                      f"b{bs}c{cin}o{cout}hw{hw}k{k}s{st}p{pad}g{gr}", d, None, None,
                      lambda bs=bs, cin=cin, cout=cout, hw=hw, k=k, st=st, pad=pad, gr=gr,
                             d=d, dt=dt, w=w, r=r, i=i:
                          _conv_point(bs, cin, cout, hw, k, st, pad, gr, d, dt, w, r, i))

        for (bs, cin, cout, length, k, st), (w, r, i) in CONV1D_GRID:
            for d in CONV_DIRECTIONS:
                emit(sink, "conv1d", name, f"b{bs}c{cin}o{cout}l{length}k{k}s{st}",
                      d, None, None,
                      lambda bs=bs, cin=cin, cout=cout, length=length, k=k, st=st, d=d, dt=dt,
                             w=w, r=r, i=i:
                          _conv1d_point(bs, cin, cout, length, k, st, d, dt, w, r, i))

        for (bs, qh, kvh, sq, sk, hd), (w, r, i) in ATTN_GRID:
            for causal in ((False, True) if sq == sk else (False,)):
                for d in DIRECTIONS:
                    emit(sink, "attn", name, f"q{sq}k{sk}h{qh}/{kvh}d{hd}", d, None, causal,
                          lambda bs=bs, qh=qh, kvh=kvh, sq=sq, sk=sk, hd=hd, causal=causal,
                                 d=d, dt=dt, w=w, r=r, i=i:
                              _attn_point(bs, qh, kvh, sq, sk, hd, causal, dt, d, w, r,
                                          _iters_for(d, i)))

        for numel in ELEMENTWISE_NUMEL:
            w, r, i = next((a, b, c) for cap, a, b, c in ELEMENTWISE_TIERS if numel <= cap)
            for d in DIRECTIONS:
                emit(sink, "elementwise", name, numel, d, None, None,
                      lambda numel=numel, d=d, dt=dt, w=w, r=r, i=i:
                          _elementwise_point(numel, dt, d, w, r, _iters_for(d, i)))

        for (bs, c, hw, k, st), (w, r, i) in POOL_GRID:
            for kind in POOL_KINDS:
                for d in DIRECTIONS:
                    emit(sink, "pool", name, f"{c}c{hw}k{k}s{st}", d, kind, None,
                          lambda bs=bs, c=c, hw=hw, k=k, st=st, kind=kind, d=d, dt=dt,
                                 w=w, r=r, i=i:
                              _pool_point(bs, c, hw, k, st, kind, dt, d, w, r, _iters_for(d, i)))

        for (bs, seq, inp, hid, layers, bidir), (w, r, i) in RNN_GRID:
            for kind in RNN_KINDS:
                for d in DIRECTIONS:
                    emit(sink, "rnn", name,
                          f"b{bs}s{seq}i{inp}h{hid}l{layers}d{2 if bidir else 1}", d, kind, None,
                          lambda bs=bs, seq=seq, inp=inp, hid=hid, layers=layers, bidir=bidir,
                                 kind=kind, d=d, dt=dt, w=w, r=r, i=i:
                              _rnn_point(bs, seq, inp, hid, layers, bidir, kind, dt, d, w, r,
                                         _iters_for(d, i)))
    return None


def _configs():
    """The (probe, dtype, size, direction, kind, causal) keys to measure, read off
    REFERENCE_CSV. That file defines the experiment; nothing here re-derives it."""
    ref = pd.read_csv(REFERENCE_CSV)

    def norm(v):
        if pd.isna(v):
            return None
        return True if v in (True, "True") else False if v in (False, "False") else v

    return {(r["probe"], r["dtype"], str(r["size"]), r["direction"],
             norm(r["kind"]), norm(r["causal"])) for _, r in ref.iterrows()}


def run():
    """Measure every configuration in REFERENCE_CSV. Returns the DataFrame."""
    configure()
    wanted = _configs()
    rows = []
    seen = set()

    def emit(sink, probe, dtype, size, direction, kind, causal, fn):
        key = (probe, dtype, str(size), direction, kind, causal)
        if key not in wanted:
            return
        seen.add(key)
        _emit(sink, probe, dtype, size, direction, kind, causal, fn)

    _collect(rows, emit)
    if wanted - seen:
        raise RuntimeError(f"{len(wanted - seen)} configs in the reference CSV are not in this "
                           f"grid, e.g. {sorted(wanted - seen, key=str)[:3]}")
    return pd.DataFrame(rows, columns=COLS)
