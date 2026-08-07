import math
import statistics

import torch

from torch._C._autograd import DeviceType   # torch.autograd re-exports it but does not declare it
from torch.profiler import ProfilerActivity, profile


# A replay still costs one host-side cudaGraphLaunch, a few microseconds that capture does not
# remove -- it only removes the per-kernel dispatch inside the graph. When the whole graph is a
# couple of microseconds of work, that launch is a large fraction of the measurement and the
# result tracks the node's CPU. Seen across three RTX 5090s: rows above 1ms agreed to 1.6%,
# rows below 10us disagreed by up to 456%, and the weakest-CPU node was the slow one on 196 of
# the 358 disagreeing rows. So iters is raised until the captured graph spans MIN_GRAPH_MS,
# which amortises the launch. Capped because every captured iteration holds its own
# intermediates in the graph's private memory pool.
MIN_GRAPH_MS = 2.0
MAX_ITERS = 200


def time_fn(fn, warmup: int, repeats: int, iters: int) -> float:
    """Median per-iteration CUDA graph ms for function fn.

    iters is a floor: a cheap fn gets more iterations per capture so one graph launch is
    amortised over enough work (see MIN_GRAPH_MS). Callers whose step already runs for
    milliseconds are unaffected."""
    # warmup on a side stream and waits for it to finish
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    # generate graph once, replay it later in timed window
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            fn()

    g.replay()
    torch.cuda.synchronize()

    # Size the graph from a replay rather than from the warmup: warming up runs eagerly and its
    # first iteration carries cuBLAS/cuDNN handle setup, which overestimated a 2us kernel by 40x
    # and left iters effectively unraised. One timed replay costs little and is accurate enough.
    s0 = torch.cuda.Event(enable_timing=True)
    e0 = torch.cuda.Event(enable_timing=True)
    s0.record()
    g.replay()
    e0.record()
    torch.cuda.synchronize()
    per_iter = s0.elapsed_time(e0) / iters
    if per_iter > 0 and per_iter * iters < MIN_GRAPH_MS:
        iters = min(MAX_ITERS, math.ceil(MIN_GRAPH_MS / per_iter))
        del g                       # release the old graph's private pool before recapturing
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        g.replay()
        torch.cuda.synchronize()

    # repeats timed windows, median so one perturbed window cannot set the result
    out = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out) / iters


def kernel_time_fn(fn, warmup: int, repeats: int, iters: int) -> tuple[float, int]:
    """(median per-iteration ms of GPU kernel time, kernels per iteration) for one step of fn.

    For steps a CUDA graph cannot capture -- object detection, where torchvision's
    GeneralizedRCNNTransform and the HF detection loss both build host tensors inside the
    forward. Returns the same quantity time_fn does, read off the profiler instead of by
    replaying a capture.

    Reads high against time_fn by a roughly fixed 2.4-6.9us per kernel. That is a per-kernel
    cost, not a percentage: it tracks mean kernel duration (10% of a step built from 80us
    kernels, 78% of one built from 6us kernels) and not kernel count. The count is returned so
    callers can record it and correct or fit the bias rather than absorb it silently."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # repeats timed windows, median so one perturbed window cannot set the result
    out, counts = [], []
    for _ in range(repeats):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        # Only DeviceType.CUDA rows are kernels. The aten:: rows carry the same device time
        # attributed up the call tree, so summing every row double-counts it (~2x).
        kernels = [e for e in prof.key_averages() if e.device_type == DeviceType.CUDA]
        out.append(sum(e.self_device_time_total for e in kernels) / 1000.0)
        counts.append(sum(e.count for e in kernels))
    return statistics.median(out) / iters, round(statistics.median(counts) / iters)
