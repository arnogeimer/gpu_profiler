import math

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

# Capture cannot reclaim cache the way eager execution can -- the allocator is not allowed to
# cudaFree an unused cached block while a capture is active (see pytorch/pytorch#159594), so a
# config that just barely fits under eager warmup can still overrun once addresses are locked in
# under torch.cuda.graph(). When THAT happens, the failure does not surface as an ordinary
# OutOfMemoryError -- it corrupts the allocator's own graph-pool bookkeeping instead
# (`!handles_.at(i) INTERNAL ASSERT FAILED`, an open, unresolved PyTorch bug: pytorch/pytorch
# #166234, #68985) and every later capture in the process fails identically. convnextv2_base
# fp32 at 128px batch 64 took a whole node's remaining sweep down this way, logged at 100%
# device memory. _clear_capture_state's repair does not help here: it targets a different piece
# of state (the RNG generator's capture-bound flag), and a repair attempt is itself a capture,
# which walks the same corrupted bookkeeping and fails the same way.
#
# So this is checked BEFORE capture rather than repaired after. If the device is already this
# full once warmup -- ordinary eager execution, which CAN reclaim -- has settled, capture is not
# attempted at all, and the config is reported as an ordinary, uncorrupting OOM instead.
CAPTURE_MEM_GUARD_PCT = 99.0


def _check_capture_headroom() -> None:
    """Raise a clean OutOfMemoryError if capture is unsafe to attempt right now.

    Deliberately the same exception type torch itself raises on a refused allocation: every
    caller up the stack (cuda_monitor.is_oom, and every workload's except block) already treats
    torch.cuda.OutOfMemoryError as a normal, recoverable OOM, so this needs no changes anywhere
    else to be handled correctly."""
    free, total = torch.cuda.mem_get_info()
    used_pct = 100.0 * (total - free) / total if total else 0.0
    if used_pct >= CAPTURE_MEM_GUARD_PCT:
        raise torch.cuda.OutOfMemoryError(
            f"skipping graph capture at {used_pct:.1f}% device memory used after warmup "
            f"(guard is {CAPTURE_MEM_GUARD_PCT}%) -- see profiler.CAPTURE_MEM_GUARD_PCT")


def _clear_capture_state() -> None:
    """Undo the RNG damage a failed capture leaves behind. Call after any capture that raised.

    A capture that raises part-way through leaves the CUDA RNG generator registered as
    capture-bound, and nothing in the normal teardown clears it: every later RNG op in the
    process -- dropout, randn, anything seeded -- then dies with "Offset increment outside graph
    capture encountered unexpectedly", forever, on a context that is otherwise healthy.

    It is a process-wide fault from a per-config failure, so it does not stay inside the workload
    that caused it. That is how it was found: bloom-1b1 and Phi-3-mini fail capture on a host-side
    copy (535 configs), and object_detection -- which runs last, and cannot even use capture --
    lost 5 076 of 5 184 rows on all 16 cards to an error raised by a workload that had already
    finished. Nothing distinguished it in the logs from a broken detection sweep.

    Of everything tried, only completing one clean capture clears it: synchronize, empty_cache,
    deleting the graph, manual_seed, and set_offset(0) all leave the generator poisoned. So the
    repair is to capture something trivial and throw it away. Best-effort by design -- if the
    context is genuinely dead this cannot help, and raising here would mask the original error."""
    try:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            torch.zeros(1, device="cuda").add_(1.0)
        g.replay()
        torch.cuda.synchronize()
    except Exception:
        pass


def time_fn(fn, warmup: int, repeats: int, iters: int) -> float:
    """Fastest per-iteration CUDA graph ms for function fn.

    iters is a floor: a cheap fn gets more iterations per capture so one graph launch is
    amortised over enough work (see MIN_GRAPH_MS). Callers whose step already runs for
    milliseconds are unaffected.

    Any failure is repaired before it propagates, so a config that cannot be captured costs its
    own row and nothing else -- see _clear_capture_state for what it would otherwise cost."""
    try:
        return _time_fn(fn, warmup, repeats, iters)
    except Exception:
        _clear_capture_state()
        raise


def _time_fn(fn, warmup: int, repeats: int, iters: int) -> float:
    # warmup on a side stream and waits for it to finish
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    _check_capture_headroom()

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
        _check_capture_headroom()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(iters):
                fn()
        g.replay()
        torch.cuda.synchronize()

    # repeats timed windows, minimum rather than median. Noise here is one-sided -- contention,
    # a clock drop or a scheduling delay can only make a replay slower, never faster -- so the
    # fastest window is the cleanest estimate of what the device can do. It also needs only one
    # uncontended repeat where a median needs six, which matters because the interference seen
    # in practice is intermittent: on a quiet machine min and median agree to ~1.6%, but on a
    # config where 9 of 10 repeats were disturbed the median read 9% high and the min did not.
    out = []
    for _ in range(repeats):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return min(out) / iters


def kernel_time_fn(fn, warmup: int, repeats: int, iters: int) -> tuple[float, int]:
    """(fastest per-iteration ms of GPU kernel time, kernels per iteration) for one step of fn.

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

    # repeats timed windows, minimum for the same reason as time_fn: interference only ever
    # adds time. The kernel count reported is the one from that fastest window, so the two
    # numbers describe the same replay.
    out, counts = [], []
    for _ in range(repeats):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        # Only DeviceType.CUDA rows are kernels. The aten:: rows carry the same device time
        # attributed up the call tree, so summing every row double-counts it (~2x).
        #
        # Raw events rather than key_averages(): that method groups every event by key to build
        # a summary table, and we immediately discard the grouping to take a sum and a count.
        # The aggregate sum equals the sum of the parts, so the two agree exactly -- 243.739 ms
        # and 3650 kernels either way on fasterrcnn_resnet50_fpn 800px bs2.
        #
        # Worth about 20% of this workload's runtime, not more. Whichever accessor is called
        # first pays ~1.1s for the profiler's lazy post-processing of the raw trace, and that is
        # unavoidable; key_averages() then adds ~0.6s of grouping on top, and only that part is
        # saved here. Measured both ways round on one window: events-first gave 1.14s/0.60s,
        # key_averages-first gave 2.00s/0.04s. A naive one-order benchmark reads this as a 37x
        # win, which it is not -- end to end a config went from 24s to 19s.
        kernels = [e for e in prof.events() if e.device_type == DeviceType.CUDA]
        out.append(sum(e.self_device_time_total for e in kernels) / 1000.0)
        counts.append(len(kernels))
    best = min(range(len(out)), key=out.__getitem__)
    return out[best] / iters, round(counts[best] / iters)
