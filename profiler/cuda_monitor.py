import threading
import time
from dataclasses import asdict, dataclass, field, is_dataclass

import torch

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _NVML_OK = True
except Exception as e:
    pynvml = None
    _NVML_HANDLE = None
    _NVML_OK = False
    print(f"[cuda_monitor] NVML unavailable: {e}")


_MAX_FIELDS = ["memory_used_pct", "memory_used_mb", "memory_used_mb_nvml"]

# Power and GPU utilisation are deliberately absent. NVML refreshes those counters on the
# driver's own cadence rather than per query, so polling every 10ms just re-reads a cached
# value; averaging over the ~0.85s a config takes averages a handful of essentially
# arbitrarily-aligned driver updates. Measured on one fixed config whose timing was stable to
# 0.2%, avg_power_w ranged 42-145W and avg_gpu_util_pct 0-68% purely with how long the GPU had
# been idle beforehand. Memory is kept because mem_get_info is an instantaneous query.
_SUMMARY_KEYS = [
    "oom", "sample_count", "duration_s", "nvml",
    "max_memory_used_pct", "max_memory_used_mb", "max_memory_used_mb_nvml",
]


def progress_line(model: str, i: int, total: int, label: str, prev_start: float | None) -> float:
    """Print one progress line per model and return its start time, to pass in on the next call.

    One line per model rather than per config: a full sweep is ~4000 configs, and that much
    output was suspected of truncating a node's logs. The timestamp and previous duration are
    here because the reduction cuts both ways -- with 20+ minutes between lines on a large
    model, silence alone cannot distinguish a slow node from a hung one, so the last line
    printed has to carry enough to tell them apart."""
    now = time.time()
    took = ""
    if prev_start is not None:
        secs = now - prev_start
        took = (f"  | previous {int(secs // 60)}m{int(secs % 60):02d}s" if secs >= 60
                else f"  | previous {secs:.1f}s")
    print(f"[{time.strftime('%H:%M:%S')}] Started training model {model}"
          f"  ({i}/{total} {label}){took}", flush=True)
    return now


def build_row(hp, phase: str, metrics: dict | None = None,
              error: str = "", avg_ms: float | None = None,
              timing_method: str = "", kernel_count: int | None = None) -> dict:
    """Build one result row from config + phase + monitor summary / error. No I/O.

    timing_method records which instrument produced avg_time_ms, because the suite uses two.
    "cuda_graph" (profiler.time_fn) replays a captured graph and is exact; "kernel_sum"
    (profiler.kernel_time_fn) sums the profiler's per-kernel device time and reads high by a
    roughly fixed 2.4-6.9us per kernel, which is 10% of a step built from 80us kernels but 78%
    of one built from 6us kernels. kernel_count is recorded alongside so that bias can be
    corrected or fitted rather than silently absorbed."""
    config = asdict(hp) if is_dataclass(hp) else dict(hp)
    return {
        **config,
        "phase": phase,
        "error": error,
        "avg_time_ms": None if avg_ms is None else avg_ms,
        "timing_method": timing_method,
        "kernel_count": kernel_count,
        **{k: None if metrics is None else metrics.get(k) for k in _SUMMARY_KEYS},
    }


@dataclass
class CUDAMonitor:
    """Polls torch.cuda.mem_get_info() + NVML full-device memory in a background thread."""
    interval_ms: int = 20
    _records: list = field(default_factory=list, init=False, repr=False)
    _timestamps: list = field(default_factory=list, init=False, repr=False)
    _thread: threading.Thread = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        self._records.clear()
        self._timestamps.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _sample(self) -> dict:
        free, total = torch.cuda.mem_get_info()
        used = total - free
        rec = {
            "memory_used_pct": 100.0 * used / total if total else 0.0,
            "memory_used_mb": used / (1024 ** 2),
        }
        if _NVML_OK:
            # Whole-device memory, which catches other tenants that torch cannot see.
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(_NVML_HANDLE)
                rec["memory_used_mb_nvml"] = mem.used / (1024 ** 2)
            except Exception:
                pass
        return rec

    def _poll(self) -> None:
        interval_s = self.interval_ms / 1000.0
        while True:
            self._timestamps.append(time.monotonic())
            self._records.append(self._sample())
            if self._stop.wait(interval_s):
                break

    def stop(self, oom: bool = False) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        records = list(self._records)
        timestamps = list(self._timestamps)
        summary = {
            "oom": oom,
            "sample_count": len(records),
            "duration_s": (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0.0,
            "nvml": _NVML_OK,
        }
        for key in _MAX_FIELDS:
            values = [r[key] for r in records if r.get(key) is not None]
            summary[f"max_{key}"] = max(values) if values else None
        return summary
