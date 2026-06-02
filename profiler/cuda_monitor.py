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


_MAX_FIELDS = ["memory_used_pct", "memory_used_mb", "memory_used_mb_nvml",
               "power_w", "gpu_util_pct"]

_SUMMARY_KEYS = [
    "oom", "sample_count", "duration_s", "nvml",
    "max_memory_used_pct", "max_memory_used_mb", "max_memory_used_mb_nvml",
    "max_power_w", "avg_power_w", "energy_j",
    "max_gpu_util_pct", "avg_gpu_util_pct",
]


def build_row(hp, phase: str, metrics: dict | None = None,
              error: str = "", avg_ms: float | None = None) -> dict:
    """Build one result row from config + phase + monitor summary / error. No I/O."""
    config = asdict(hp) if is_dataclass(hp) else dict(hp)
    return {
        **config,
        "phase": phase,
        "error": error,
        "avg_time_ms": None if avg_ms is None else avg_ms,
        **{k: None if metrics is None else metrics.get(k) for k in _SUMMARY_KEYS},
    }


@dataclass
class CUDAMonitor:
    """Polls torch.cuda.mem_get_info() + NVML (power, util, full-device memory) in a background thread."""
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
            try:
                rec["power_w"] = pynvml.nvmlDeviceGetPowerUsage(_NVML_HANDLE) / 1000.0
            except Exception:
                pass
            try:
                rec["gpu_util_pct"] = pynvml.nvmlDeviceGetUtilizationRates(_NVML_HANDLE).gpu
            except Exception:
                pass
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

        powers = [(t, r["power_w"]) for t, r in zip(timestamps, records) if r.get("power_w") is not None]
        if powers:
            summary["avg_power_w"] = sum(p for _, p in powers) / len(powers)
            energy_j = 0.0
            for i in range(1, len(powers)):
                dt = powers[i][0] - powers[i-1][0]
                p_avg = (powers[i-1][1] + powers[i][1]) / 2
                energy_j += p_avg * dt
            summary["energy_j"] = energy_j

        utils = [r["gpu_util_pct"] for r in records if r.get("gpu_util_pct") is not None]
        if utils:
            summary["avg_gpu_util_pct"] = sum(utils) / len(utils)

        return summary
