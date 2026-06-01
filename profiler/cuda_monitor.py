import csv
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path

import torch


_OUTPUT_FIELDS = ["memory_used_pct", "memory_used_mb"]

_SUMMARY_KEYS = [
    "oom", "sample_count",
    "max_memory_used_pct", "max_memory_used_mb",
]


def record_run(csv_path: Path, hp, phase: str, metrics: dict | None = None,
               error: str = "", avg_ms: float | None = None) -> None:
    """Append one row to the workload CSV with config + phase + monitor summary / error."""
    config = asdict(hp) if is_dataclass(hp) else dict(hp)
    row = {
        **config,
        "phase": phase,
        "error": error,
        "avg_time_ms": "" if avg_ms is None else avg_ms,
        **{k: "" if metrics is None else metrics.get(k, "") for k in _SUMMARY_KEYS},
    }
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


@dataclass
class CUDAMonitor:
    """Polls torch.cuda.mem_get_info() in a background thread. No NVML dependency."""
    interval_ms: int = 20
    _records: list = field(default_factory=list, init=False, repr=False)
    _thread: threading.Thread = field(default=None, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)

    def start(self) -> None:
        self._records.clear()
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll(self) -> None:
        interval_s = self.interval_ms / 1000.0
        while True:
            free, total = torch.cuda.mem_get_info()
            used = total - free
            self._records.append({
                "memory_used_pct": 100.0 * used / total if total else 0.0,
                "memory_used_mb": used / (1024 ** 2),
            })
            if self._stop.wait(interval_s):
                break

    def stop(self, oom: bool = False) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1)
        records = list(self._records)
        summary = {"oom": oom, "sample_count": len(records)}
        for f in _OUTPUT_FIELDS:
            values = [r[f] for r in records if f in r]
            summary[f"max_{f}"] = max(values) if values else None
        return summary
