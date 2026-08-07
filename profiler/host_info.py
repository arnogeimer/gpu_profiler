"""Host identity for one profiling node: the CPU it runs on, the GPU it drives, and the
stock-TDP table used to decide whether that GPU is admissible.

The CPU fields matter because launch-bound rows (low avg_gpu_util_pct) have their runtime
set by the host rather than the card, and Salad allocates heterogeneous hosts. The TDP table
catches hosts running a custom sub-spec power limit, whose measurements would understate the
silicon.
"""

import json
import os
import platform
import re
from pathlib import Path

import torch


# CUDA cores per SM, indexed by compute capability (major, minor)
_CUDA_CORES_PER_SM = {
    (3, 0): 192, (3, 5): 192, (3, 7): 192,
    (5, 0): 128, (5, 2): 128, (5, 3): 128,
    (6, 0): 64,  (6, 1): 128, (6, 2): 128,
    (7, 0): 64,  (7, 2): 64,  (7, 5): 64,
    (8, 0): 64,  (8, 6): 128, (8, 7): 128, (8, 9): 128,
    (9, 0): 128,
    (10, 0): 128,
    (12, 0): 128,   # Blackwell consumer (RTX 50xx)
}


def _cpu_info() -> dict:
    """Host CPU identity and clock ceiling.

    Rows where the step is launch-bound rather than compute-bound (visible as a low
    avg_gpu_util_pct) have their runtime set by this CPU, not the GPU. Salad allocates
    heterogeneous hosts, so without these fields such rows are unexplainable variance:
    the same GPU and workload can produce different timings with nothing to distinguish them."""
    info: dict = {"cpu_count": os.cpu_count()}
    try:
        text = Path("/proc/cpuinfo").read_text()
        name = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
        if name:
            info["cpu_name"] = name.group(1).strip()
        # Physical cores, which bound dispatch throughput better than the SMT count above.
        ids = set(re.findall(r"^core id\s*:\s*(\d+)$", text, re.M))
        if ids:
            info["cpu_physical_cores"] = len(ids)
    except Exception:
        pass
    info.setdefault("cpu_name", platform.processor() or platform.machine() or None)
    try:
        khz = Path("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read_text().strip()
        info["cpu_max_mhz"] = int(khz) / 1000.0
    except Exception:
        pass
    return info


def get_gpu_name() -> str:
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown_gpu"
    return name.replace(" ", "_")


def get_gpu_uuid() -> str:
    """Persistent hardware UUID for the physical GPU (NVML). Same value across reboots/rentals
    of the same card — uniquely identifies the silicon, not the Salad container.
    Format: 'GPU-xxxxxxxx-...'; we strip the 'GPU-' prefix and keep the first 8 hex chars
    so the filename stays short and readable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        uuid_raw = pynvml.nvmlDeviceGetUUID(h)
        if isinstance(uuid_raw, bytes):
            uuid_raw = uuid_raw.decode()
        pynvml.nvmlShutdown()
        return uuid_raw.removeprefix("GPU-").split("-")[0]
    except Exception:
        return "unknownuuid"


def _measure_tflops(dtype: torch.dtype, n: int = 8192, iters: int = 10, warmup: int = 3) -> float | None:
    """Time a large square matmul to estimate achieved TFLOPS for the given dtype.
    fp32 routes through CUDA cores; fp16/bf16 routes through Tensor Cores (Volta+)."""
    A = B = C = None
    try:
        A = torch.randn(n, n, device="cuda", dtype=dtype)
        B = torch.randn(n, n, device="cuda", dtype=dtype)
        for _ in range(warmup):
            C = A @ B
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            C = A @ B
        end.record()
        torch.cuda.synchronize()

        elapsed_s = start.elapsed_time(end) / 1000.0
        flops = 2.0 * (n ** 3) * iters
        return flops / elapsed_s / 1e12
    except Exception:
        return None
    finally:
        del A, B, C
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def collect() -> dict:
    info = _cpu_info()
    try:
        props = torch.cuda.get_device_properties(0)
        cc = (props.major, props.minor)
        cores_per_sm = _CUDA_CORES_PER_SM.get(cc)
        info.update({
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "sm_count": props.multi_processor_count,
            "cores_per_sm": cores_per_sm,
            "cuda_cores": props.multi_processor_count * cores_per_sm if cores_per_sm else None,
            "vram_total_mb": props.total_memory // (1024 ** 2),
            "l2_cache_kb": props.L2_cache_size // 1024 if hasattr(props, "L2_cache_size") else None,
        })
    except Exception:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        for key, fn in [
            ("uuid",                  lambda: pynvml.nvmlDeviceGetUUID(h)),
            ("power_limit_w",         lambda: pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000.0),
            ("max_graphics_clock_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_GRAPHICS)),
            ("max_sm_clock_mhz",       lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
            ("max_memory_clock_mhz",   lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM)),
            ("driver_version",         lambda: pynvml.nvmlSystemGetDriverVersion()),
        ]:
            try:
                v = fn()
                info[key] = v.decode() if isinstance(v, bytes) else v
            except Exception:
                pass
        pynvml.nvmlShutdown()
    except Exception:
        pass

    if torch.cuda.is_available():
        info["measured_tflops_fp32"] = _measure_tflops(torch.float32)
        info["measured_tflops_fp16"] = _measure_tflops(torch.float16)

    return info


def save(output_path: Path) -> None:
    output_path.write_text(json.dumps(collect(), indent=2))


# torch.cuda.get_device_name(0).replace(" ", "_") -> stock TDP (W)
STOCK_TDP_W: dict[str, float] = {
    # Turing (20xx)
    "NVIDIA_GeForce_RTX_2070":             175,
    "NVIDIA_GeForce_RTX_2070_SUPER":       215,
    "NVIDIA_GeForce_RTX_2080":             215,
    "NVIDIA_GeForce_RTX_2080_SUPER":       250,
    "NVIDIA_GeForce_RTX_2080_Ti":          250,
    # Ampere (30xx)
    "NVIDIA_GeForce_RTX_3050":             130,
    "NVIDIA_GeForce_RTX_3060":             170,
    "NVIDIA_GeForce_RTX_3060_Ti":          200,
    "NVIDIA_GeForce_RTX_3070":             220,
    "NVIDIA_GeForce_RTX_3070_Ti":          290,
    "NVIDIA_GeForce_RTX_3080":             320,
    "NVIDIA_GeForce_RTX_3080_Ti":          350,
    "NVIDIA_GeForce_RTX_3090":             350,
    "NVIDIA_GeForce_RTX_3090_Ti":          450,
    # Ada (40xx)
    "NVIDIA_GeForce_RTX_4060":             115,
    "NVIDIA_GeForce_RTX_4060_Ti":          165,
    "NVIDIA_GeForce_RTX_4070":             200,
    "NVIDIA_GeForce_RTX_4070_SUPER":       220,
    "NVIDIA_GeForce_RTX_4070_Ti":          285,
    "NVIDIA_GeForce_RTX_4070_Ti_SUPER":    285,
    "NVIDIA_GeForce_RTX_4080":             320,
    "NVIDIA_GeForce_RTX_4080_SUPER":       320,
    "NVIDIA_GeForce_RTX_4090":             450,
    # Blackwell (50xx)
    "NVIDIA_GeForce_RTX_5060":             145,
    "NVIDIA_GeForce_RTX_5060_Ti":          180,
    "NVIDIA_GeForce_RTX_5070":             250,
    "NVIDIA_GeForce_RTX_5070_Ti":          300,
    "NVIDIA_GeForce_RTX_5080":             360,
    "NVIDIA_GeForce_RTX_5090":             575,
    "NVIDIA_GeForce_RTX_5080_Laptop_GPU":  150,
    "NVIDIA_GeForce_RTX_5090_Laptop_GPU":  150,
    # Workstation
    "NVIDIA_RTX_A5000":                    230,
    "NVIDIA_RTX_A6000":                    300,
}


def check_full_power(gpu_name: str, measured_power_limit_w: float | None,
                     threshold: float = 0.95) -> tuple[bool, str]:
    """Returns (ok, reason). ok=False means this host is running below threshold of stock TDP."""
    if measured_power_limit_w is None:
        return False, "no power limit reported by NVML — cannot verify undervolting"
    spec = STOCK_TDP_W.get(gpu_name)
    if spec is None:
        return True, f"stock TDP unknown for {gpu_name} — cannot verify, proceeding"
    pct = measured_power_limit_w / spec
    if pct < threshold or pct > 1.10:
        return False, (f"measured power_limit_w={measured_power_limit_w:.0f}W is "
                       f"{pct*100:.1f}% of stock TDP {spec:.0f}W (threshold {threshold*100:.0f}%)")
    return True, f"power_limit_w={measured_power_limit_w:.0f}W is {pct*100:.1f}% of stock TDP {spec:.0f}W"
