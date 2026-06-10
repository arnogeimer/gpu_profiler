import io
import json
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


def get_gpu_name() -> str:
    try:
        name = torch.cuda.get_device_name(0)
    except Exception:
        name = "unknown_gpu"
    return name.replace(" ", "_")


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
    info = {}
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


def upload_to_hf(token: str, repo_id: str, gpu_name: str) -> None:
    """Push the collected GPU info JSON to a HuggingFace dataset repo."""
    from huggingface_hub import create_repo, upload_file
    create_repo(repo_id, repo_type="dataset", token=token, exist_ok=True, private=True)
    payload = json.dumps(collect(), indent=2).encode("utf-8")
    upload_file(
        path_or_fileobj=io.BytesIO(payload),
        repo_id=repo_id,
        path_in_repo=f"{gpu_name}/gpu_info.json",
        token=token,
        repo_type="dataset",
    )
