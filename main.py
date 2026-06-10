import io
import os

# MUST be set before torch is imported (transitively imported by the workload modules
# below). Disables PyTorch's caching allocator so out-of-memory fails immediately with
# a clean OutOfMemoryError instead of silently thrashing via retry/empty_cache loops.
# Costs ~10-20% extra allocation overhead in the normal regime, fine for profiling.
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

import pandas as pd
from huggingface_hub import HfApi

from profiler import gpu_info
from profiler.gpu_specs import check_full_power
from workloads.computer_vision import image_classification
from workloads.audio import audio_classification
from workloads.generative import diffusion_inference
from workloads.multimodal import vlm_inference
from workloads.nlp import llm_finetune
from workloads.robotics import le_world_model


WORKLOADS = [
    ("image_classification", image_classification),
    #("audio_classification", audio_classification),
    #("llm_finetune", llm_finetune),
    #("vlm_inference", vlm_inference),
    #("le_world_model", le_world_model),
    #("diffusion_inference", diffusion_inference),
]


def make_upload_fn(workload_name: str, gpu_name: str, token: str, repo_id: str):
    """Returns a function that uploads a DataFrame to {gpu_name}/{workload_name}.csv."""
    from huggingface_hub import upload_file
    target = f"{gpu_name}/{workload_name}.csv"
    def upload(df: pd.DataFrame) -> None:
        payload = df.to_csv(index=False).encode("utf-8")
        upload_file(
            path_or_fileobj=io.BytesIO(payload),
            repo_id=repo_id,
            path_in_repo=target,
            token=token,
            repo_type="dataset",
        )
        print(f"  uploaded {len(df)} rows -> {repo_id}:{target}")
    return upload


def main() -> None:
    gpu_name = gpu_info.get_gpu_name()
    print(f"GPU: {gpu_name}")

    # Refuse to run on a host that has clamped the GPU below its stock TDP — those
    # measurements would under-represent the silicon and pollute the dataset.
    info = gpu_info.collect()
    ok, reason = check_full_power(gpu_name, info.get("power_limit_w"), threshold=0.95)
    print(f"power check: {reason}")
    if not ok:
        print("aborting — re-rent until a Salad host with stock power limit is allocated.")
        return

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    if not (token and repo_id):
        print("HF_TOKEN or HF_REPO_ID not set; running without uploads or resume support.")
        for name, module in WORKLOADS:
            print(f"\n=== {name} ===")
            module.run_all(upload_fn=None)
        return

    api = HfApi(token=token)

    # Decide what's left to do BEFORE touching gpu_info.json. If every workload is
    # already complete for this GPU, the existing gpu_info.json was captured on a
    # presumably-good host — don't overwrite it with this rerun's measurements
    # (which may come from a power-capped or otherwise different physical host).
    pending = []
    for name, module in WORKLOADS:
        full_path = f"{gpu_name}/{name}_full.csv"
        if api.file_exists(repo_id, full_path, repo_type="dataset"):
            print(f"=== {name} ===  already complete for {gpu_name}, skipping.")
        else:
            pending.append((name, module))
    if not api.file_exists(repo_id, full_path, repo_type="dataset"):
        gpu_info.upload_to_hf(token, repo_id, gpu_name)

    if not pending:
        print(f"\nAll workloads already complete for {gpu_name}. Not overwriting gpu_info.json.")
        return

    for name, module in pending:
        print(f"\n=== {name} ===")
        progressive_fn = make_upload_fn(name, gpu_name, token, repo_id)
        df = module.run_all(upload_fn=progressive_fn)
        # Final marker upload: the presence of *_full.csv signals completion for resume logic.
        make_upload_fn(f"{name}_full", gpu_name, token, repo_id)(df)


if __name__ == "__main__":
    main()
