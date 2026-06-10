import io
import os

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

# How many independent rentals to collect per (GPU, workload). Salad reassigns
# physical hosts on restart, so each version is from a different node — taking
# median across versions later cancels per-host variance.
MAX_VERSIONS = 5


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


def next_version(api: HfApi, repo_id: str, gpu_name: str, workload_name: str) -> int | None:
    """Returns the lowest version number in [1..MAX_VERSIONS] for which
    {gpu}/{workload}_full_v{N}.csv does NOT yet exist, or None if all versions are done."""
    for v in range(1, MAX_VERSIONS + 1):
        full_path = f"{gpu_name}/{workload_name}_full_v{v}.csv"
        if not api.file_exists(repo_id, full_path, repo_type="dataset"):
            return v
    return None


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

    # For each enabled workload, find the next version slot to fill (v1..v3). Workloads
    # that already have all MAX_VERSIONS done get skipped — Salad's auto-restart will
    # naturally land each container on a different physical host, so each version comes
    # from a different rental.
    pending = []
    for name, module in WORKLOADS:
        v = next_version(api, repo_id, gpu_name, name)
        if v is None:
            print(f"=== {name} ===  all {MAX_VERSIONS} versions complete for {gpu_name}, skipping.")
        else:
            print(f"=== {name} ===  next version: v{v}")
            pending.append((name, module, v))

    if not pending:
        print(f"\nAll workloads complete (v1..v{MAX_VERSIONS}) for {gpu_name}. Exiting.")
        return

    # gpu_info captured on THIS host — tagged with the lowest version we're about to run
    # so each rental session has its own host snapshot beside its workload CSVs.
    lowest_v = min(v for _, _, v in pending)
    gpu_info_versioned_name = f"gpu_info_v{lowest_v}"
    from huggingface_hub import upload_file
    import json
    upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(info, indent=2).encode("utf-8")),
        repo_id=repo_id,
        path_in_repo=f"{gpu_name}/{gpu_info_versioned_name}.json",
        token=token,
        repo_type="dataset",
    )
    print(f"  uploaded {gpu_info_versioned_name}.json")

    for name, module, v in pending:
        print(f"\n=== {name} v{v} ===")
        # No progressive uploads — saves on HF commits. Only the final marker upload runs.
        df = module.run_all(upload_fn=None)
        # Final upload — presence of {workload}_full_v{N}.csv = this version complete.
        make_upload_fn(f"{name}_full_v{v}", gpu_name, token, repo_id)(df)


if __name__ == "__main__":
    main()
