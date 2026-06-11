import io
import json
import os

import pandas as pd
from huggingface_hub import HfApi, upload_file

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

# How many independent rentals to collect per (GPU, workload). Each Salad replica is
# guaranteed to land on a different physical node, so deploying with replicas=N gets
# us N independent measurements; analysis takes the median across these.
MAX_VERSIONS = 5

# Per-replica identifier = the physical GPU's NVML UUID (short prefix). Different Salad
# replicas land on different physical cards, so this distinguishes concurrent uploaders
# without races. A re-rental of the same physical card reuses the same ID — the
# "already-uploaded" check then turns into a free skip instead of duplicate work.
INSTANCE_ID = gpu_info.get_gpu_uuid()


def make_upload_fn(workload_name: str, gpu_name: str, token: str, repo_id: str):
    """Returns a function that uploads a DataFrame to {gpu_name}/{workload_name}.csv."""
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


def count_existing_versions(api: HfApi, repo_id: str, gpu_name: str, workload_name: str) -> tuple[int, bool]:
    """Returns (count, already_done_by_this_instance) for {gpu}/{workload}_full_*.csv files.

    Listing happens fresh on each call so concurrently-uploaded versions from sibling
    replicas are visible. The 'already done' flag prevents a Salad restart from re-running
    work this same physical instance already completed."""
    files = api.list_repo_files(repo_id, repo_type="dataset")
    prefix = f"{gpu_name}/{workload_name}_full_"
    versions = [f for f in files if f.startswith(prefix) and f.endswith(".csv")]
    mine = f"{gpu_name}/{workload_name}_full_{INSTANCE_ID}.csv"
    return len(versions), mine in versions


def main() -> None:
    gpu_name = gpu_info.get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"INSTANCE_ID: {INSTANCE_ID}")

    info = gpu_info.collect()
    ok, reason = check_full_power(gpu_name, info.get("power_limit_w"), threshold=0.95)
    print(f"power check: {reason}")
    if not ok:
        print("aborting — re-rent until a Salad host with stock power limit is allocated.")
        return

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    if not (token and repo_id):
        print("HF_TOKEN or HF_REPO_ID not set; running without uploads.")
        for name, module in WORKLOADS:
            print(f"\n=== {name} ===")
            module.run_all(upload_fn=None)
        return

    api = HfApi(token=token)

    # For each workload: skip if MAX_VERSIONS already collected globally, or if THIS
    # specific instance already uploaded its own version (e.g. after a Salad restart
    # to the same node).
    pending = []
    for name, module in WORKLOADS:
        count, mine_already_uploaded = count_existing_versions(api, repo_id, gpu_name, name)
        if mine_already_uploaded:
            print(f"=== {name} ===  this instance already uploaded; skipping.")
        elif count >= MAX_VERSIONS:
            print(f"=== {name} ===  already has {count} versions (>= {MAX_VERSIONS}); skipping.")
        else:
            print(f"=== {name} ===  {count}/{MAX_VERSIONS} versions exist; this instance will add one.")
            pending.append((name, module))

    if not pending:
        print(f"\nNothing to do for {gpu_name}/{INSTANCE_ID}. Exiting.")
        return

    # gpu_info captured on THIS host — tagged with our instance id alongside the workload CSVs.
    upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(info, indent=2).encode("utf-8")),
        repo_id=repo_id,
        path_in_repo=f"{gpu_name}/gpu_info_{INSTANCE_ID}.json",
        token=token,
        repo_type="dataset",
    )
    print(f"  uploaded gpu_info_{INSTANCE_ID}.json")

    for name, module in pending:
        print(f"\n=== {name} ({INSTANCE_ID}) ===")
        df = module.run_all(upload_fn=None)   # no progressive uploads — final only
        make_upload_fn(f"{name}_full_{INSTANCE_ID}", gpu_name, token, repo_id)(df)


if __name__ == "__main__":
    main()
