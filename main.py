import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from profiler import gpu_info

WORKLOADS = [
    "workloads.computer_vision.classification_train",
]


def upload_results(gpu_name: str, token: str, repo_id: str) -> None:
    from huggingface_hub import upload_file
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    csvs = list(Path("workloads").rglob("*_metrics.csv"))
    if not csvs:
        print("\nNo CSV results found to upload.")
        return
    print(f"\nUploading {len(csvs)} CSV(s) to {repo_id}:")
    for csv in csvs:
        workload = csv.parent.name
        target = f"{gpu_name}/{workload}_{timestamp}.csv"
        upload_file(
            path_or_fileobj=str(csv),
            repo_id=repo_id,
            path_in_repo=target,
            token=token,
            repo_type="dataset",
        )
        print(f"  {csv} -> {repo_id}:{target}")


def main():
    gpu_name = gpu_info.get_gpu_name()
    print(f"GPU: {gpu_name}")

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ.get("HF_REPO_ID")
    upload_enabled = bool(token and repo_id)
    if upload_enabled:
        gpu_info.upload_to_hf(token, repo_id, gpu_name)
    else:
        print("HF_TOKEN or HF_REPO_ID not set; skipping uploads.")

    for module in WORKLOADS:
        print(f"\n=== {module} ===")
        subprocess.run([sys.executable, "-m", module], check=False)

    if upload_enabled:
        upload_results(gpu_name, token, repo_id)


if __name__ == "__main__":
    main()
