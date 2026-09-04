"""Probe-only node. Sanity-check the hardware, sweep every kernel, publish one frame.

    python run.py

Three steps, in order, and the first one gates the other two:

  1. hardware sanity check   host_info.check_full_power against the card's stock TDP. A host
                             running a sub-spec power limit measures the host, not the silicon,
                             so it is rejected before a single kernel is timed.
  2. probes                  probes.run_probes() sweeps every family in probes.PROBES in one
                             pass -- the core shapes, the large ones above them, the per-layer
                             costs between them, and the loss and optimizer at the end.
  3. publish                 the frame as CSV plus a metadata JSON, to the HuggingFace dataset
                             repo named by HF_REPO_ID.

Self-contained on purpose: nothing here imports from the parent repo, and the workload suite,
the dataset-cleanup stages and the offline reference tooling are all absent. What that buys is
the image -- no torchvision, timm, transformers, peft or diffusers, none of which a probe needs.

    HF_TOKEN     write token for the dataset repo. Unset -> the frame is written locally only.
    HF_REPO_ID   e.g. arge23/gpu-profiling-results

Published per card:

    {gpu}/probes_{uuid}.csv        one row per (probe, dtype, size, direction, kind, causal)
    {gpu}/probe_meta_{uuid}.json   host_info plus the sweep's `device` block -- torch/driver
                                   versions and the clocks sampled DURING the sweep, which is
                                   where throttling shows up and which the flat frame cannot
                                   carry.

`uuid` is the physical card's NVML UUID (first 8 hex chars), stable across rentals, so repeated
rentals of the same silicon overwrite rather than accumulate.

NOTE ON THE ARTEFACT FORMAT. The fleet's older split JSONs (device_probe_{uuid}.json and
probe_ext_v2_{uuid}.json) are NOT written here -- this publishes one frame instead. The
repo-side tooling that builds references from published probes (build_ground_truth.py,
export_probe_traces.py) reads those JSONs, so it does not ingest a node running this folder until
it learns to read probes_{uuid}.csv. Everything it needs is present, in a flatter shape.
"""
import os

# Set before torch is imported: the allocator reads it once, at first initialisation, and a
# later assignment is silently ignored. Segments grow in place instead of being pinned at their
# original size, which is the fragmentation that otherwise strands a card mid-sweep.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import io
import json
import time
from pathlib import Path

import pandas as pd
import torch
from huggingface_hub import upload_file

import host_info
import probes

# Cap torch's allocator below the physical limit so a shape that does not fit raises
# OutOfMemoryError instead of thrashing against a full device and reporting allocator churn as a
# measurement. Same value the workload nodes run at, so probe rows stay comparable to theirs.
MEMORY_FRACTION = 0.995

# A host below this fraction of the card's stock TDP is rejected outright -- see host_info.
POWER_THRESHOLD = 0.95

# Publishing happens after the sweep has already been paid for, so a transient network error
# must not discard it.
RETRIES = 4
BACKOFF_S = 5

COLS = ["gpu", "uuid", "probe", "dtype", "size", "direction", "kind", "causal",
        "ms", "oom", "error"]


def configure() -> None:
    """Runtime settings applied once, before anything is timed."""
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION, 0)


def sanity_check() -> tuple[bool, str, dict]:
    """(admissible, reason, host_info). The gate: a sub-spec host is not worth probing."""
    info = host_info.collect()
    ok, reason = host_info.check_full_power(host_info.get_gpu_name(),
                                            info.get("power_limit_w"),
                                            threshold=POWER_THRESHOLD)
    return ok, reason, info


def _rows(sig: dict, gpu: str, uuid: str) -> list[dict]:
    """The sweep's rows, flattened. Every family emits the same row shape, so one reader serves
    all of them -- families that have no kind or causal simply leave those unset, and the family
    name in `probe` is what keeps their row keys from colliding."""
    return [{"gpu": gpu, "uuid": uuid,
             "probe": r.get("probe"), "dtype": r.get("dtype"), "size": r.get("size"),
             "direction": r.get("direction"), "kind": r.get("kind"), "causal": r.get("causal"),
             "ms": r.get("ms"), "oom": bool(r.get("oom", False)), "error": r.get("error", "")}
            for r in sig.get("probes", [])]


def run() -> tuple[pd.DataFrame, dict]:
    """Sweep every family. Returns (frame, device block).

    Per-family containment lives in probes.run_probes, so a family that dies does not cost the
    families after it. This wrapper catches what is left -- the clock sampler or device_meta on a
    context too broken to query -- because a node that cannot report is worse than one that
    reports nothing."""
    configure()
    gpu, uuid = host_info.get_gpu_name(), host_info.get_gpu_uuid()

    print(f"\n=== probes ({uuid}) ===", flush=True)
    try:
        sig = probes.run_probes()
    except Exception as e:
        print(f"  sweep failed outright: {type(e).__name__}: {e}")
        return pd.DataFrame(columns=COLS), {}

    df = pd.DataFrame(_rows(sig, gpu, uuid), columns=COLS)
    for probe, t in df.groupby("probe", sort=False):
        print(f"  {probe:12} {len(t):5} rows, {int(t.oom.sum()):4} OOM, "
              f"{int((t.error != '').sum()):4} error")
    return df, sig.get("device", {})


def publish(payload: bytes, path: str, token: str, repo_id: str) -> bool:
    """Upload one artefact with bounded backoff. Reports failure rather than raising."""
    for attempt in range(1, RETRIES + 1):
        try:
            upload_file(path_or_fileobj=io.BytesIO(payload), repo_id=repo_id,
                        path_in_repo=path, token=token, repo_type="dataset")
            print(f"  uploaded {path} ({len(payload) / 1024:.0f} KB)")
            return True
        except Exception as e:
            if attempt == RETRIES:
                print(f"  FAILED after {attempt} attempts — {path}: {type(e).__name__}: {e}")
                return False
            wait = BACKOFF_S * 2 ** (attempt - 1)
            print(f"  {path}: attempt {attempt}/{RETRIES} failed ({type(e).__name__}); "
                  f"retry in {wait}s")
            time.sleep(wait)
    return False


def main() -> None:
    gpu, uuid = host_info.get_gpu_name(), host_info.get_gpu_uuid()
    print(f"GPU: {gpu}\nUUID: {uuid}")

    ok, reason, info = sanity_check()
    print(f"power check: {reason}")
    if not ok:
        print("aborting — re-rent until a host with a stock power limit is allocated.")
        return

    df, device = run()
    if df.empty:
        print("\nno probe rows produced; nothing to publish.")
        return

    print(f"\n{len(df)} rows total over {df.probe.nunique()} families   "
          f"{int(df.oom.sum())} OOM, {int((df.error != '').sum())} error, "
          f"{int(df.ms.notna().sum())} timed")

    local = Path(f"probes_{uuid}.csv")
    df.to_csv(local, index=False)
    print(f"wrote {local}")

    token, repo_id = os.environ.get("HF_TOKEN"), os.environ.get("HF_REPO_ID")
    if not (token and repo_id):
        print("HF_TOKEN or HF_REPO_ID not set; not uploading.")
        return

    print()
    publish(json.dumps({"host": info, "device": device}, indent=2, default=str).encode("utf-8"),
            f"{gpu}/probe_meta_{uuid}.json", token, repo_id)
    publish(df.to_csv(index=False).encode("utf-8"),
            f"{gpu}/probes_{uuid}.csv", token, repo_id)


if __name__ == "__main__":
    main()
    # Printed only on a normal return, so its absence in a node's log means the process was
    # killed rather than finishing.
    print("\n[exit] main() returned normally", flush=True)
