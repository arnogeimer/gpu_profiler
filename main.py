"""Cloud-node entrypoint. Rents a GPU, checks it is admissible, then writes three artefacts
per node into the HuggingFace dataset repo.

================================================================================================
WHAT GETS UPLOADED
================================================================================================

  {gpu_name}/host_info_{uuid}.json      one object   -- what hardware produced everything else
  device_probe/{uuid}.json              one object   -- synthetic kernel sweep, hardware only
  {gpu_name}/{workload}_full_{uuid}.csv one per row  -- the workload measurements

{uuid} is the physical GPU's NVML UUID (first 8 hex chars), stable across rentals of the same
card, so all three join on it.

================================================================================================
THE WORKLOAD CSV -- every column and where it comes from
================================================================================================

Three functions contribute to each row, in this order:

1. The workload's own Hyperparams dataclass. These differ per workload, which is why the four
   CSVs are not union-compatible:

     image_classification   model, img_size,         batch_size,            precision
     object_detection       model, img_size,         batch_size,            precision
     audio_classification   model, seconds,          batch_size,            precision
     llm_finetune           model, sequence_length,  batch_size, lora_rank, precision

2. profiler.cuda_monitor.build_row:

     phase           "train" for every active workload, or "setup" when the model failed to
                     build. inference.py's parked phases also emit "infer"/"prefill"/"decode".
     error           "" on success, else a message. A row can be present and still be useless.
     avg_time_ms     THE MEASUREMENT. GPU kernel time for one step. None on OOM or error.
     timing_method   which instrument produced avg_time_ms -- "cuda_graph" (profiler.time_fn,
                     exact) or "kernel_sum" (profiler.kernel_time_fn, reads high by ~2.4-6.9us
                     per kernel). Only object_detection uses the latter; it cannot be captured.
     kernel_count    kernels per step, recorded only for "kernel_sum" rows so that bias can be
                     corrected or fitted. None everywhere else.

3. profiler.cuda_monitor.CUDAMonitor.stop -- a background thread polls torch + NVML at 10ms
   through the timed region:

     oom                       True if the config ran out of VRAM
     sample_count, duration_s  how many polls backed the aggregates below, and over how long
     nvml                      False if NVML was unavailable, so max_memory_used_mb_nvml is None
     max_memory_used_pct/_mb   peak VRAM, torch's view
     max_memory_used_mb_nvml   peak VRAM, whole-device view (catches other tenants)

   Power, energy and GPU utilisation used to sit here and were removed. NVML refreshes those
   counters on the driver's own cadence rather than per query, so a ~0.85s config window
   averages a handful of arbitrarily-aligned updates: on one config whose timing was stable to
   0.2%, power read 42-145W and utilisation 0-68% purely with how long the card had been idle
   beforehand. Measuring power properly needs a dedicated multi-second replay window, which
   would roughly double the sweep -- see the note in cuda_monitor._SUMMARY_KEYS.

   NOTE: rows written by run_all's except-branch ("phase": "config_failed") carry only the
   Hyperparams plus phase and error -- an exception escaped run() before build_row was reached,
   so every column above is absent rather than None.

================================================================================================
host_info.json -- one object describing the node, joined to the CSVs on {uuid}
================================================================================================

  CPU     cpu_name, cpu_count, cpu_physical_cores, cpu_max_mhz
          Present because launch-bound rows (low avg_gpu_util_pct) have their runtime set by
          the host, not the card, and Salad allocates heterogeneous hosts.
  GPU     name, compute_capability, sm_count, cores_per_sm, cuda_cores, vram_total_mb,
          l2_cache_kb, uuid, power_limit_w, driver_version,
          max_graphics_clock_mhz, max_sm_clock_mhz, max_memory_clock_mhz
  Derived measured_tflops_fp32, measured_tflops_fp16  (one large square matmul each)

  host_info.STOCK_TDP_W + check_full_power gate the whole run: a host whose power_limit_w is
  below 95% of the card's stock TDP is rejected before any measurement is taken.

================================================================================================
device_probe/{uuid}.json -- {"device": {...}, "probes": [...]}
================================================================================================

  device   the same hardware facts as host_info plus torch/cuda versions and a timestamp
  probes   one row per synthetic kernel measured: {probe, dtype, size, kind|causal, direction,
           ms}. gemm, bmm, conv, attn, elementwise, pool and rnn, each swept over fp32/fp16/bf16.
           This is the hardware signature the workload timings are meant to be predicted from.
"""

import io
import json
import os
import time

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, upload_file

from profiler import device_probe, host_info
from profiler.host_info import check_full_power
from workloads.computer_vision import image_classification, object_detection
from workloads.audio import audio_classification
from workloads.generative import diffusion_inference
from workloads.multimodal import vlm_inference
from workloads.nlp import llm_finetune
from workloads.robotics import le_world_model


# Training only for now. vlm_inference and diffusion_inference record no training phase at all,
# so they sit out entirely; the inference halves cut from the three active ones are parked in
# inference.py. Uncommenting a line here brings that workload back unchanged.
WORKLOADS = [
    ("image_classification", image_classification),
    ("audio_classification", audio_classification),
    ("llm_finetune", llm_finetune),
    # object_detection is parked, not broken. It is the only workload a CUDA graph cannot
    # capture, so it reports GPU kernel time via the profiler instead -- and detection kernels
    # are small enough (4.8-22.7us) that the profiler's per-kernel overhead is 20-94% of the
    # reported time, varying with both config and card speed. That is a confound in exactly the
    # dimension being predicted. Re-enable once the overhead is calibrated on detection-shaped
    # kernels; the backbone alone does capture, so both instruments can be compared directly.
    #("object_detection", object_detection),
    #("vlm_inference", vlm_inference),
    #("le_world_model", le_world_model),
    #("diffusion_inference", diffusion_inference),
]

# How many independent rentals to collect per (GPU, workload).
MAX_VERSIONS = 5
INSTANCE_ID = host_info.get_gpu_uuid()

# Every artefact is written at the END of a run that has already spent hours computing, so a
# transient network error or HF rate-limit would otherwise discard all of it. Retries are
# bounded and failures are reported rather than raised: losing one artefact beats aborting
# before the others are written.
RETRIES = 4
BACKOFF_S = 5


def configure_runtime() -> None:
    """Global torch settings applied once before any workload runs."""
    torch.backends.cudnn.benchmark = False
    torch.cuda.set_per_process_memory_fraction(1.0, 0)


def with_retries(fn, what: str) -> tuple[bool, object]:
    """Run fn with bounded exponential backoff. Returns (ok, result); never raises."""
    for attempt in range(1, RETRIES + 1):
        try:
            return True, fn()
        except Exception as e:
            if attempt == RETRIES:
                print(f"  FAILED after {attempt} attempts — {what}: {type(e).__name__}: {e}")
                return False, None
            wait = BACKOFF_S * 2 ** (attempt - 1)
            print(f"  {what}: attempt {attempt}/{RETRIES} failed ({type(e).__name__}); retry in {wait}s")
            time.sleep(wait)
    return False, None


def upload_bytes(payload: bytes, path_in_repo: str, token: str, repo_id: str) -> bool:
    """Upload one artefact, retried. Returns True on success."""
    ok, _ = with_retries(
        lambda: upload_file(path_or_fileobj=io.BytesIO(payload), repo_id=repo_id,
                            path_in_repo=path_in_repo, token=token, repo_type="dataset"),
        f"upload {path_in_repo}",
    )
    if ok:
        print(f"  uploaded {path_in_repo} ({len(payload) / 1024:.0f} KB)")
    return ok


def list_repo_files(api: HfApi, repo_id: str) -> tuple[bool, list]:
    """Repo listing, retried. Returns (ok, files); an unreadable repo must not be read as empty,
    which would make every workload look uncollected and duplicate the whole run."""
    ok, files = with_retries(lambda: api.list_repo_files(repo_id, repo_type="dataset"),
                             "list repo files")
    return ok, list(files) if ok else []


def count_existing_versions(files: list, gpu_name: str, workload_name: str) -> tuple[int, bool]:
    """Returns (count, already_done_by_this_instance) for {gpu}/{workload}_full_*.csv files.

    The 'already done' flag prevents a Salad restart from re-running work this same physical
    instance already completed."""
    prefix = f"{gpu_name}/{workload_name}_full_"
    versions = [f for f in files if f.startswith(prefix) and f.endswith(".csv")]
    mine = f"{gpu_name}/{workload_name}_full_{INSTANCE_ID}.csv"
    return len(versions), mine in versions


def main() -> None:
    configure_runtime()
    gpu_name = host_info.get_gpu_name()
    print(f"GPU: {gpu_name}")
    print(f"INSTANCE_ID: {INSTANCE_ID}")

    info = host_info.collect()
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
            module.run_all()
        return

    api = HfApi(token=token)

    # One listing for the whole run. A failed listing aborts rather than proceeding, because
    # treating it as empty would re-run work that already exists.
    listed, files = list_repo_files(api, repo_id)
    if not listed:
        print("aborting — cannot read the dataset repo, so pending work cannot be determined.")
        return

    # For each workload: skip if MAX_VERSIONS already collected globally, or if THIS
    # specific instance already uploaded its own version (e.g. after a Salad restart
    # to the same node).
    pending = []
    for name, module in WORKLOADS:
        count, mine_already_uploaded = count_existing_versions(files, gpu_name, name)
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

    # host_info captured on THIS host — tagged with our instance id alongside the workload CSVs.
    upload_bytes(json.dumps(info, indent=2).encode("utf-8"),
                 f"{gpu_name}/host_info_{INSTANCE_ID}.json", token, repo_id)

    probe_path = f"{gpu_name}/device_probe_{INSTANCE_ID}.json"
    if probe_path in files:
        print(f"\n=== device_probe ===  {INSTANCE_ID} already present; skipping.")
    else:
        print(f"\n=== device_probe ({INSTANCE_ID}) ===")
        sig = device_probe.run_probe()
        print(f"  {len(sig['probes'])} probe rows")
        upload_bytes(json.dumps(sig, indent=2).encode("utf-8"), probe_path, token, repo_id)

    # Each workload checkpoints to _partial_ as it goes and writes _full_ only on completion.
    # Salad containers reset at arbitrary points, so without checkpoints a node that runs for
    # hours and is preempted near the end contributes nothing. Both names carry this GPU's own
    # UUID, so a resume can only ever pick up this card's work -- never another node's.
    failed = []
    for name, module in pending:
        print(f"\n=== {name} ({INSTANCE_ID}) ===")
        partial = f"{gpu_name}/{name}_partial_{INSTANCE_ID}.csv"

        prior, done = None, set()
        if partial in files:
            ok, path = with_retries(
                lambda: hf_hub_download(repo_id, partial, repo_type="dataset"),
                f"fetch {partial}")
            if ok:
                try:
                    prior = pd.read_csv(path)
                    done = set(prior["model"].dropna().unique())
                    print(f"  resuming this GPU's checkpoint: {len(prior)} rows, "
                          f"{len(done)} models already done", flush=True)
                except Exception as e:
                    print(f"  checkpoint unreadable, starting fresh: {type(e).__name__}: {e}")
                    prior = None

        def merged(new: pd.DataFrame) -> pd.DataFrame:
            return pd.concat([prior, new], ignore_index=True) if prior is not None else new

        def checkpoint(new: pd.DataFrame, _p=partial) -> None:
            upload_bytes(merged(new).to_csv(index=False).encode("utf-8"), _p, token, repo_id)

        df = merged(module.run_all(skip_models=done or None, checkpoint_fn=checkpoint))
        target = f"{gpu_name}/{name}_full_{INSTANCE_ID}.csv"
        if upload_bytes(df.to_csv(index=False).encode("utf-8"), target, token, repo_id):
            # the partial has served its purpose; leaving it would double the repo's rows
            with_retries(lambda _p=partial: api.delete_file(_p, repo_id, repo_type="dataset"),
                         f"delete {partial}")
        else:
            failed.append(target)
        print(f"  {len(df)} rows")

    if failed:
        print(f"\n{len(failed)} artefact(s) could not be uploaded: " + ", ".join(failed))


if __name__ == "__main__":
    main()
