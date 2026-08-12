"""Cloud-node entrypoint. Rents a GPU, checks it is admissible, then writes its artefacts into
the HuggingFace dataset repo.

================================================================================================
WHAT GETS UPLOADED
================================================================================================

  {gpu_name}/host_info_{uuid}.json      one per card  -- what hardware produced everything else
  {gpu_name}/device_probe_{uuid}.json   one per card  -- synthetic kernel sweep, hardware only
  {gpu_name}/probe_ext_v2_{uuid}.json   one per card  -- the workload shapes the probe grid
                                                         never reached (see probe_extension)
  {gpu_name}/{workload}.csv             one per MODEL -- the workload measurements, appended to
                                                         across cards and resumed by checkpoint

{uuid} is the physical GPU's NVML UUID (first 8 hex chars), stable across rentals of the same
card, so the per-card artefacts join on it, and the workload CSV carries it as instance_id so
its rows can be traced back to the card that produced them.

READ, never written by a node:

  reference/{gpu_name}_reference_v1.json  -- the frozen health reference, built offline by
                                             build_ground_truth.py from a probe snapshot. A card
                                             screens against this and nothing else; a model with
                                             no reference file runs no workloads.

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
import pathlib
import signal
import statistics
import sys
import time

import pandas as pd
import torch
from huggingface_hub import HfApi, hf_hub_download, upload_file

import salad
from profiler import device_probe, host_info, probe_extension
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
    # Detection is the one workload a CUDA graph cannot capture, so it reports GPU kernel time
    # via the profiler instead. Its kernels are small (4.8-22.7us), so the profiler's per-kernel
    # overhead is 20-94% of the reported time depending on config -- timing_method and
    # kernel_count are recorded per row so that bias can be corrected rather than absorbed.
    ("object_detection", object_detection),
    #("vlm_inference", vlm_inference),
    #("le_world_model", le_world_model),
    #("diffusion_inference", diffusion_inference),
]

INSTANCE_ID = host_info.get_gpu_uuid()

# PROBE_ONLY is read from the environment so it can be flipped in the SaladCloud console
# without a rebuild. Two passes:
#   PROBE_ONLY=1  many instances per GPU model, probe and exit. Builds the reference.
#   unset         one instance, probe, screen against that reference, then run workloads.
# A card that fails the screen exits so the platform reallocates -- cheap, because a failed
# attempt costs one probe rather than a workload sweep, and roughly 1 card in 5 was slow.
PROBE_ONLY = os.environ.get("PROBE_ONLY", "").strip().lower() in ("1", "true", "yes", "on")

# A card may be this much slower than the reference -- the per-row MEDIAN of its model's prior
# probes -- and still contribute. 1.0 is a typical card of that model, not the best one seen.
#
# Calibration, from fourteen RTX 4080 SUPERs and seven RTX 4070 Ti SUPERs:
#     4080 SUPER    healthy 0.945-1.017   degraded 1.103 (2550MHz vs a healthy 2730-2925)
#     4070 Ti SUPER healthy 0.951-1.024   none degraded
# 1.05 sits in the 8.6-point gap between the slowest healthy card and the degraded one, with
# margin on both sides. It is tighter than the 1.10 used while the reference was the per-row
# fastest, and yet admits strictly more healthy cards: against a fastest-reference the whole
# population was pushed up (worst healthy card 1.078) and drifted higher with every probe
# added, so 1.10 there was a moving threshold where 1.05 here is a fixed one.
PERF_TOLERANCE = 1.05
# Below this many prior probes the screen is skipped and the card proceeds, so that the very
# first probe -- good or bad -- cannot define "normal" on its own. Two rather than three
# because older or rarer cards may never accumulate three probes on SaladCloud, and a screen
# that never engages is worse than one built on a thin reference. The cost is that with two
# references the fastest is a noisier floor, so an unusually good card tightens the effective
# threshold for everything after it.
MIN_REFERENCE_PROBES = 2

# Every artefact is written at the END of a run that has already spent hours computing, so a
# transient network error or HF rate-limit would otherwise discard all of it. Retries are
# bounded and failures are reported rather than raised: losing one artefact beats aborting
# before the others are written.
RETRIES = 4
BACKOFF_S = 5


def install_signal_logging() -> None:
    """Make a platform stop distinguishable from a crash.

    A node that vanishes mid-model looks identical in the log whether SaladCloud preempted it,
    the container hit a memory limit, or the process died on a driver fault -- in every case the
    last line is just whatever we printed last. SIGTERM is what an orchestrator sends before
    SIGKILL, so catching it and saying so separates "the platform reclaimed us" from everything
    else. A hard SIGKILL or an OOM-killer kill still leaves nothing, which is itself informative:
    no SIGTERM line means it was not a graceful reclaim."""
    def handler(signum, _frame):
        name = signal.Signals(signum).name
        print(f"\n[signal] received {name} -- platform asked this container to stop; "
              f"work since the last checkpoint is lost", flush=True)
        sys.stdout.flush()
        raise SystemExit(128 + signum)
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, handler)
        except Exception:
            pass


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


def load_json(repo_id: str, path: str) -> dict | None:
    ok, local = with_retries(lambda: hf_hub_download(repo_id, path, repo_type="dataset"),
                             f"fetch {path}")
    if not ok:
        return None
    try:
        return json.loads(pathlib.Path(local).read_text())
    except Exception as e:
        print(f"  unreadable {path}: {type(e).__name__}: {e}")
        return None


def _variant(probe: dict) -> tuple:
    """The silicon identity of a probed card, for deciding whether two probes are comparable.

    gpu_name alone is not enough. The driver does distinguish the RTX 4080 from the 4080 SUPER
    by name (76 vs 80 SMs), but it reports the 10GB and 12GB RTX 3080 identically despite 68 vs
    70 SMs -- a ~3% timing gap that sits well inside PERF_TOLERANCE, so the screen would wave it
    through and chain two different chips into one dataframe."""
    dev = probe.get("device", {})
    # VRAM to whole GB: the figure is net of a driver reserve (every 16GB card here reports
    # 15.99), so exact float equality would be hostage to a driver-version difference, while
    # the variants this must separate differ by whole gigabytes.
    vram = dev.get("total_memory_gb")
    return (dev.get("sm_count"), dev.get("cuda_cores"), dev.get("compute_capability"),
            round(vram) if isinstance(vram, (int, float)) else None)


REFERENCE_VERSION = "v1"
REFERENCE_PATH = "reference/{gpu}_reference_" + REFERENCE_VERSION + ".json"


def load_reference(gpu_name: str, repo_id: str, files: list) -> dict | None:
    """The frozen health reference for this GPU model, or None if it has none.

    Checked against the repo listing first: most models have no reference and never will
    (pre-Ampere, or seen once), and letting those fall through to with_retries would spend four
    attempts and its backoff on a 404 that the listing already answered."""
    path = REFERENCE_PATH.format(gpu=gpu_name)
    if path not in files:
        return None
    return load_json(repo_id, path)


def screen_against_reference(sig: dict, ref: dict) -> bool:
    """True if this card is close enough to its model's frozen reference to contribute.

    The reference no longer rebuilds itself from whatever probes happen to be published. It is
    a fixed file, built once from 253 probes by build_ground_truth.py, holding the per-row median
    over that model's healthy cards across the heaviest 25% of each kernel family. Freezing it is
    what makes a verdict mean the same thing in six months as it does today, and what lets the
    probe grid be extended (see probe_extension) without retroactively changing who passed.

    Verified against the screen it replaces: over 240 cards, zero verdicts differ, and the median
    ratio moves by +0.0005. The threshold travels inside the file rather than being read from
    PERF_TOLERANCE here, so a reference and the tolerance it was validated at cannot drift apart.
    """
    rows = {tuple(r[:-1]): r[-1] for r in ref.get("rows", [])}
    tol = ref.get("perf_tolerance", PERF_TOLERANCE)

    # A card whose silicon differs from the reference's is not slow, it is a different chip --
    # the 10GB and 12GB 3080 report the same name at 68 vs 70 SMs. Screening it against the
    # wrong variant would be measuring the wrong thing in both directions.
    mine = _variant(sig)
    theirs = tuple(ref.get("variant", []))
    if theirs and mine != theirs:
        print(f"  variant mismatch: this card is {mine}, reference is {theirs} -- "
              f"cannot screen, so not contributing.")
        return False

    mine_rows = {}
    for r in sig.get("probes", []):
        if r.get("ms"):
            mine_rows[(r.get("probe"), r.get("dtype"), r.get("size"),
                       r.get("direction"), r.get("kind"), r.get("causal"))] = r["ms"]

    # Missing rows are an error rather than a smaller comparison: the reference pins the exact
    # row keys, so a probe that no longer emits them is a probe that changed underneath it, and
    # silently scoring on what survives would compare two different things.
    missing = [k for k in rows if k not in mine_rows]
    if missing:
        print(f"  probe is missing {len(missing)} of {len(rows)} reference rows "
              f"(e.g. {missing[0]}) -- probe and reference {ref.get('version')} disagree; "
              f"not contributing.")
        return False

    ratios = [mine_rows[k] / rows[k] for k in rows if rows[k] > 0]
    if not ratios:
        print("  no comparable rows against the reference -- not contributing.")
        return False
    ratio = statistics.median(ratios)
    verdict = "OK" if ratio <= tol else "REJECTED"
    print(f"  performance screen: {ratio:.3f}x the {ref.get('version')} reference over "
          f"{len(ratios)} rows, built from {ref.get('n_cards_healthy')} healthy cards "
          f"(tolerance {tol:.2f}x) -> {verdict}", flush=True)
    return ratio <= tol


def main() -> None:
    install_signal_logging()
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

    upload_bytes(json.dumps(info, indent=2).encode("utf-8"),
                 f"{gpu_name}/host_info_{INSTANCE_ID}.json", token, repo_id)

    probe_path = f"{gpu_name}/device_probe_{INSTANCE_ID}.json"

    # --- probe-only pass: build the pool a reference can later be frozen from -----------------
    if PROBE_ONLY:
        if probe_path in files:
            print(f"\n=== device_probe ===  {INSTANCE_ID} already probed; nothing to do.")
            return
        print(f"\n=== device_probe ({INSTANCE_ID}) ===")
        sig = device_probe.run_probe()
        print(f"  {len(sig['probes'])} probe rows")
        upload_bytes(json.dumps(sig, indent=2).encode("utf-8"), probe_path, token, repo_id)
        print("\nPROBE_ONLY set — probe published, not running workloads.")
        return

    # --- screen: a card unlike its peers should not contribute to their shared dataframe -----
    # A model with no frozen reference is one we never collected enough healthy probes for
    # (pre-Ampere, or a single sighting). It gets no workloads rather than unscreened ones: an
    # unscreened card is exactly what the reference exists to prevent, and running it would put
    # timings of unknown provenance into a dataset whose whole purpose is the distribution.
    ref = load_reference(gpu_name, repo_id, files)
    if ref is None:
        print(f"\nno {REFERENCE_VERSION} reference published for {gpu_name} — "
              "not eligible for workloads.")
        return

    # The allowlist is checked BEFORE probing, not after. INSTANCE_ID is the physical card's NVML
    # UUID, so a card that was one of the 228 the reference was built from is already known good
    # and re-probing it would spend ten minutes to compare it against a median it helped define.
    # This is what the PROBE_ONLY pass bought: by the time workloads run, most cards are known.
    if INSTANCE_ID in ref.get("healthy_uuids", []):
        print(f"\n=== screen ===  {INSTANCE_ID} is a known-healthy card in "
              f"{ref.get('version')}; skipping probe and screen.")
    else:
        sig = None
        if probe_path in files:
            print(f"\n=== device_probe ===  {INSTANCE_ID} already probed; reusing.")
            sig = load_json(repo_id, probe_path)
            if sig is None:
                # Published but unreadable: a fetch that failed every retry, or a truncated
                # upload. Re-probe rather than give up -- exiting here would forfeit the rental,
                # and with it the checkpoint progress waiting behind the screen.
                print("  stored probe could not be read — re-running it.")
        if sig is None:
            print(f"\n=== device_probe ({INSTANCE_ID}) ===")
            sig = device_probe.run_probe()
            print(f"  {len(sig['probes'])} probe rows")
            upload_bytes(json.dumps(sig, indent=2).encode("utf-8"), probe_path, token, repo_id)
        if not screen_against_reference(sig, ref):
            print("exiting so the platform reallocates to a different node.")
            return

    # --- v2 extension: the workload shapes the frozen probe grid never reached ---------------
    # Written to its own file and never consulted by the screen above. Extending device_probe
    # in place would strand the probes the v1 reference was built from; collecting the extension
    # separately keeps every published ratio meaning what it meant, and the files are merged
    # into a v2 reference offline once enough cards have run.
    ext_path = f"{gpu_name}/probe_ext_{probe_extension.EXT_VERSION}_{INSTANCE_ID}.json"
    if ext_path in files:
        print(f"\n=== probe_extension ===  {INSTANCE_ID} already extended; skipping.")
    else:
        print(f"\n=== probe_extension {probe_extension.EXT_VERSION} ({INSTANCE_ID}) ===",
              flush=True)
        try:
            ext = probe_extension.run_extension()
            n_oom = sum(1 for r in ext["probes"] if r.get("oom"))
            print(f"  {len(ext['probes'])} rows, {n_oom} OOM")
            upload_bytes(json.dumps(ext, indent=2).encode("utf-8"), ext_path, token, repo_id)
        except Exception as e:
            # The extension is additional data, not a gate. A card that cannot produce it has
            # already passed the screen, so its workload rows are still wanted.
            print(f"  extension failed: {type(e).__name__}: {e} — continuing to workloads.")

    # --- workloads: ONE dataframe per (gpu model, workload), assembled across nodes ----------
    # Not one file per card. A card only reaches this point after matching its model's
    # reference, so its rows are comparable with the ones already there; instance_id records
    # which card produced each row so they can still be separated afterwards.
    failed = []
    for name, module in WORKLOADS:
        target = f"{gpu_name}/{name}.csv"
        prior, done = None, set()
        if target in files:
            ok, path = with_retries(lambda t=target: hf_hub_download(repo_id, t, repo_type="dataset"),
                                    f"fetch {target}")
            if ok:
                try:
                    prior = pd.read_csv(path)
                    done = set(prior["model"].dropna().unique())
                except Exception as e:
                    print(f"  {target} unreadable, starting fresh: {type(e).__name__}: {e}")
                    prior = None

        todo = [m for m in module.MODELS if m not in done]
        if not todo:
            print(f"\n=== {name} ===  complete ({len(done)} models); skipping.")
            continue
        print(f"\n=== {name} ({INSTANCE_ID}) ===  {len(done)}/{len(module.MODELS)} models done, "
              f"{len(todo)} to go", flush=True)

        def merged(new: pd.DataFrame) -> pd.DataFrame:
            new = new.copy()
            new["instance_id"] = INSTANCE_ID
            return pd.concat([prior, new], ignore_index=True) if prior is not None else new

        def checkpoint(new: pd.DataFrame, _t=target) -> None:
            upload_bytes(merged(new).to_csv(index=False).encode("utf-8"), _t, token, repo_id)

        df = merged(module.run_all(skip_models=done or None, checkpoint_fn=checkpoint))
        if not upload_bytes(df.to_csv(index=False).encode("utf-8"), target, token, repo_id):
            failed.append(target)
        print(f"  {len(df)} rows total in {target}")

    if failed:
        print(f"\n{len(failed)} artefact(s) could not be uploaded: " + ", ".join(failed))

    print("\nall assigned workloads complete for this GPU model.")

if __name__ == "__main__":
    main()
    # Printed only on a normal return. Its absence in a node's log means the process was
    # killed rather than finishing, without needing to guess from where the output stops.
    print("\n[exit] main() returned normally", flush=True)
