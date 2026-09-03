"""Export one paper's slice of the dataset: its workload, plus the hardware characterisation.

    python export_paper_data.py                 # both papers
    python export_paper_data.py --paper vision  # one of them

Each paper gets four files, self-contained and independent of the hub:

    {paper}_paper/profiling_results/
        {workload}_clean.csv        the measurements: one row per (config, GPU model)
        gpu_info.csv                one row per GPU model, joins the above on `gpu`
        reference/*_full.json       v1 + v2 probe reference per GPU model
        MANIFEST.txt                counts, provenance, and what was left out

Three tables, one join key each. `{workload}_clean.csv` carries only config axes, the target,
and the OOM flag; everything describing the hardware lives in `gpu_info.csv` (static: cores,
cache, clocks, VRAM, measured TFLOPS) or in `reference/` (measured: per-kernel probe times).
That split is the point -- a runtime model needs a config, a card, and a way to describe the
card, and it can choose whether to describe it by spec sheet or by probe.

The probe reference goes to both papers rather than one. It is the hardware signature, the input
side of any runtime model built on this data, so a paper shipping the timings without it ships
half the experiment.

Everything here is a reduction, and nothing is a record: the per-card CSVs and probe JSONs are
not copied. The fleet's raw output stays on the hub, this script rebuilds these four files from
it in one command, and dataset_cleanup reproduces the fully filtered frame on demand.

object_detection is deliberately absent from the vision export. Every one of its 5 184 collected
rows is a `config_failed` with no timing -- a poisoned CUDA RNG generator killed the workload
fleet-wide (see profiler._clear_capture_state) -- so there is nothing to export yet. It returns
here once the re-run lands.
"""
from pathlib import Path
import argparse
import json

import statistics
import sys

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_cleanup import clean, relabel_escaped_ooms, stage1_vram_class
from main import MIN_REFERENCE_PROBES
from plotting.data import REPO, load
from profiler.device_probe import HEAVY_FRACTION, _heaviest

# Which workloads belong to which paper. A list, because vision regains object_detection after
# the re-run and audio may gain a second workload later.
PAPERS = {
    "vision": ["image_classification"],
    "audio": ["audio_classification"],
}

# Dropped from the exported frame. Each is either constant, redundant, or actively misleading as
# a feature; none is a property of the config whose latency is being predicted. The raw per-GPU
# CSVs keep all of them, so nothing here is destroyed.
#
#   timing_method            one value ("cuda_graph") in both workloads -- only object_detection
#                            uses the other instrument, and it is not exported yet
#   kernel_count             zero non-null rows; it is only recorded by the kernel_sum instrument
#   error                    zero non-null rows, and every untimed row already carries oom=True,
#                            so no failure is left unexplained by dropping it
#   n_attempted              always 1: main.py resumes from the checkpoint, so each config is
#                            measured once per GPU model, by whichever card reached it first
#   n_timed                  0 or 1, exactly redundant with avg_time_ms being present
#   sample_count             how many times the memory monitor polled -- a property of the
#                            instrument, not of the workload
#   max_memory_used_pct      whole-device readings that include whatever the previous config left
#   max_memory_used_mb       in the allocator's cache. Measured directly: the same audio config
#   max_memory_used_mb_nvml  reads 18.1% fresh and 47.6% after eight heavy models. They overstate
#                            a config's own footprint by up to ~30 points, so as a memory-demand
#                            feature they would teach a model the sweep order.
#   phase                    one value ("train"); it only carries information if an inference
#                            phase is ever measured alongside training again
#   duration_s               wall time of the monitored window, which includes building the model
#                            and moving it to the device -- ~1.38 s where the step itself is
#                            42 ms. It describes what the sweep cost to run, not what the step
#                            costs, so as a feature it would mostly encode setup time.
DROP_COLS = ["timing_method", "kernel_count", "error", "n_attempted", "n_timed",
             "sample_count", "max_memory_used_pct", "max_memory_used_mb",
             "max_memory_used_mb_nvml", "phase", "duration_s"]

# gpu_info.csv, one row per GPU MODEL, joining the workload frame on `gpu`.
#
# Split by what the field actually is. These seven are silicon identity -- checked across 173
# cards, every one is identical for every card of a model, so they are copied straight through:
GPU_CONSTANT = ["compute_capability", "sm_count", "cores_per_sm", "cuda_cores",
                "l2_cache_kb", "vram_total_mb"]

# These vary between cards of the same model and are reduced to a median. Clocks and power move
# 3-10% at most (worst: RTX 4080 power, 320-352 W) -- board-partner variation, and a median is a
# fair summary of it. The TFLOPS pair is different: it is a measurement, and on the RTX 3080 it
# spans 2.57x (24.6 to 63.2 fp16) because one rental was degraded. The median lands at 62.9,
# beside the max rather than between the two, which is exactly why it is the median and not the
# mean -- the same reasoning the probe reference uses.
GPU_MEDIAN = ["power_limit_w", "max_sm_clock_mhz", "max_graphics_clock_mhz",
              "max_memory_clock_mhz", "measured_tflops_fp32", "measured_tflops_fp16"]

# cpu_name, cpu_count, cpu_physical_cores and uuid are deliberately absent. The CPU fields
# describe the rented host, not the GPU, and change from rental to rental (20 of 22 models saw
# more than one CPU) -- so they cannot be an attribute of a GPU model, and a median over them
# would be meaningless. uuid identifies a card, and the export is per model.


def _probe_rows(sig: dict) -> dict:
    """Row key -> ms, exactly as device_probe.compare_to_reference extracts it.

    The v1 and v2 probes emit the same row shape, so one extractor serves both; v2 simply leaves
    kind/causal unset and carries its own family names (gemm_ext, conv1d, conv_ext), which cannot
    collide with v1's. Keep this in step with compare_to_reference -- a reference built on keys
    the screen does not produce compares nothing."""
    out = {}
    for r in sig.get("probes", []):
        key = (r.get("probe"), r.get("dtype"), r.get("size"),
               r.get("direction"), r.get("kind"), r.get("causal"))
        if r.get("ms"):
            out[key] = r["ms"]
    return out


def _reduce(cards: dict, uuids: list, heavy: bool = False) -> dict | None:
    """Per-row median over the given cards, optionally cut to the heaviest fraction per family.

    `heavy` is off for the v2 extension, because v2 has no launch-bound tail to cut. The cut
    exists to drop rows so small that every card returns the same number -- v1 sweeps down to
    1 us, and keeping those narrowed the fleet span from 1.98x to 1.66x. v2 bottoms out at 17 us
    with a 5th percentile of 146 us, so there is nothing there to remove.

    Note it is NOT that v2 is the heavier arm. Against v1's heavy subset -- the right comparison,
    and not the one first made here -- v2 is marginally lighter: median 2.58 ms against 2.76,
    p95 26.8 against 38.1. What v2 adds is coverage in shape space, not weight: larger GEMM M
    dimensions, wider conv2d, and conv1d, which v1 does not sample at all."""
    maps = [_probe_rows(cards[u]) for u in uuids if u in cards]
    if len(maps) < MIN_REFERENCE_PROBES:
        return None
    common = set.intersection(*[set(m) for m in maps])
    if not common:
        return None
    ref = {k: statistics.median([m[k] for m in maps]) for k in common}
    keys = sorted(_heaviest(ref, HEAVY_FRACTION) if heavy else ref)
    out = {"n_cards": len(maps), "n_rows_common": len(common), "n_rows_kept": len(keys),
           "rows": [list(k) + [ref[k]] for k in keys]}
    if heavy:
        out["heavy_fraction"] = HEAVY_FRACTION
    return out


def gpu_info(snapshot: Path) -> pd.DataFrame:
    """One row per GPU model, from every host_info published for it. Joins the frame on `gpu`."""
    out = []
    for gpu_dir in sorted(p for p in snapshot.iterdir() if p.is_dir() and p.name != "reference"):
        cards = [json.loads(p.read_text()) for p in sorted(gpu_dir.glob("host_info_*.json"))]
        if not cards:
            continue
        row = {"gpu": gpu_dir.name, "name": cards[0].get("name"), "n_cards": len(cards)}
        for k in GPU_CONSTANT:
            vals = {json.dumps(c.get(k)) for c in cards if c.get(k) is not None}
            row[k] = cards[0].get(k)
            if len(vals) > 1:                       # never seen; loud rather than silent if it is
                print(f"  WARNING {gpu_dir.name}: {k} differs between cards {sorted(vals)}")
        for k in GPU_MEDIAN:
            vals = [c[k] for c in cards if c.get(k) is not None]
            row[k] = statistics.median(vals) if vals else None
        drivers = sorted({c.get("driver_version") for c in cards if c.get("driver_version")})
        row["driver_versions"] = ";".join(drivers)
        out.append(row)
    return pd.DataFrame(out)


def export_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """The aggregated frame as the papers ship it: OOMs corrected, timings otherwise untouched.

    Only the two stages that settle whether a config *ran* are applied.

      stage 0  an OOM recorded as a generic failure becomes an OOM row
      stage 1  a config another card of the same VRAM class OOM'd on becomes an OOM row here too

    Both answer "did this fit", and both leave a row that says so in the ordinary way -- `oom`
    true, no timing. Nothing marks them as edited, because there is nothing unusual left to mark.

    The precision-artifact and slow-outlier stages are deliberately NOT applied. They do not
    decide whether a config ran; they judge whether a timing is trustworthy, and that judgement
    depends on a threshold. Baking it in would hand every reader one particular cut with no way
    to see what it removed. So those rows ship with their measured value, and dataset_cleanup
    reproduces the full four-stage version on demand -- the counts are in MANIFEST.txt so the
    artefacts are known to be there rather than discovered."""
    out = stage1_vram_class(relabel_escaped_ooms(raw.assign(cleanup="")))
    hit = out.class_oom & out.avg_time_ms.notna()
    out.loc[hit, "avg_time_ms"] = np.nan
    out.loc[hit, "oom"] = True
    out = out.drop(columns=["cleanup", "class_oom"])
    return out.drop(columns=[c for c in DROP_COLS if c in out.columns])


def merged_reference(gpu_dir: Path, v1_ref: dict) -> dict | None:
    """v1 reference + a v2 reference built the same way, in one file.

    Every card that published a v2 probe contributes to the v2 arm, with no further screening.
    That is not a shortcut: main.py runs the health check first and only reaches the v2 extension
    on a card that already passed it, so the existence of a v2 file *is* the pass certificate.
    Screening again here would apply the rule twice.

    The frozen v1 healthy_uuids cannot be reused for this. They name the cards in the snapshot v1
    was built from, and every card that has since published a v2 probe was rented afterwards --
    the two sets are disjoint, so intersecting them yields nothing. The populations differ by
    construction; what makes the two arms comparable is that both consist only of cards that
    passed the same screen against the same frozen reference."""
    cards = {p.stem.split("_")[-1]: json.loads(p.read_text())
             for p in sorted(gpu_dir.glob("probe_ext_v2_*.json"))}
    v2 = _reduce(cards, list(cards)) if cards else None

    out = {k: v1_ref[k] for k in ("gpu", "variant", "compute_capability", "sm_count", "torch",
                                  "heavy_fraction", "perf_tolerance", "n_cards_total",
                                  "n_cards_healthy", "healthy_uuids") if k in v1_ref}
    out["version"] = "full"
    out["arms"] = ["v1"] + (["v2"] if v2 else [])
    out["v1"] = {"n_cards": v1_ref.get("n_cards_healthy"),
                 **{k: v1_ref[k] for k in ("n_rows_common", "n_rows_heavy", "rows")
                    if k in v1_ref}}
    if v2:
        out["v2"] = {**v2, "uuids": sorted(cards)}
    else:
        out["v2_absent_reason"] = (f"{len(cards)} v2 probe(s) published, "
                                   f"need {MIN_REFERENCE_PROBES}")
    return out


def export(paper: str, workloads: list[str], root: Path, snapshot: Path) -> None:
    out = root / f"{paper}_paper" / "profiling_results"
    out.mkdir(parents=True, exist_ok=True)
    lines = [f"paper: {paper}", f"source: {REPO}", f"workloads: {', '.join(workloads)}", ""]

    info = gpu_info(snapshot)
    info.to_csv(out / "gpu_info.csv", index=False)
    n_gpu = len(info)

    ref_src, ref_dst = snapshot / "reference", out / "reference"
    n_ref = n_v2 = 0
    if ref_src.exists():
        ref_dst.mkdir(exist_ok=True)
        for f in sorted(ref_src.glob("*_reference_v1.json")):
            v1 = json.loads(f.read_text())
            gpu = v1.get("gpu") or f.name.replace("_reference_v1.json", "")
            m = merged_reference(snapshot / gpu, v1)
            if m is None:
                continue
            (ref_dst / f"{gpu}_reference_full.json").write_text(json.dumps(m, indent=1))
            n_ref += 1
            n_v2 += "v2" in m["arms"]

    lines += [f"GPU models: {n_gpu}",
              f"gpu_info.csv -- one row per GPU model, joins the frame on `gpu`.",
              f"  silicon identity verbatim ({', '.join(GPU_CONSTANT)});",
              f"  median over that model's cards for clocks, power and measured TFLOPS;",
              f"  CPU fields and uuid omitted -- they describe the rented host, not the GPU.",
              f"merged references: {n_ref}  ({n_v2} with a v2 arm)",
              f"  reference/{{gpu}}_reference_full.json",
              f"    v1  frozen health reference: per-row median over that model's healthy cards,",
              f"        cut to the heaviest {HEAVY_FRACTION:.0%} per kernel family (788 of 3165 rows)",
              f"    v2  per-row median over every card that published a v2 probe, all 381 rows.",
              f"        No screen: main.py only reaches v2 after the v1 check passes. No heavy",
              f"        cut: v2 is already the large end v1 was missing.",
              ""]

    for w in workloads:
        try:
            raw = load(w)
        except RuntimeError as e:
            lines.append(f"{w}: NOT EXPORTED -- {e}")
            print(f"  {paper}: skipping {w} -- {e}")
            continue
        shipped = export_frame(raw)
        shipped.to_csv(out / f"{w}_clean.csv", index=False)

        # What the two unapplied stages WOULD flag, reported but not acted on -- so the artefacts
        # are documented rather than left to be rediscovered.
        full = clean(raw)
        flagged = full.cleanup.isin(["precision_artifact", "slow_outlier"])
        n_oom = int((shipped.oom.fillna(False) & shipped.avg_time_ms.isna()).sum())
        lines += [
            f"{w}_clean.csv",
            f"  rows                 {len(shipped):,}  (one per config per GPU model, "
            f"median across cards of that model)",
            f"  timings              {int(shipped.avg_time_ms.notna().sum()):,}",
            f"  OOM rows             {n_oom:,}  (config did not fit; no timing by definition)",
            f"  NOT removed, kept as measured:",
            f"    precision_artifact {int((full.cleanup == 'precision_artifact').sum()):,}"
            f"   a precision that departs from its siblings (memory-constrained fallback)",
            f"    slow_outlier       {int((full.cleanup == 'slow_outlier').sum()):,}"
            f"   >=3x the card's own established ratio to its peers",
            f"  reproduce the filtered version with: dataset_cleanup.clean(load('{w}'))",
            f"  columns: {', '.join(shipped.columns)}",
            f"  dropped (constant, redundant, or misleading -- see export_paper_data.DROP_COLS;",
            f"           all of them survive in the per-GPU raw CSVs): {', '.join(DROP_COLS)}",
            "",
        ]
        print(f"  {paper}/{w}: {len(shipped):,} rows, "
              f"{int(shipped.avg_time_ms.notna().sum()):,} timings, {n_oom:,} OOM, "
              f"{int(flagged.sum())} artefact rows kept as measured")

    (out / "MANIFEST.txt").write_text("\n".join(lines) + "\n")
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"  {paper}_paper/profiling_results: {n_gpu} GPU models, {size:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper", action="append", choices=sorted(PAPERS),
                    help="restrict to one paper (repeatable)")
    ap.add_argument("--root", default=".", help="where the *_paper directories go")
    args = ap.parse_args()

    snapshot = Path(snapshot_download(repo_id=REPO, repo_type="dataset"))
    for paper in (args.paper or sorted(PAPERS)):
        export(paper, PAPERS[paper], Path(args.root), snapshot)


if __name__ == "__main__":
    main()
