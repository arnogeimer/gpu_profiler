"""Flatten the workload measurements into one table per GPU model, for use outside this repo.

    python export_gpu_traces.py                   # -> gpu_traces/{gpu}.csv
    python export_gpu_traces.py --out traces

The workload half of what probe_traces.csv is the hardware half of. Both are keyed on `gpu`, both
are already reduced to the per-model median, and together they are the whole experiment: what the
card does on a synthetic kernel, and what it does on a real training step.

One file per GPU model rather than one table, because that is how the data is consumed -- a node
predicting its own step times needs its own card's traces and nothing else, and the per-model files
are 25-90 KB against the 3.9 MB of the concatenation. `pd.concat` over the folder gives that
concatenation back exactly; `gpu` is kept as a column for precisely that reason, and so the file
joins gpu_info.csv and probe_traces.csv without parsing its own filename.

The sweep that produced these rows travels with them, as gpu_traces/image_classification.py -- a
copy of the workload the fleet ran, so the folder says what was measured and how, the way
probe_traces.py sits beside probe_traces.csv. It is a copy for reading, not an import path; the
live one is workloads/computer_vision/image_classification.py.

Columns are the config axes, the target, and nothing else:

    model batch_size img_size precision   the config -- axes are read off the frame, so a
                                          workload with different axes needs no change here
    avg_time_ms                           median step time across that model's cards, or empty
    oom                                   True if the config did not fit
    gpu vram_gb                           the card
    cleanup                               see below; empty for 99.5% of rows

WHAT IS APPLIED. The two stages that settle whether a config *ran*: stage 0 recovers OOMs that
were recorded as generic failures, stage 1 marks a config OOM for a whole VRAM class once one card
of the class OOM'd on it. Both leave an ordinary OOM row -- `oom` true, no timing -- and neither
involves a threshold.

WHAT IS NOT. The two stages that judge whether a timing is *trustworthy* -- precision_artifact and
slow_outlier -- are labelled in `cleanup` and left in place with their measured value. They depend
on a cut (3x a card's own established ratio to its peers, and so on), and baking one in would hand
every consumer a single choice with no way to see what it removed. Labelling instead of removing
is the difference from export_paper_data, which ships the same frame with no such column and
documents the counts in prose: a file feeding a compute-time *distribution* should not carry a
30x-slow row silently, and `df[df.cleanup == ""]` is the filtered version.

Rows are the aggregate across every card of a model, median per config, as plotting.data.load
builds them. A card outside PERF_TOLERANCE never contributed one -- main.py screens it against its
model's probe reference before it may run a workload at all.
"""
from pathlib import Path
import argparse
import shutil
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dataset_cleanup import clean
from export_paper_data import export_frame
from plotting.data import REPO, load

WORKLOAD = "image_classification"
SOURCE = Path("workloads/computer_vision/image_classification.py")

# The judgement stages, labelled rather than applied. The other two labels clean() can emit --
# oom_relabelled and vram_class_oom -- are deliberately absent: export_frame has already acted on
# them, and the rows they produced are ordinary OOM rows with nothing unusual left to mark.
KEPT_LABELS = ["precision_artifact", "slow_outlier"]


def build(workload: str) -> pd.DataFrame:
    """The shipped frame, with the unapplied stages' verdicts in `cleanup`.

    clean() and export_frame() both derive from the same `raw` and both preserve its index, so
    the verdict column aligns by index -- no merge, which would return a fresh RangeIndex and
    label the wrong rows."""
    raw = load(workload)
    out = export_frame(raw)
    verdict = clean(raw).cleanup
    out["cleanup"] = verdict.where(verdict.isin(KEPT_LABELS), "")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--workload", default=WORKLOAD)
    ap.add_argument("--out", help="folder for the per-GPU CSVs (default: gpu_traces)")
    args = ap.parse_args()

    df = build(args.workload)
    # Files are named by GPU alone, so a second workload needs its own folder or it silently
    # overwrites the first -- and the stale sweep below would delete what it did not replace.
    out = Path(args.out or ("gpu_traces" if args.workload == WORKLOAD
                            else f"gpu_traces_{args.workload}"))
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.csv"):     # a GPU dropped from the fleet should not linger
        stale.unlink()

    for gpu, t in df.groupby("gpu"):
        t.sort_values([c for c in t.columns if c not in ("avg_time_ms", "oom", "cleanup")]) \
         .to_csv(out / f"{gpu}.csv", index=False)

    # The sweep definition beside the sweep's results. Copied on every export so it cannot drift
    # from the code that actually ran.
    src = Path(__file__).resolve().parent / SOURCE
    if args.workload == WORKLOAD and src.exists():
        shutil.copyfile(src, out / src.name)

    size = sum(f.stat().st_size for f in out.glob("*.csv")) / 1e6
    print(f"{out}/: {df.gpu.nunique()} GPU models, {len(df):,} rows, {size:.1f} MB   "
          f"source: {REPO}")
    print(f"  columns: {', '.join(df.columns)}")
    print(f"  workload definition: {out / src.name}")

    n = df.groupby("gpu").size()
    ident = n.nunique() == 1
    print(f"  rows per GPU: {n.min():,}-{n.max():,}   "
          f"every model swept identically: {'YES' if ident else 'NO'}")
    if not ident:
        short = n[n < n.max()]
        print("       short: " + ", ".join(f"{g.replace('NVIDIA_GeForce_', '')} {v:,}"
                                           for g, v in short.items()))

    timed = int(df.avg_time_ms.notna().sum())
    n_oom = int((df.oom.fillna(False) & df.avg_time_ms.isna()).sum())
    print(f"  timings {timed:,}   OOM {n_oom:,}   neither {len(df) - timed - n_oom:,}")
    for label in KEPT_LABELS:
        k = int((df.cleanup == label).sum())
        print(f"  flagged, kept as measured: {label:19} {k:,}")


if __name__ == "__main__":
    main()
