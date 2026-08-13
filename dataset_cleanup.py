"""Remove timings that measure a memory-constrained fallback rather than the model.

The mechanism is described in README under the OOM section: a config that lands *just* inside
the allocator budget still completes, but with less workspace than cuDNN wants, so it picks a
slower algorithm and returns a plausible-looking number. `max_memory_used_pct` cannot separate
those from healthy rows -- both read 100%, because it is a whole-device measure that includes
the CUDA context while the cap applies only to torch's own allocations.

Two stages, in order, because the first changes what the second sees.

================================================================================================
STAGE 1 -- VRAM class
================================================================================================

If any card of a given VRAM class OOM'd on a config, the config is marked OOM for every card of
that class. Fitting is a property of the config and the capacity, so a lone survivor is a card
that got lucky with allocator ordering, not a card that can do something its classmates cannot.

This is exactly true where it can be checked. Across image classification's 4 860
(config, VRAM class) cells there is not one disagreement -- every class either OOMs unanimously
or times unanimously -- so the stage is a no-op there and rests on 4 860 confirmations. Audio has
16 mixed cells, and the stage resolves them by relabelling 30 survivor rows.

It is deliberately conservative rather than precise. Judged against each card's own typical
speed factor, roughly two thirds of those 30 survivors were measuring cleanly; the rule discards
them anyway. Two reasons that is the right trade for this dataset. Every one of them sits at
100% memory, and a config that only runs on the luckiest card in its class is not a runtime a
scheduler should promise a node. And the alternative -- gating on what fraction of the class
OOM'd, corroborated by a per-card slowdown estimate -- buys a handful of rows for a rule nobody
can check by eye.

Two things it does NOT claim. "Same VRAM" is not the same capacity: the 16 GB class spans
16 302-16 379 MiB and the 4060 Ti sits at the top, so it genuinely fits configs its classmates
cannot -- its wavlm-large 16 s batch 4 fp16 row measured cleanly and is discarded regardless.
And the class is the rounded GB, not the exact MiB, because that is the granularity anything
downstream will reason about.

================================================================================================
STAGE 2 -- precision ratio
================================================================================================

Stage 1 only reaches configs that pushed some card of the class over the line. A config can
degrade every card in its class without OOMing any of them, and then the class agrees and stage
1 sees nothing. What catches those is a precision ratio. fp16 and bf16 have identical footprints
and identical tensor-core throughput on every card in the fleet, and fp32 is slower by a factor
that is stable per config across the fleet. So a config where *one* precision on *one* card
departs from what its siblings and its peer cards do is measuring something other than the model.

Two arms, because they fail in different places:

  fp32/fp16   the original arm. Catches the large-activation vision cases, where fp32 is the
              only precision heavy enough to cross the line. Blind exactly where the config is
              so large that fp16 OOMs too, since then there is no denominator.
  bf16/fp16   covers that blind spot. Both siblings survive the same configs, so the ratio
              exists wherever either does, and it is the tighter of the two: peer-relative p99
              is 1.13 and 1.12, against 1.43 and 1.71 for fp32/fp16.

WHY NOT A FLAT THRESHOLD. The "fp32/fp16 > 8" rule in README was fitted on 692 groups, where the
healthy p99 was 6.24 and the lowest artifact 16.1. At 7 637 groups that gap has closed to
7.57 / 7.98 -- a 5% margin, on a quantity whose healthy value depends on the model. convnext at
224 px legitimately runs 3.0-7.6 fp32/fp16 on every card in the fleet, so a flat cut placed for
convnext is far too loose for a model whose baseline is 1.3, and one placed for that model
deletes convnext everywhere.

WHY NOT PEER NORMALISATION ALONE. Dividing by the median ratio across cards at the same config
removes the model term, and where only a minority of cards are affected it isolates the artifact
exactly. But the peer median is itself a measurement, and at the configs that sit on the
capacity boundary of a whole VRAM class it is contaminated: at vit_large 224/32 the four 24 GB+
cards run 2.70-3.18 while all six 16 GB cards run 7.98-53.99 at 100% memory. Six of ten peers
are artifacts, the median is 9.59, and the worst-affected 16 GB card scores rel 0.83 -- healthier
than healthy.

So three rules, each covering the case the others miss, and a group is dropped if any fires:

  peer_relative    ratio / median ratio across all cards at this config      minority affected
  clean_relative   ratio / median ratio across cards NOT at >=99% memory     majority affected
  raw              the ratio on its own                                      no usable peer set

`clean_relative` only applies to a group that is itself at >=99% memory and only where at least
one unpressured card measured the same config -- it asks "what did this config cost on a card
with room to run it", which is the question the contaminated median cannot answer. It rests on a
single witness in most cases (n=1 for 12 of the 14 image flags), so it is deliberately the
loosest of the three and never the only evidence reported.

Thresholds sit at roughly twice each arm's healthy p99, in that arm's own units:

  arm         healthy p99 (img/audio)   threshold   raw backstop
  fp32/fp16   1.43 / 1.71               3.5         8.0
  bf16/fp16   1.13 / 1.12               2.5         2.0

RAW_FLOOR keeps a group from being flagged on a relative rule alone when the peers are merely
fast: a config whose peer ratio is 0.5 can reach rel 4 while still being quicker in the low
precision than in the high one, which is not an artifact of any kind.

The boundary is fuzzy and the report is meant to be read, not just applied. Degradation is
continuous -- at whisper-medium 16 s batch 8 every 16 GB card is affected, from clean_relative
9.80 down to 3.43, and any cut inside that range separates siblings that failed the same way.
Flags at low memory are a different phenomenon again: pvt_v2_b0 and mobilevit_xxs at 64 px flag
on the bf16 arm at 12-20% memory, where no fallback can be involved and the cause is kernel
selection on kernels of a few milliseconds. Both are per-card measurement departures and both
are unusable as ground truth, but only the first is evidence about VRAM.

================================================================================================

Both stages null `avg_time_ms` rather than deleting the row, and record which stage did it in a
`cleanup` column. A dropped row still carries its config, its memory readings and (for stage 1)
its OOM flag, which is what makes the removal auditable and lets a consumer decide to reinstate
it. Stage 1 additionally sets `oom` -- the config did not fit the class -- while stage 2 leaves
`oom` alone, since an artifact config demonstrably ran.

Usage:
    python dataset_cleanup.py                     # report both stages, both workloads
    python dataset_cleanup.py --csv removed.csv   # also write out what was removed

    from dataset_cleanup import clean
    df = clean(load("audio_classification"))      # cleaned frame, `cleanup` column added
"""
import argparse

import numpy as np
import pandas as pd

from plotting.data import load, config_keys

WORKLOADS = ["image_classification", "audio_classification"]

# Stage 1 groups cards by rounded GB rather than exact MiB: the class spans ~80 MiB within a GB
# and nothing downstream reasons at finer granularity than the marketing capacity.
MIB_PER_GB = 1024

# Ratios are (numerator precision / fp16), so >1 means the numerator is slower.
ARMS = [("fp32", "fp16"), ("bf16", "fp16")]

# Applied to both relative rules. See the docstring for placement; the arms have different
# spreads and the numbers are not interchangeable between them.
REL_THRESHOLD = {"fp32": 3.5, "bf16": 2.5}

# A relative rule also needs the raw ratio to be unreasonable on its own terms.
RAW_FLOOR = {"fp32": 3.0, "bf16": 1.3}

# The backstop, for groups with no usable peer set at all.
RAW_MAX = {"fp32": 8.0, "bf16": 2.0}

# Below this many cards the peer median summarises nothing, so peer_relative does not apply.
MIN_PEERS = 4

# Peak device memory at or above this counts as pressured: it decides which cards are excluded
# from the clean witness set, gates the clean_relative rule, and labels a flag's reason.
MEM_PRESSURE_PCT = 99.0

# config_keys treats every non-result column as a config axis, so anything added to the frame --
# by plotting.data.load, or by stage 1 here -- has to be named or it silently becomes an axis and
# splits the peer groups.
_NOT_CONFIG = {"gpu", "n_attempted", "n_timed", "phase", "precision",
               "vram_gb", "class_oom", "cleanup"}


def _axes(df: pd.DataFrame) -> list[str]:
    """The config axes of this workload, minus precision -- what a group is compared within."""
    return [c for c in config_keys(df) if c not in _NOT_CONFIG]


def vram_gb(df: pd.DataFrame) -> pd.Series:
    """Rounded VRAM per GPU model, derived from the rows themselves.

    `max_memory_used_mb / max_memory_used_pct` recovers the device total, so this needs no
    host_info alongside the CSVs -- it agrees with the probe's `vram_total_mb` to within 0.6 MB
    on all ten GPU models. The median over rows rather than any single row, because the two
    columns are sampled by a background thread and need not come from the same instant."""
    d = df[df.max_memory_used_pct.gt(0) & df.max_memory_used_mb.notna()]
    total = d.max_memory_used_mb / (d.max_memory_used_pct / 100.0)
    return (total.groupby(d.gpu).median() / MIB_PER_GB).round().astype(int)


def stage1_vram_class(df: pd.DataFrame) -> pd.DataFrame:
    """Mark a config OOM for a whole VRAM class as soon as one card of the class OOM'd.

    Returns the frame with `vram_gb` and `class_oom` added. `class_oom` is True for every row of
    a (config, precision, class) cell in which any card recorded an OOM, including the rows that
    produced a timing -- those are the ones stage 1 exists to remove."""
    out = df.copy()
    out["vram_gb"] = out.gpu.map(vram_gb(df))
    cell = _axes(df) + ["precision", "vram_gb"]
    # A row counts as an OOM witness only if it OOM'd *and* produced no timing. A config that
    # OOM'd on one instance of a GPU model and ran on another aggregates to oom=True with a
    # median time attached (see plotting.data), and that is a healthy config, not a witness.
    out["_witness"] = out.oom.fillna(False) & out.avg_time_ms.isna()
    out["class_oom"] = out.groupby(cell, dropna=False)["_witness"].transform("any")
    return out.drop(columns="_witness")


def ratio_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (gpu, config, arm) with the ratio and both peer baselines.

    Only configs that produced a timing in both precisions of an arm appear: a ratio against an
    OOM'd or errored sibling does not exist, and inventing one would flag the survivor."""
    axes = _axes(df)
    timed = df[df.avg_time_ms.notna()]
    ms = timed.pivot_table(index=["gpu"] + axes, columns="precision",
                           values="avg_time_ms", aggfunc="median")
    mem = timed.pivot_table(index=["gpu"] + axes, columns="precision",
                            values="max_memory_used_pct", aggfunc="max")
    levels = list(range(1, len(axes) + 1))    # every index level except gpu

    out = []
    for num, den in ARMS:
        if num not in ms or den not in ms:
            continue
        t = pd.DataFrame({"ratio": ms[num] / ms[den]}).dropna()
        if t.empty:
            continue
        t["arm"] = f"{num}/{den}"
        t["num_ms"], t["den_ms"] = ms[num], ms[den]
        # Peak memory over the two precisions of this arm, not over all three: fp32 can be
        # pressured at a config where the fp16 sibling has room to spare, and it is the arm's
        # own worst case that decides whether a fallback could be involved.
        t["mem_pct"] = mem[[num, den]].max(axis=1)

        t["n_peers"] = t.groupby(level=levels)["ratio"].transform("size")
        peer = t.groupby(level=levels)["ratio"].transform("median")
        # NaN rather than a number below MIN_PEERS: with one card the "peer median" is the row
        # itself and rel is 1.00 by construction, which reads as healthy for the worst rows in
        # the set. Those groups fall through to the raw backstop instead.
        t["peer_ratio"] = peer.where(t.n_peers >= MIN_PEERS)
        t["peer_relative"] = t.ratio / t.peer_ratio

        clean = t[t.mem_pct < MEM_PRESSURE_PCT].groupby(level=levels)["ratio"]
        t = t.join(clean.median().rename("clean_ratio")).join(clean.size().rename("n_clean"))
        t["n_clean"] = t.n_clean.fillna(0).astype(int)
        # Only meaningful for a pressured row: an unpressured one is its own witness.
        t["clean_relative"] = (t.ratio / t.clean_ratio).where(
            (t.mem_pct >= MEM_PRESSURE_PCT) & (t.n_clean > 0))
        out.append(t.reset_index())
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def flag(table: pd.DataFrame) -> pd.DataFrame:
    """Add `artifact`, `rule` and `reason` to a ratio_table. One row in, one row out."""
    t = table.copy()
    num = t.arm.str.split("/").str[0]
    rel_thr, floor, raw_thr = num.map(REL_THRESHOLD), num.map(RAW_FLOOR), num.map(RAW_MAX)

    over_floor = t.ratio >= floor
    hits = {
        "peer_relative": (t.peer_relative >= rel_thr) & over_floor,
        "clean_relative": (t.clean_relative >= rel_thr) & over_floor,
        "raw": t.ratio >= raw_thr,
    }
    t["artifact"] = np.logical_or.reduce([h.fillna(False) for h in hits.values()])
    t["rule"] = ["+".join(k for k, h in hits.items() if bool(h.iloc[i]))
                 for i in range(len(t))]
    t["reason"] = np.where(~t.artifact, "",
                           np.where(t.mem_pct >= MEM_PRESSURE_PCT,
                                    "memory_fallback", "kernel_selection"))
    return t


def flag_workload(workload: str) -> pd.DataFrame:
    """Stage 2 flagged groups for one workload, worst first."""
    t = flag(ratio_table(load(workload)))
    t = t[t.artifact].copy()
    t["workload"] = workload
    return t.sort_values("ratio", ascending=False)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Both stages. Returns the frame with `cleanup` added and removed timings nulled.

    Stage order matters and is not a preference: stage 1 removes timings that stage 2 would
    otherwise use as peers, and a peer baseline built from rows already known to be unusable is
    worth less than one built without them.

    Single pass on purpose. Feeding the output back in finds one further audio row, because each
    removal shrinks a peer set and can move a median across a threshold. Iterating to a fixed
    point would let the cut depth depend on how many rounds it happened to take rather than on
    the data, so the rule is applied once, to the dataset as measured."""
    axes = _axes(df)
    out = stage1_vram_class(df)
    out["cleanup"] = ""

    hit = out.class_oom & out.avg_time_ms.notna()
    out.loc[hit, ["avg_time_ms", "cleanup", "oom"]] = [np.nan, "vram_class_oom", True]

    # Stage 2 re-derives its ratios from the post-stage-1 frame, so a config whose fp16 sibling
    # stage 1 removed is correctly treated as having no denominator rather than a stale one.
    bad = flag(ratio_table(out))
    bad = bad[bad.artifact]
    if len(bad):
        # Only the numerator's row is removed, not the whole group. A high ratio means the
        # numerator is slow *relative to* the denominator, and the denominator is what
        # established that -- were it degraded too, the ratio would sit near 1 and nothing
        # would have flagged. Noise here is one-sided (see profiler.time_fn: minimum of
        # repeats), so "fp16 was anomalously fast" is not an available explanation. Dropping
        # the group would discard the healthy sibling that did the detecting.
        marks = bad[["gpu"] + axes].assign(precision=bad.arm.str.split("/").str[0])
        idx = out.reset_index().merge(marks.drop_duplicates(),
                                      on=["gpu"] + axes + ["precision"])["index"]
        hit2 = out.index.isin(idx) & out.avg_time_ms.notna()
        out.loc[hit2, ["avg_time_ms", "cleanup"]] = [np.nan, "precision_artifact"]
    return out.drop(columns="class_oom")


def _fmt(x: float, width: int = 7) -> str:
    return f"{x:{width}.2f}" if pd.notna(x) else "-".rjust(width)


def _report_stage1(workload: str, df: pd.DataFrame) -> None:
    axes = _axes(df)
    s = stage1_vram_class(df)
    cell = axes + ["precision", "vram_gb"]
    g = s.groupby(cell, dropna=False).agg(
        n_oom=("avg_time_ms", lambda v: v.isna().sum()), n_timed=("avg_time_ms", "count"))
    mixed = g[(g.n_oom > 0) & (g.n_timed > 0)]
    removed = s[s.class_oom & s.avg_time_ms.notna()]

    classes = "/".join(str(int(v)) for v in sorted(s.vram_gb.unique()))
    print(f"\n  STAGE 1  VRAM class      {len(g):,} (config, class) cells over "
          f"{s.vram_gb.nunique()} classes: {classes} GB")
    print(f"           {len(g) - len(mixed):,} unanimous, {len(mixed)} mixed "
          f"-> {len(removed)} timings removed ({len(removed) / len(df):.2%} of rows)")
    if len(removed):
        print(f"      {'gpu':18} {'GB':>3}  {'ms':>9}  {'mem':>6}  config")
        for _, r in removed.sort_values("avg_time_ms", ascending=False).iterrows():
            cfg = " ".join(str(r[c]) for c in axes + ["precision"])
            print(f"      {r.gpu.replace('NVIDIA_GeForce_', ''):18} {r.vram_gb:3d}  "
                  f"{r.avg_time_ms:9.1f}  {r.max_memory_used_pct:5.1f}%  {cfg}")


def _report(workload: str) -> pd.DataFrame:
    df = load(workload)
    axes = _axes(df)
    print(f"\n{'=' * 108}\n{workload}   {len(df):,} aggregated rows   axes: {', '.join(axes)}")
    _report_stage1(workload, df)

    # Stage 2 is reported on the post-stage-1 frame, matching what clean() applies.
    staged = stage1_vram_class(df).copy()
    staged.loc[staged.class_oom, "avg_time_ms"] = np.nan
    t = flag(ratio_table(staged))
    print(f"\n  STAGE 2  precision ratio")

    for arm in t.arm.unique():
        a = t[t.arm == arm]
        ok = a[~a.artifact & a.peer_relative.notna()]
        q = np.percentile(ok.peer_relative, [50, 90, 99])
        print(f"\n  {arm:11} {len(a):5,} groups   peer-relative: med {q[0]:.2f}  p90 {q[1]:.2f} "
              f" p99 {q[2]:.2f}   raw median {ok.ratio.median():.2f}")
        bad = a[a.artifact].sort_values("ratio", ascending=False)
        if bad.empty:
            print("              no groups flagged")
            continue
        print(f"              {len(bad)} flagged ({len(bad) / len(a):.2%})")
        print(f"      {'raw':>7} {'peer':>7} {'clean':>7}  {'mem':>6}  "
              f"{'gpu':18} {'config':44} rule")
        for _, r in bad.iterrows():
            cfg = " ".join(str(r[c]) for c in axes)
            print(f"      {_fmt(r.ratio)} {_fmt(r.peer_relative)} {_fmt(r.clean_relative)} "
                  f" {r.mem_pct:5.1f}%  {r.gpu.replace('NVIDIA_GeForce_', ''):18} {cfg:44} "
                  f"{r.rule}")

    bad = t[t.artifact]
    if len(bad):
        print(f"\n           reason: {dict(bad.reason.value_counts())}")
        print(f"           rule:   {dict(bad.rule.value_counts())}")

    done = clean(df)
    n = done[done.cleanup != ""].cleanup.value_counts()
    had, kept = df.avg_time_ms.notna().sum(), done.avg_time_ms.notna().sum()
    print(f"\n  TOTAL    {had - kept} timings removed of {had:,} "
          f"({(had - kept) / max(had, 1):.2%}), {kept:,} kept   {dict(n)}")
    out = done[done.cleanup != ""].copy()
    out["workload"] = workload
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", help="write the removed rows here")
    ap.add_argument("--workload", action="append", help="restrict to one workload (repeatable)")
    args = ap.parse_args()

    removed = [_report(w) for w in (args.workload or WORKLOADS)]
    if args.csv:
        pd.concat(removed, ignore_index=True).to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
