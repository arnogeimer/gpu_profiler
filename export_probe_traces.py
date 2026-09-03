"""Flatten the probe references into one table, for use outside this repo.

    python export_probe_traces.py                  # -> probe_traces.csv
    python export_probe_traces.py --out t.csv

One row per (GPU model, probe row). The value is the per-row MEDIAN over that model's healthy
cards, so no single degraded rental defines a trace.

    gpu        NVIDIA_GeForce_RTX_4090        joins gpu_info.csv from export_paper_data
    arm        v1 | v2
    probe      attn bmm conv elementwise gemm pool rnn   (v1)
               gemm_ext conv1d conv_ext                  (v2)
    dtype      fp32 | fp16 | bf16
    size       the shape, encoded per family: q1024k1024h12/12d64 for attn,
               m16384n768k768 for gemm. Opaque but exact, and identical across GPUs, so the
               same string names the same work everywhere.
    direction  fwd | fwd_bwd  (conv: fprop | dgrad | wgrad)
    kind       conv/rnn/pool subtype where the family has one, else empty
    causal     attention masking flag, else empty
    flops      forward-equivalent work implied by the shape -- the ranking key, not a measurement
    ms         median milliseconds
    n_cards    how many cards that median was taken over

HEAVY IS FLOPS, NOT TIME. v1 keeps the heaviest 25% within each family, ranked by the work the
shape implies (device_probe.row_flops). The earlier version of this table ranked by measured
time, which made the selection a property of the card: each GPU kept whichever rows IT was slow
on, so the 23 references shared only 647 of 953 rows and no two cards were described on the same
basis. Ranked by FLOPs the row set is identical everywhere.

v2 keeps all 381 rows -- it has no launch-bound tail to cut, bottoming out at 17 us where v1's
raw grid reaches 1 us. The two arms are comparable in weight rather than ordered; v2's
contribution is coverage in shape space (larger GEMM M, wider conv2d, and conv1d, which v1 does
not sample at all).

Sources. v1 probes are taken from the hub AND from the frozen local snapshot, deduplicated by
card UUID: nine workload-contributing cards never had their probe re-uploaded after the hub was
wiped, and the RTX 4070 has none online at all, so the hub alone cannot represent every GPU.
Each card is screened against its model's frozen reference before contributing, exactly as the
fleet screens it. v2 probes come from the hub and are not re-screened -- main.py only reaches the
extension after the v1 check passes, so their existence is the pass certificate.

DROPPED_ROWS are excluded from the v1 candidate pool before ranking. They are the two largest
fp32 RNN rows, which OOM on 8 GB cards under main.MEMORY_FRACTION=0.995 and were removed from
the published references for the same reason. Leaving them in would hand the 8 GB models two
fewer rows than everyone else and break the one property this rebuild is for.
"""
from pathlib import Path
import argparse
import json
import statistics
import sys

import pandas as pd
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import PERF_TOLERANCE, MIN_REFERENCE_PROBES, _variant
from plotting.data import REPO
from profiler.device_probe import HEAVY_FRACTION, _heaviest, row_flops

LOCAL_SNAPSHOT = Path("results/profiling-12-08-2026")
COLS = ["gpu", "arm", "probe", "dtype", "size", "direction", "kind", "causal",
        "flops", "ms", "n_cards"]

# Unmeasurable on 8 GB cards under the current allocator cap; see the module docstring.
DROPPED_ROWS = {
    ("rnn", "fp32", "b128s512i1152h512l1d2", "fwd_bwd", "gru", None),
    ("rnn", "fp32", "b128s512i1152h512l1d2", "fwd_bwd", "lstm", None),
}


def _rows(sig: dict) -> dict:
    out = {}
    for r in sig.get("probes", []):
        if r.get("ms"):
            out[(r.get("probe"), r.get("dtype"), r.get("size"),
                 r.get("direction"), r.get("kind"), r.get("causal"))] = r["ms"]
    return out


def _healthy(cards: dict, ref: dict) -> list:
    """UUIDs of cards that pass their model's frozen reference, as the fleet screens them."""
    pinned = {tuple(r[:-1]): r[-1] for r in ref.get("rows", [])}
    tol = ref.get("perf_tolerance", PERF_TOLERANCE)
    variant = tuple(ref.get("variant", []))
    out = []
    for uid, sig in sorted(cards.items()):
        if variant and _variant(sig) != variant:
            continue
        mine = _rows(sig)
        usable = [k for k in pinned if k in mine]
        if len(usable) < len(pinned) * 0.9:      # a probe that barely overlaps is not comparable
            continue
        if statistics.median([mine[k] / pinned[k] for k in usable]) <= tol:
            out.append(uid)
    return out


def _median_rows(cards: dict, uuids: list) -> tuple[dict, int]:
    maps = [_rows(cards[u]) for u in uuids]
    if len(maps) < MIN_REFERENCE_PROBES:
        return {}, 0
    common = set.intersection(*[set(m) for m in maps])
    return {k: statistics.median([m[k] for m in maps]) for k in common}, len(maps)


def build(hub: Path, snap: Path) -> pd.DataFrame:
    out = []
    for f in sorted((hub / "reference").glob("*_reference_v1.json")):
        ref = json.loads(f.read_text())
        gpu = ref.get("gpu") or f.name.replace("_reference_v1.json", "")

        # v1: hub and local snapshot unioned, deduplicated by card UUID.
        cards = {}
        for root in (snap / gpu, hub / gpu):
            if root.exists():
                for p in sorted(root.glob("device_probe_*.json")):
                    cards[p.stem.split("_")[-1]] = json.loads(p.read_text())
        healthy = _healthy(cards, ref)
        med, n = _median_rows(cards, healthy)
        med = {k: v for k, v in med.items() if k not in DROPPED_ROWS}
        for k in sorted(_heaviest(med, HEAVY_FRACTION)):
            out.append((gpu, "v1", *k, row_flops(k), med[k], n))

        # v2: hub only, every published probe, no heavy cut.
        v2 = {p.stem.split("_")[-1]: json.loads(p.read_text())
              for p in sorted((hub / gpu).glob("probe_ext_v2_*.json"))} if (hub / gpu).exists() else {}
        med2, n2 = _median_rows(v2, list(v2))
        for k in sorted(med2):
            out.append((gpu, "v2", *k, row_flops(k), med2[k], n2))
    return pd.DataFrame(out, columns=COLS)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="probe_traces.csv")
    args = ap.parse_args()

    hub = Path(snapshot_download(repo_id=REPO, repo_type="dataset"))
    df = build(hub, LOCAL_SNAPSHOT)
    df.to_csv(args.out, index=False)

    print(f"{args.out}: {len(df):,} rows, {df.gpu.nunique()} GPU models, "
          f"{Path(args.out).stat().st_size / 1e6:.1f} MB")
    for arm, s in df.groupby("arm"):
        # t["size"], never t.size -- the latter is DataFrame.size, an int.
        sets = {g: frozenset(zip(t["probe"], t["dtype"], t["size"], t["direction"],
                                 t["kind"].fillna(""), t["causal"].fillna("")))
                for g, t in s.groupby("gpu")}
        ident = len(set(sets.values())) == 1
        print(f"  {arm}: {len(s):,} rows over {s.gpu.nunique()} models, "
              f"{len(next(iter(sets.values())))} rows each   "
              f"row set identical across GPUs: {'YES' if ident else 'NO'}")
        print(f"       cards per model: {s.groupby('gpu').n_cards.first().min()}"
              f"-{s.groupby('gpu').n_cards.first().max()}   "
              f"families: {', '.join(f'{k} {v}' for k, v in sorted(s.probe.value_counts().items()))}")
    missing = sorted(set(df[df.arm == "v1"].gpu) - set(df[df.arm == "v2"].gpu))
    print(f"  models lacking a v2 arm: {missing or 'none'}")


if __name__ == "__main__":
    main()
