"""Drop rows from every v1 reference that the probe can no longer produce.

The frozen v1 reference pins the exact row keys a card must supply, and treats a missing one as
disqualifying -- a probe that stopped emitting a pinned row is a probe that changed underneath
the reference, and scoring on what survives would compare two different things.

That premise broke for one row pair. `MEMORY_FRACTION = 0.995` was added to main.py after the
reference was frozen, and the two largest fp32 RNN rows now OOM on 8 GB cards where they
previously ran (~200 ms on a 3060 Ti). Before the cap they squeezed in against a card with no
allocator limit, almost certainly by thrashing. So the rows are not measurable on that hardware
any more, and every 8 GB probe -- all 12 of them, across the 3060 Ti, 3070 and 5060 -- fails the
screen for missing them and contributes nothing. The 3060 Ti has never produced a workload row
because of it.

Removed from ALL references, not only the 8 GB ones. The screen's ratio is a median over the
pinned rows, so leaving the pair in place for larger cards would mean each GPU model was judged
over a different row set, and a verdict on a 4090 would no longer mean the same thing as a
verdict on a 3070. Uniformity is worth more than two rows out of 788.

This edits published references in place, under the same v1 name, which is a real cost: a card
screened before this ran was judged on 788 rows and one screened after on 786. The alternative
-- a v1.1 that every node must be told about -- buys strictly less, because the change only ever
REMOVES a constraint and cannot flip a pass to a fail. Each amended file records what was taken
out and why, so the edit is auditable rather than silent.

    python amend_reference.py            # write locally, show what changes
    python amend_reference.py --push     # ...and upload to the hub
"""
from pathlib import Path
import argparse
import io
import json
import os
import sys

from huggingface_hub import snapshot_download, upload_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plotting.data import REPO

# The rows to drop, as (probe, dtype, size, direction, kind, causal).
# `causal` is null on RNN rows -- it only applies to attention -- so the key ends in None, not
# False. Matching on False silently removed nothing.
DROP = [
    ["rnn", "fp32", "b128s512i1152h512l1d2", "fwd_bwd", "gru", None],
    ["rnn", "fp32", "b128s512i1152h512l1d2", "fwd_bwd", "lstm", None],
]
REASON = ("OOMs on 8 GB cards under main.MEMORY_FRACTION=0.995, which was introduced after this "
          "reference was frozen; removed from every model so the screen judges each on the same "
          "row set")


def amend(ref: dict) -> tuple[dict, int]:
    """Return (amended reference, rows removed). Idempotent."""
    drop = {tuple(d[:6]) for d in DROP}
    kept = [r for r in ref["rows"] if tuple(r[:6]) not in drop]
    n = len(ref["rows"]) - len(kept)
    if n:
        ref = dict(ref)
        ref["rows"] = kept
        ref["n_rows_heavy"] = len(kept)
        ref["amended_removed_rows"] = DROP
        ref["amended_reason"] = REASON
    return ref, n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--push", action="store_true", help="upload the amended files to the hub")
    ap.add_argument("--out", default="reference_amended")
    args = ap.parse_args()

    loc = Path(snapshot_download(repo_id=REPO, repo_type="dataset"))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    # Falls back to the cached CLI login, so this works without HF_TOKEN exported.
    token = os.environ.get("HF_TOKEN")
    if args.push and not token:
        cached = Path.home() / ".cache/huggingface/token"
        if not cached.exists():
            raise SystemExit("no HF_TOKEN and no cached login; refusing to push")
        token = cached.read_text().strip()

    print(f"{'gpu':24} {'before':>7} {'after':>7} {'removed':>8}")
    total = pushed = 0
    for f in sorted((loc / "reference").glob("*_reference_v1.json")):
        ref = json.loads(f.read_text())
        new, n = amend(ref)
        total += n
        print(f"{f.name.replace('_reference_v1.json','').replace('NVIDIA_GeForce_',''):24} "
              f"{len(ref['rows']):7} {len(new['rows']):7} {n:8}")
        payload = json.dumps(new, indent=1)
        (outdir / f.name).write_text(payload)
        if args.push and n:
            upload_file(path_or_fileobj=io.BytesIO(payload.encode()), repo_id=REPO,
                        repo_type="dataset", path_in_repo=f"reference/{f.name}", token=token,
                        commit_message=f"amend {f.name}: drop 2 unmeasurable fp32 RNN rows")
            pushed += 1
    print(f"\n{total} rows removed across the fleet -> {outdir}/")
    print(f"pushed: {pushed} file(s)" if args.push else "not pushed (use --push)")


if __name__ == "__main__":
    main()
