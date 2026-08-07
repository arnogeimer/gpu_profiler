#!/usr/bin/env python3
"""Collapse raw CUDA kernel names so one kernel *type* counts once.

Applied offline to kernel_census.py's output, so the rules can be retuned without
re-running the sweep.

Three granularities, because "the same kernel" has more than one defensible meaning:

  tile   same kernel, any blocking          -> the headline count
         tile/warp/stage/instruction shapes are erased, dtypes and the operation
         (the functor inside the template) are kept. sm80_ vs sm86_ collapse too:
         cuDNN emits both for one op on one device, and they are the same kernel
         built for a different arch.

  dtype  same operation, any precision      -> tile + fp16/fp32/bf16 erased
         An fp16 and an fp32 GEMM are genuinely different kernels on the device,
         so this is the looser reading, useful for counting *operations*.

  family same family, any instantiation     -> dtype + all template arguments gone
         Collapses every elementwise variant to one entry. Too coarse to microbenchmark
         from, but it shows how few distinct kernel shapes actually exist.
"""

import re

# Order matters — the specific size spellings have to go before the generic ones.
# NB: \b does not fire after an underscore, and these names are all underscore-joined
# (_s16816gemm, _ldg8, _stages_), so word starts are spelled as "not preceded by
# alphanumeric" instead.
_S = r"(?<![A-Za-z0-9])"
_E = r"(?![A-Za-z0-9])"

_TILE = [
    (r"^void\s+", ""),
    (_S + r"sm\d+_", "sm_"),               # sm80/sm86: same kernel, different arch build
    (_S + r"cutlass_\d+_", "cutlass_"),    # cutlass_80_: same, spelled differently
    (r"\d+x\d+(?:x\d+)*", "N"),            # tiles: 128x128x32, warpsize2x2x1, tensor16x8x16
    (_S + r"s\d+gemm", "sNgemm"),          # mma shape in the name: s16816gemm / s1688gemm
    (_S + r"s\d+(?:dgrad|wgrad|fprop)", "sNconv"),
    (_S + r"k\d+c\d+r\d+s\d+", "kN"),      # conv geometry: k64c4r7s7
    (r"_t\d+r\d+s\d+", "_tN"),             # filter geometry: t1r3s3
    (_S + r"stages?_?\d*", "stage"),
    (_S + r"ldg\d+", "ldg"),
    (_S + r"align\d+", "align"),
    (r"_g\d+" + _E, "_g"),                 # conv groups
    (r"_v\d+" + _E, "_v"),
    (r"Li\d+E", "LiNE"),                   # mangled cutlass the demangler could not parse
    (r"(?<=[<,])\s*\d+u?l?\s*(?=[,>])", "N"),   # integer template args: vec width, block size
]

_DTYPE = [
    (r"c10::Half|c10::BFloat16|__half|__nv_bfloat16", "T"),
    (r"\b(?:float|double)\b", "T"),
    (r"\bf(?:16|32|64)(?:f(?:16|32|64))*\b", "fT"),   # f16f16_f16f32_f32
    (r"\bfp(?:16|32|64)\b", "fpT"),
    (r"\bs?[hsdi]gemm\b", "gemm"),
]


def _strip_args(name: str) -> str:
    """Drop the trailing C++ argument list, which duplicates the template types.

    Matched by walking back from the final ')' rather than by regex: these names are
    full of other parens — '(anonymous namespace)', 'operator()()', lambda signatures —
    and a greedy pattern eats from the first one, collapsing every at::native kernel
    into the bare namespace."""
    if not name.endswith(")"):
        return name
    depth = 0
    for i in range(len(name) - 1, -1, -1):
        if name[i] == ")":
            depth += 1
        elif name[i] == "(":
            depth -= 1
            if depth == 0:
                return name[:i]
    return name


def _apply(name: str, rules) -> str:
    for pat, rep in rules:
        name = re.sub(pat, rep, name)
    return re.sub(r"\s+", " ", name).strip()


def _strip_templates(name: str) -> str:
    """Drop every <...> block, honouring nesting."""
    out, depth = [], 0
    for ch in name:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out).strip()


def normalize(name: str, level: str = "tile") -> str:
    """level: 'tile' | 'dtype' | 'family' — see the module docstring."""
    n = _apply(_strip_args(name), _TILE)
    if level == "tile":
        return n
    n = _apply(n, _DTYPE)
    if level == "dtype":
        return n
    return _strip_templates(n)


def main() -> None:
    import argparse
    import collections
    import json

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("census", help="kernels.json from kernel_census.py")
    ap.add_argument("--level", default="tile", choices=("tile", "dtype", "family"))
    ap.add_argument("--show", type=int, default=0, help="print the N largest groups")
    ap.add_argument("--out", help="write the normalized set as JSON")
    args = ap.parse_args()

    raw = json.load(open(args.census))["kernels"]
    groups: dict[str, list[str]] = collections.defaultdict(list)
    for name in raw:
        groups[normalize(name, args.level)].append(name)

    print(f"{len(raw)} raw names -> {len(groups)} distinct at level '{args.level}'")
    if args.show:
        print()
        for norm, members in sorted(groups.items(), key=lambda kv: -len(kv[1]))[:args.show]:
            print(f"  [{len(members):>3} raw] {norm[:150]}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({k: sorted(v) for k, v in sorted(groups.items())}, f, indent=1)
        print(f"\nwritten to {args.out}")


if __name__ == "__main__":
    main()
