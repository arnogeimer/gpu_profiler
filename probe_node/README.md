# probe_node

A self-contained probe collector: hardware sanity check, every kernel family in one sweep, one
frame published to HuggingFace. No workloads, no dataset tooling, no imports from the parent repo.

```
run.py               entrypoint — sanity check → probes → frame → upload
probes.py            every kernel family, in one sweep
profiler.py          the timing instrument — CUDA graph capture, and the guards around it
host_info.py         CPU/GPU identity, NVML clocks and power, and the stock-TDP table
```

## Families

Sixteen, swept in **fp32 only**, identified by `(probe, dtype, size, direction, kind)`.
**1,369 rows per card.**

| group | families | what it is | rows |
| --- | --- | --- | --- |
| core | gemm, bmm, conv, attn, elementwise, pool, rnn | the shapes a step is mostly made of | 1,063 |
| large shape | gemm_ext, conv_ext, conv1d | where the core grid stops but the workloads do not — 18.8% of measured step time sits above it | 127 |
| per layer | activation, norm, dropout, reduction | the costs between the matmuls | 146 |
| per step | loss, optimizer | once per step rather than once per layer | 33 |

Per-layer and per-step detail:

| family | kinds | grid |
| --- | --- | --- |
| `activation` | relu, relu6, gelu, silu, hardswish, mish | 10-point numel ladder, 256 → 67M |
| `norm` | batchnorm2d, layernorm, groupnorm, rmsnorm | per-architecture shapes, `.train()` mode |
| `dropout` | dropout, drop_path | same ladder as activation, `p=0.1` |
| `reduction` | softmax, log_softmax, mean_last | 8 (rows × width) shapes |
| `loss` | cross_entropy | batch 16/32/64 over 16 classes |
| `optimizer` | zero_grad, sgd_momentum | 3 tensor counts × 5 total sizes |

Everything is `fwd_bwd` except the core families (which sweep both directions), the conv
families (fprop/dgrad/wgrad), and `optimizer` — which is neither a forward nor a backward and is
recorded as `fwd`.

### Why fp32 only

The mixed-precision dtypes were dropped deliberately, not for cost. On a probe they measure the
tensor cores and the cast machinery as much as the kernel itself, and which of the two dominates
moves with the shape — so an fp16 row is a blend whose mixture is not a property of the hardware.
fp32 is one path through the SM for every family here, which is what makes a row comparable
between shapes and between cards.

Widening `DTYPES` back out is a one-line change and nothing assumes a single dtype. But rows
collected under a widened `DTYPES` are not comparable with these at the same `(probe, size)` —
`dtype` is part of the row key precisely so that mixing them is visible rather than silent.

### Three details worth not undoing

- **`activation` is separate from `elementwise`.** The elementwise family times sigmoid; gelu is
  an erf, silu a sigmoid-multiply, mish a softplus-tanh, hardswish pure arithmetic. Their backward
  passes diverge further still. One is not a proxy for the others.
- **`drop_path` draws one Bernoulli per sample**, broadcast over everything else, not a
  per-element mask. Aliasing it to ordinary dropout would draw a mask `numel/16` times larger and
  measure the wrong kernel. Both kinds are shaped `(16, numel//16)` so mask granularity is the
  only difference between them.
- **`optimizer` is fp32 regardless of `DTYPES`** — an optimizer's parameters stay fp32 whatever
  the autocast dtype of the step that produced their gradients. Adam is absent because the
  workloads use SGD with momentum.

## Adding a family

Everything is local to `probes.py`:

1. **Shapes** — a constant list, plus a `_random_*_shapes()` helper if they are drawn rather than
   literal. Seed it through `_rng("<family>")`: one stream per family, so a new one cannot shift
   another's draws.
2. **Measurement** — a `_<family>_point(...)` that builds its tensors and returns `time_fn(...)`
   in ms, and a `probe_<family>(rows, dt, name)` that walks the shapes appending
   `_point_result(...)` (which is what turns a failure into a row instead of an exception).
3. **Registration** — add the probe function to `PROBES`.

`PROBES` is the only place the sweep's membership is written down; `run_probes()` walks it, so
step 3 is the whole wiring change.

Adding a family only adds rows. Changing an **existing** family's shapes, size strings or kinds
changes what its rows mean, and anything comparing against rows already collected for that family
has to be rebuilt — so prefer a new family over widening an old one.

## Run

```bash
docker build -t arge23/gpu-probe:latest probe_node/
docker run --gpus all -e HF_TOKEN=hf_... -e HF_REPO_ID=arge23/gpu-profiling-results \
           arge23/gpu-probe:latest
```

Without `HF_TOKEN`/`HF_REPO_ID` it still runs and writes `probes_{uuid}.csv` locally, uploading
nothing. Locally, outside Docker: `python run.py`.

The repo root's `Dockerfile` builds the same thing from a GitHub clone instead of the local
folder — use that to ship what is pushed, this one to iterate.

## What it publishes

| path | contents |
| --- | --- |
| `kernel_probes/{gpu}.csv` | one file per GPU **model**, one row per probe point per card that has run it |
| `kernel_probes/meta/{gpu}_{uuid}.json` | per card: host_info plus the sweep's device block — torch/driver versions, and the clocks sampled *during* the sweep, which is where throttling shows up |

One file per **model**, not per card, so several cards of the same model accumulate into one
table rather than overwriting each other. `uuid` is a column, so which card produced a row is
never lost, and a card that runs twice replaces its **own** rows rather than duplicating them —
which matters because anything taking a per-row median would otherwise weight that card twice.

If the existing file cannot be *read* — a network failure, a rate limit, a truncated download —
nothing is published at all, rather than this card's rows replacing every other card's with
nothing. A file that simply does not exist yet is a different answer and publishes normally.

Frame columns: `gpu, uuid, probe, dtype, size, direction, kind, causal, ms, oom, error`.
A row has exactly one outcome — a `ms` timing, `oom=True`, or an `error` string.

`run()` is importable and returns `(DataFrame, device_block)`, so the frame can be used directly
without going near the upload path. `probes.run_probes()` is one level below that, returning the
raw `{"device": ..., "probes": [...]}` dict.

Families are contained: one that dies mid-sweep does not cost the families after it, and the rows
it already produced are kept.

## The sanity check

`host_info.check_full_power` compares the host's reported `power_limit_w` against the card's
stock TDP and rejects anything below 95% of it. A host running a sub-spec power limit measures
the host's configuration rather than the silicon, and every row it produced would be quietly
wrong rather than obviously broken. The check runs before a single kernel is timed; failing it
aborts so the platform reallocates.

Cards with no entry in `host_info.STOCK_TDP_W` cannot be verified and are allowed through with a
warning, since refusing every unknown card would exclude exactly the rare hardware worth probing.
Most Turing cards fall in this gap.

## Keeping in step with the parent repo

`probes.py` is `../profiler/device_probe.py` and `../profiler/probe_extension.py` merged;
`profiler.py` and `host_info.py` are copies of their namesakes. Deliberate differences:

- **the two probe modules are one file**, with families registered in a flat `PROBES` tuple
  rather than split by module, and one clock sampler spanning the whole sweep.
- **imports are flat** (`from profiler import time_fn`, not `from profiler.profiler import ...`)
- **fp32 only**, where the parent repo sweeps three dtypes.
- **offline-only machinery is stripped** — `HEAVY_FRACTION`, `row_flops`, `_shape_work`,
  `_heaviest`, `PINNED_SIZES`, `compare_to_reference` and `kernel_time_fn`. None is reachable
  from the sweep; they exist for reference-building, probe selection and object detection, all of
  which live in the parent repo. `_shape_work_v2` in particular does a
  `from profiler import probe_extension` that cannot resolve in a flat folder, so shipping it
  would ship something silently broken.

The shape grid is verified byte-identical to the parent repo's — every shape list, timing tuple
and the RNG seed. A fix to the shared timing path in the parent repo needs porting here by hand.
