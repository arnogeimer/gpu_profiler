# probe_node

A self-contained probe collector: hardware sanity check, every probe arm, one combined frame,
published to HuggingFace. No workloads, no dataset tooling, no imports from the parent repo.

```
run.py               entrypoint — sanity check → probes → combined frame → upload
probes.py            every kernel, all three arms, in one sweep
profiler.py          the timing instrument — CUDA graph capture, and the guards around it
host_info.py         CPU/GPU identity, NVML clocks and power, and the stock-TDP table
```

## Arms

| arm | families | what it is |
| --- | --- | --- |
| `v1` | gemm, bmm, conv, attn, elementwise, pool, rnn | the frozen grid every card was screened against |
| `v2` | gemm_ext, conv1d, conv_ext | the workload shapes above v1's ceiling — 18.8% of measured step time |
| `v3` | activation, norm, dropout, reduction, loss, optimizer | the per-layer costs a step is built from that neither earlier arm measures |

`v3` in detail — 477 rows, roughly a third of the sweep:

| family | kinds | grid | rows |
| --- | --- | --- | --- |
| `activation` | relu, relu6, gelu, silu, hardswish, mish | 10-point numel ladder, 256 → 67M | 180 |
| `norm` | batchnorm2d, layernorm, groupnorm, rmsnorm | per-architecture shapes, `.train()` mode | 126 |
| `dropout` | dropout, drop_path | same ladder as activation, `p=0.1` | 60 |
| `reduction` | softmax, log_softmax, mean_last | 8 (rows × width) shapes | 72 |
| `loss` | cross_entropy | batch 16/32/64 over 16 classes | 9 |
| `optimizer` | zero_grad, sgd_momentum | 3 tensor counts × 5 total sizes | 30 |

All `v3` families are `fwd_bwd` except `optimizer`, which is neither a forward nor a backward and
is recorded as `fwd`.

Three things in `v3` are deliberate and worth not undoing:

- **`activation` is separate from `elementwise`.** The elementwise arm times sigmoid; gelu is an
  erf, silu a sigmoid-multiply, mish a softplus-tanh, hardswish pure arithmetic. Their backward
  passes diverge further still. One is not a proxy for the others.
- **`drop_path` draws one Bernoulli per sample**, broadcast over everything else, not a
  per-element mask. Aliasing it to ordinary dropout would draw a mask `numel/16` times larger and
  measure the wrong kernel. Both kinds are shaped `(16, numel//16)` so the mask granularity is the
  only difference between them.
- **`optimizer` is fp32 only**, skipped under fp16/bf16 rather than repeated beneath a different
  label — an optimizer's parameters stay fp32 whatever the autocast dtype of the step that
  produced their gradients, so a bf16 row there would name a configuration nothing runs. Adam is
  absent because the workloads use SGD with momentum.

## Adding a kernel family

Everything is local to `probes.py`:

1. **Shapes** — a constant list, plus a `_random_*_shapes()` helper if they are drawn rather
   than literal. Seed it through `_rng("<family>")`: one stream per family, so a new one cannot
   shift another's draws.
2. **Measurement** — a `_<family>_point(...)` that builds its tensors and returns `time_fn(...)`
   in ms, and a `probe_<family>(rows, dt, name)` that walks the shapes appending
   `_point_result(...)` (which is what turns a failure into a row instead of an exception).
3. **Registration** — add the probe function to `ARMS`, under the arm it belongs to.

`ARMS` is the only place an arm's membership is written down; `run_probes()` walks it, so step 3
is the whole wiring change. The frame picks the new rows up automatically.

Which arm matters. Adding a family to an **existing** arm changes what that arm's rows are, so
anything screening against a frozen reference for it has to be rebuilt. Adding a **new** arm
never can — which is exactly why v2 was added as an arm rather than folded into v1.

## Run

```bash
docker build -t arge23/gpu-probe:latest probe_node/
docker run --gpus all -e HF_TOKEN=hf_... -e HF_REPO_ID=arge23/gpu-profiling-results \
           arge23/gpu-probe:latest
```

Without `HF_TOKEN`/`HF_REPO_ID` it still runs and writes `probes_{uuid}.csv` locally, uploading
nothing. Locally, outside Docker: `python run.py`.

## What it publishes

| path | contents |
| --- | --- |
| `{gpu}/probes_{uuid}.csv` | one row per probe point, every arm, `arm` column distinguishing them |
| `{gpu}/probe_meta_{uuid}.json` | host_info plus the sweep's device block — torch/driver versions, and the clocks sampled *during* the sweep, which is where throttling shows up |

`uuid` is the card's NVML UUID (first 8 hex chars), stable across rentals, so re-renting the same
silicon overwrites rather than accumulates.

Frame columns: `gpu, uuid, arm, probe, dtype, size, direction, kind, causal, ms, oom, error`.
A row has exactly one outcome — a `ms` timing, `oom=True`, or an `error` string.

`run()` is importable and returns `(DataFrame, device_block)`, so the frame can be used directly
without going near the upload path. `probes.run_probes()` is one level below that, returning the
raw `{"device": ..., "probes": [...]}` dict.

Arms are contained: a card whose v1 sweep dies mid-way still gets its v2 attempt, and whatever
rows either produced are still returned. On a card that fails v1, the v2 rows are the only
evidence of what it could do.

## The sanity check

`host_info.check_full_power` compares the host's reported `power_limit_w` against the card's
stock TDP and rejects anything below 95% of it. A host running a sub-spec power limit measures
the host's configuration rather than the silicon, and every row it produced would be quietly
wrong rather than obviously broken. The check runs before a single kernel is timed; failing it
aborts so the platform reallocates.

Cards with no entry in `host_info.STOCK_TDP_W` cannot be verified and are allowed through with a
warning, since refusing every unknown card would exclude exactly the rare hardware worth probing.

## Compatibility note

This publishes a combined frame, not the fleet's older per-arm JSONs (`device_probe_{uuid}.json`,
`probe_ext_v2_{uuid}.json`). The parent repo's reference tooling — `build_ground_truth.py`,
`export_probe_traces.py` — reads those JSONs, so it will not ingest nodes running this folder
until it learns to read `probes_{uuid}.csv`. No information is lost; it is a flatter shape.

## Keeping in step with the parent repo

`probes.py` is `../profiler/device_probe.py` and `../profiler/probe_extension.py` merged;
`profiler.py` and `host_info.py` are copies of their namesakes. Three deliberate differences:

- **the two probe modules are one file**, with the arm carried on each row instead of by the
  module it came from. The `ARMS` registry replaces the hardcoded call lists in `run_probe()`
  and `run_extension()`, and one clock sampler now spans the whole sweep rather than one per arm.
- **imports are flat** (`from profiler import time_fn`, not `from profiler.profiler import ...`)
- **offline-only machinery is stripped** — `HEAVY_FRACTION`, `row_flops`, `_shape_work`,
  `_heaviest`, `PINNED_SIZES`, `compare_to_reference` and `kernel_time_fn`. None is reachable
  from the sweep; they exist for reference-building, probe selection and object detection, all
  of which live in the parent repo. `_shape_work_v2` in particular does a
  `from profiler import probe_extension` that cannot resolve in a flat folder, so shipping it
  would ship something silently broken.

The shape grid is verified byte-identical to the parent repo's — every shape list, timing tuple,
dtype set and the RNG seed, both arms — so rows from this node stay comparable to everything
already collected. A fix to the shared timing path in the parent repo needs porting here by hand.
