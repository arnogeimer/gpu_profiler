# gpu_profiling

Per-step training and inference performance dataset across heterogeneous consumer
NVIDIA GPUs, measured on Salad community hardware.

The collected dataset lives at <https://huggingface.co/datasets/arge23/gpu-profiling-results>.

## Power-capped GPU instances

Salad nodes can configure custom power limits below the manufacturer's stock TDP
(common for hosts that want to reduce electricity cost). The cards below are
known to have been measured on at least one node running below spec — when these
appear under-performing, the host's power cap (not the silicon) is partly to blame.

Currently still measured below stock TDP — rerun until a fresh Salad host
lands at full power:

| GPU | spec TDP | measured `power_limit_w` | % of spec |
|---|---:|---:|---:|
| RTX 4070 SUPER | 220 W | 187 W | 85% |
| RTX 3060 | 170 W | 146 W | 86% |
| RTX 4080 | 320 W | 282 W | 88% |
| RTX 3050 | 130 W | 115 W | 89% |

Resolved (measurements now at or above spec):

| GPU | spec TDP | latest `power_limit_w` |
|---|---:|---:|
| RTX 3080 Ti | 350 W | 350 W (100%) ✓ |
| RTX 5060 Ti | 180 W | 180 W (100%) ✓ |

A few cards run *above* their stock TDP (host has unlocked / overclocked):

| GPU | spec TDP | measured `power_limit_w` | % of spec |
|---|---:|---:|---:|
| RTX 3090 | 350 W | 420 W | +20% |
| RTX 3080 | 320 W | 370 W | +16% |
| RTX 4070 | 200 W | 215 W | +7.5% |

## Issues encountered

**Hosts may configure GPUs to run below their stock TDP.** This biases
throughput measurements — a card capped at 75% of TDP looks weaker than the
silicon actually is. Mitigation: after each sweep we compare the NVML-reported
`power_limit_w` to the manufacturer's spec TDP. If a card was measured below
spec, we rerun on a fresh Salad instance until we land a host running at full
power, then overwrite the under-spec measurements.

**Salad's GPU label may not match the actual hardware.** Salad assigns nodes by
their advertised GPU model, but the physical card returned by
`torch.cuda.get_device_name()` (which we use as the dataset key) can disagree —
e.g. a node listed as "RTX 4070" can return "RTX 4070 SUPER", or a 3050 8GB and
a 3050 6GB Low-Profile both report as "RTX 3050". Two physically different
cards may collide under the same dataset folder. We use the nvml-reported name
as the source of truth, so the dataset reflects the silicon that ran the
workload — but you cannot map a Salad listing 1:1 to a folder name.

**Out-of-memory failures aren't always actual `OutOfMemoryError`s.** PyTorch +
cuDNN actively avoid OOM whenever a slower path exists, which produces fuzzy
boundaries instead of clean failure modes:

- **Clean `OutOfMemoryError`** happens when the request misses by a lot — a
  single tensor allocation exceeds VRAM, model parameters alone don't fit, or
  allocator fragmentation prevents finding contiguous space.
- **"Thrashing" (no error, just very slow training)** happens when the model
  *almost* fits with no headroom: cuDNN's fastest conv algorithm needs more
  workspace than available, so it silently falls back to a slower algorithm that
  fits. Training completes but at 10–100× slower than on a card with enough
  VRAM.

Example from our data: `coatnet_1_rw_224` at fp32 / bs=64 / 224×224 on RTX 4080
(16 GB) takes ~8 s/batch vs ~150 ms/batch on RTX 5090 — that's a ~53× slowdown
without any error, because cuDNN gave up the fastest conv algorithm to free
workspace memory.

The dataset stores `oom=True` only for the cleanly-failed runs; thrashing rows
are kept as honest "this is what happens when your VRAM is tight" measurements,
and we identify them post-hoc as rows whose `avg_time_ms` exceeds 10× the
fastest GPU's time for the same (model, img_size, batch_size, precision)
config.


