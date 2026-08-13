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

We originally kept thrashing rows as honest "this is what happens when your VRAM
is tight" measurements, identified post-hoc as rows exceeding 10× the fastest
GPU's time for the same config. **That was the wrong call, and the first three
workload nodes showed why.**

Most of the slowdown was not cuDNN choosing a slower algorithm. It was the
caching allocator itself: with `set_per_process_memory_fraction` left at its
default of `1.0`, torch never raises — on the last allocation it flushes its
cache and retries `cudaMalloc` against a full device, over and over, and the
config still "succeeds" while reporting a time that is mostly allocator churn.
The tell is that the timing tracks memory rather than shape:

| wav2vec2-large, bs=4, fp16, RTX 3090 | 2 s | 4 s | 8 s | 16 s | 30 s |
|---|---:|---:|---:|---:|---:|
| step time (ms) | 51 | 84 | 148 | 326 | **21 766** |
| peak device memory (%) | 44 | 51 | 67 | 96 | **100** |

16 s at 96% is on trend; 30 s at 100% is 66× above it. wav2vec2-**base** at 30 s
uses 55% of memory and lands exactly on trend — same family, same config, no
wall. Across the three nodes, **14% of audio rows sat at 100% and consumed 94%
of the wall time — 8.4 hours of 8.9**.

Worse, the effect is a function of VRAM, so it corrupts precisely the
cross-hardware comparison the dataset exists for. The same config records
~21 800 ms on a 24 GB card, would run at roughly 600 ms on a 32 GB one, and OOMs
on 8 GB. Image classification had it too, which is why a 16 GB RTX 5070 Ti spent
**5.04 h** against a 24 GB RTX 3090's **1.98 h** on identical work — left alone,
that enters the dataset as "Blackwell is slower than Ampere".

Mitigation: `main.MEMORY_FRACTION = 0.995` caps the allocator just below the
physical limit, so a config that does not fit raises instead of grinding. 0.995
rather than lower because the boundary is sharp — 96% still measured cleanly, so
a tighter cap would discard valid rows near the top of the range rather than only
the bad ones. The 315 pre-fix rows at ≥99.9% memory were deleted from the dataset
rather than kept.

**A capped allocator exposed a second bug: OOM arrives as two different
exceptions.** Both workloads caught `torch.cuda.OutOfMemoryError`, which is what
torch's own allocator raises. But when the cap sits close to the physical limit,
the *driver* can get there first, and an allocation made during CUDA graph
capture then comes back as a plain `RuntimeError` carrying
`cudaErrorMemoryAllocation`. Same event, different type. Catching only the first
left rows with no timing and `oom=False` — indistinguishable in the CSV from a
code fault. `profiler.cuda_monitor.is_oom` now covers both, and after the fix the
slowest single config anywhere is 6.5 s rather than 20+ minutes, with zero
unlabelled failures.

**A residue survives the cap, and `max_memory_used_pct` cannot find it.** Capping
the allocator converts the clear misses into `oom` rows, but a config that lands
*just* inside the budget still completes — with less workspace than cuDNN wants,
so it picks a slower algorithm and returns a plausible-looking number that
measures the fallback rather than the model. This is the original "thrashing"
mechanism, now confined to a narrow band instead of dominating the sweep: across
381 configs re-run after the fix, exactly one regressed.

Peak memory does not identify these. An RTX 3090 Ti ran `vit_large` 224/32/fp32
at 99.3% of VRAM and returned a clean 786 ms; healthy rows routinely read 100%,
because the figure is a whole-device measure that includes the CUDA context
while the cap applies only to torch's own allocations.

**The fp32/fp16 ratio does identify them.** Low precision is faster by a stable
factor across the fleet, so a config where fp32 alone explodes — while its fp16
and bf16 siblings stay normal and agree with each other — is memory-constrained
rather than slow:

| gpu | config | fp32 | fp16 | ratio |
|---|---|---:|---:|---:|
| RTX 4080 SUPER | convnext_large 224/32 | 13 032 ms | 138 ms | **94.6×** |
| RTX 5070 Ti | vit_large 224/32 | 9 826 ms | 189 ms | **52.1×** |
| RTX 5070 Ti | regnety_080 224/64 | 2 080 ms | 120 ms | **17.3×** |
| RTX 3090 Ti | vit_large 224/64 | 7 468 ms | 463 ms | **16.1×** |

Over 692 (gpu, model, img_size, batch_size) groups the ratio is 1.68 median,
2.60 at p90, 6.24 at p99 — and then a gap to 16.1. Four groups sit above it,
0.6% of the total, every one a 224 px large-batch fp32 config at the corner of
its card's capacity. Note the 3090 Ti case: with 24 GB it hits the wall one batch
size later than the 16 GB cards rather than avoiding it, so this is not a
small-VRAM problem, it is a "config that just barely fits" problem and every card
has one somewhere.

Mitigation is post-hoc, since the row is only recognisable once its siblings
exist: **drop groups whose `fp32/fp16` exceeds 8**. The threshold sits in the gap
between the healthy p99 and the lowest artefact, so nothing about it is a
judgement call. Two limits — it needs both precisions present, so it cannot
classify a config whose fp16 sibling OOM'd, and it deliberately does not catch a
card that is uniformly slow at every precision. That is the device-probe screen's
job, not this one's.

**A large share of what we were timing was the host CPU, not the GPU.** Kernel
launches are dispatched from Python: every op walks the torch dispatcher into
`cudaLaunchKernel`, costing roughly 5–20 µs of *host* time regardless of where
the tensors live. Putting inputs on the GPU removes the transfer, not the
dispatch. When kernels are large the GPU hides that cost; when they are small
the GPU starves waiting for work. Measured against a CUDA-graph capture of the
same step:

| config | eager wall | GPU kernel time | host share |
|---|---:|---:|---:|
| mobilenetv3_small_100 @64 | 27.97 ms | 3.44 ms | 88% |
| resnet50 @64 | 29.20 ms | 9.14 ms | 69% |
| resnet50 @224 | 44.68 ms | 44.02 ms | 1% |

This is worse than ordinary noise for a *cross-hardware* dataset: host time does
not transfer between GPUs, and Salad allocates heterogeneous CPUs, so two nodes
with the same card could disagree for reasons nothing in the schema explained.
It also erased the axes we were sweeping — `mobilenetv3_small_100` read ~27 ms
at both 64 px and 224 px, hiding a real 2.4× difference in GPU work. Mitigation:
every step is now timed inside a captured CUDA graph (`profiler.time_fn`), so
the timed window contains no Python at all, and `host_info.json` additionally
records the node's CPU model, core counts and clock ceiling.

**`model.generate()` measured Python, not the model.** The LLM workload
originally timed 64 autoregressive decode steps through HuggingFace's generate
loop. That loop costs ~30 ms of host dispatch *per token*, which swamped the GPU
entirely: SmolLM2-135M took 1819 ms, SmolLM2-360M 1984 ms, and Qwen2.5-1.5B —
eleven times the parameters — 1741 ms, i.e. *faster*. Enabling or disabling the
KV cache changed nothing (4134 ms vs 4062 ms), and raising the batch size 32×
moved it 3%. PEFT's per-module wrappers alone doubled it. Mitigation: `generate()`
was dropped in favour of two separate measurements — a prefill forward over the
whole sequence (compute-bound) and one decode step against a prefilled
`StaticCache` (weight-bandwidth-bound), both graph-captured. Both now scale with
model size as they should.

**One workload cannot be graph-captured, so the dataset carries two timing
instruments.** Object detection fails capture in three independent places:
torchvision's `GeneralizedRCNNTransform` builds host tensors inside the forward,
the DETR-family loss constructs a criterion module and moves it to the device per
call, and YOLOS trips an RNG-offset error. Detection therefore reports GPU kernel
time by summing the profiler's per-kernel device time (`profiler.kernel_time_fn`)
rather than by replaying a capture. The two agree on the quantity but not
exactly: kernel-sum reads high by a roughly fixed **2.4–6.9 µs per kernel**,
which is ~10% of a step built from 80 µs kernels but ~78% of one built from 6 µs
kernels. Every row records `timing_method` (`cuda_graph` | `kernel_sum`) and
`kernel_count`, so the bias can be corrected or fitted rather than silently
absorbed. This still beats the alternative — eager wall-clock for detection was
5% host at 800 px/bs 8 but 62% at 320 px/bs 2.

**NVML power and utilisation are meaningless over a sub-second window.** Both
counters refresh on the driver's own cadence rather than per query, so polling
every 10 ms re-reads a cached value and averaging over the ~0.85 s a config takes
averages a handful of arbitrarily-aligned driver updates. On one fixed config
whose timing was reproducible to 0.2%, `avg_power_w` ranged 42–145 W and
`avg_gpu_util_pct` 0–68% purely with how long the card had been idle beforehand —
and utilisation moved *upward* with more preceding idle. Mitigation:
`max_power_w`, `avg_power_w`, `energy_j`, `max_gpu_util_pct` and
`avg_gpu_util_pct` were removed from the schema rather than shipped looking
meaningful. Memory is kept, because `mem_get_info` is an instantaneous query.
Trustworthy power would need a dedicated multi-second replay window (stable to
1.4% when measured that way), which would roughly double the sweep. Note that
CSVs collected before this change still contain those columns.

**Untrained detectors diverge, and detection losses raise rather than return
garbage.** We build models from `config.json` without checkpoints, since step
timing depends on architecture and not weight values. That is safe for
classification and language models — a NaN loss still dispatches identical
kernels — but set-prediction detectors validate their own box geometry before
the Hungarian match, so once weights blow up the step raises `ValueError` and the
config is lost. At `lr=0.01` an untrained detector diverges by step 2;
`conditional-detr-resnet-50` failed exactly this way, and it is also why RT-DETR
and Deformable-DETR were initially misdiagnosed as unusable. Mitigation:
detection trains at `lr=0.0`. SGD issues an identical kernel sequence at zero
learning rate (`param.add_(d_p, alpha=-0)` is not short-circuited), so the
measurement is unchanged while the weights stay finite. Detection's error handler
also catches `Exception` rather than `RuntimeError`, since these losses raise
`ValueError`.

**Models silently ignore the axis you are sweeping.** Several configurations
accept a shape parameter and then discard it, producing rows that differ only in
their label:

- `ssd300_vgg16` and `ssdlite320_mobilenet_v3_large` hardcode 300/320 in their
  transforms, ignoring `min_size`/`max_size` — all three `img_size` values gave
  the same measurement. Both were dropped from the model list.
- torchvision's other detectors rescale every input to an 800/1333 default
  unless `min_size`/`max_size` are pinned per config.
- Whisper pads every clip to 30 s unless `max_source_positions` is resized, so
  the `seconds` axis measured nothing for those models.
- `efficientformer_l1`/`l3` hardcode a 49-token attention bias and only run at
  224 px, failing 24 configs per GPU. Both were dropped.
- The image-classification inference phase used a hardcoded batch of 128 while
  the row was stamped with the config's batch size, so three identical runs were
  labelled 16/32/64.

The general lesson: after setting a shape parameter, read back what the model
actually ran at rather than trusting that it took effect.

**Roughly half the audio step was numpy running on the CPU.** wav2vec2-family
configs enable SpecAugment by default, and `_compute_mask_indices` builds its
masks in numpy on the host. Cost ranged from 1.7% of the step (`wavlm-base`) to
45.4% (`hubert-base-ls960`) — large, per-model-variable, and stochastic, since
masks are redrawn every step. Mitigation: `apply_spec_augment = False` is set on
every config before the model is built.



**bf16 batched matmul is much slower than fp16 on Blackwell, and the cost is
cuBLAS, not silicon.** Across 118 `bmm` shapes, the RTX 5090's advantage over
the 4090 collapses as precision drops — the median 4090/5090 step time is
1.538 at fp32, 1.096 at fp16 and 0.995 at bf16, where the 4090 is actually the
faster card on 60 of 118 rows. Comparing each card against *itself* on the same
shape shows where it comes from:

| card | median bf16 / fp16 | range |
|---|---|---|
| 5080 | 1.355 | 0.27 – 6.61 |
| 5090 | 1.243 | 0.27 – 6.12 |
| 4080 SUPER | 1.169 | 0.27 – 5.19 |
| 4090 | 1.041 | 0.18 – 4.19 |

Every card pays something for bf16 here, but Blackwell pays 24–36% against
Ada's 4–17%, consistently across two cards of each generation. The per-shape
range is the tell: 0.27× to 6.6× on one card between two dtypes that use the
same tensor cores is algorithm selection, not hardware. `b16m768n768k416` is
the clearest case — the 5090 takes 0.546 ms at bf16 against 0.149 ms at fp16, a
3.7× penalty, while the 4090 sits at 0.180/0.183 with none, making the 5090 3×
*slower* than a 4090 on that shape despite being far faster at fp16.

This is a real cost a rented node would pay, so it stays in the dataset rather
than being corrected out. But it is a property of the pinned
`torch==2.11.0+cu130` toolchain, not of the cards, and a future CUDA or cuBLAS
version may change or erase it. Recorded here so that a later re-run showing
different bf16 numbers reads as a library change rather than a data bug.
