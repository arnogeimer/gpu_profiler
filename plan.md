# Cross-Hardware Runtime Predictor — Pipeline Plan

## Goal

Predict the **runtime** of an ML workload on a **target GPU it was never profiled on**, given a
kernel trace collected on some **source GPU we do have**. Profile once on cheap/available
hardware → predict on expensive/unavailable hardware (capacity planning, hardware selection).

Memory is **out of scope** for this model — we approximate peak memory statically with
`torchinfo.summary` (mult-adds / activation sizes), so the transformer only targets runtime.

## Core framing: cross-hardware transfer

The kernel *sequence* dispatched for a workload differs by GPU (arch/tile/version baked into
each kernel name, e.g. `sm86_xmma_wgrad_...` vs `sm80_...`). That GPU-dependence is the signal.

A training example is:

```
INPUT:
  - kernel trace profiled on SOURCE gpu A     (the workload fingerprint)
  - source GPU A hardware features            (so the encoder can read A's kernel names)
  - target GPU B hardware features            (what we're predicting for)
TARGET:
  - measured runtime of the same workload on GPU B  (log-space)
```

**Data requirement:** the same `(model, img_size, batch_size, precision)` must be profiled on
multiple GPUs. If a config is profiled on *K* GPUs, every ordered pair (A→B) is a training
example → up to *K²* pairs per config (include A→A as a "reproduce your own runtime" anchor).
This is what turns tens of thousands of traces into hundreds of thousands of training pairs.

## Architecture (factored)

```
source trace + source-GPU emb ──► Transformer encoder ──► z   (workload embedding)
                                                          │
                                   target-GPU features ──►│
                                                          ▼
                                              cost head (MLP) ──► log runtime on target
```

- **`z` = workload embedding, ideally hardware-*invariant*.** Same workload profiled on A100 vs
  V100 should map to the same `z` — "the workload is the workload."
- **Cost head = `z × target-GPU features → runtime`.** This is a learned analytical cost model;
  all GPU-dependence of the *prediction* lives here, in the head, not in `z`.

This factoring resolves the contrastive trap: we want invariance to the **source** GPU (where we
happened to profile) but dependence on the **target** GPU (what we predict for).

## Losses

- **Main:** Huber (or MSE) on **log runtime**, conditioned on target-GPU features. Log-space
  because configs span orders of magnitude (bs=16/img=32 vs bs=64/img=224).
- **Consistency (optional, recommended):** pull `z(workload, profiled on A)` and
  `z(workload, profiled on B)` together — same workload, different source GPU = positive pair.
  Trains the source-hardware fingerprint *out* of `z`. This is the correctly-specified version
  of the contrastive idea (invariance to source, not to target).

## Pipeline stages

### 1. Trace collection — `model_profiler.py` (DONE)
`profile_kernels(model, img_size, batch_size, device, train)` returns the ordered, demangled
CUDA kernel sequence for one forward (`train=False`) or forward+backward (`train=True`) pass.
Names carry the arch/tile/version signal; `torch._C._demangle` cleans raw C++ symbols.

**Next:** a collection loop over `MODELS × IMG_SIZES × BATCH_SIZES × PRECISIONS × GPUs` that
saves, per run: kernel sequence, measured runtime (from the CUDA-event timing already in
`image_classification.py`), and the GPU id + hardware features.

### 2. Tokenization / vocab
- **Subword (BPE/WordPiece) over kernel names** — so `ampere_sgemm_256x64` and `..._128x64`
  share tokens and unseen kernels degrade gracefully.
- **Normalize template noise** — collapse giant `at::native::vectorized_elementwise_kernel<...
  lambda...>` signatures to a stable identity so lambda/anon-namespace junk doesn't blow up the
  vocab. A couple of cutlass kernels stay mangled (demangler can't parse them) — fine, they're
  still stable unique ids; regex their fingerprint if readability is ever needed.
- **Per-kernel numeric features** (optional but recommended): launch config / shapes projected
  through a small linear and summed with the token embedding — makes the embedding size-aware.

### 3. Dataset
Join traces on shared workload configs to emit `(source_trace, source_feats, target_feats) →
log_runtime` pairs. GPU hardware-feature table: one row per GPU (SM count, mem bandwidth,
FP16/FP32 peak TFLOPS, L2 size, ...).

### 4. Model + training
Transformer encoder (4–8 layers) → pooled `z` (CLS or mean-pool) → cost head.

**Trained end-to-end** (decided): encoder + cost head jointly under one runtime loss — *not* a
two-stage "pretrain the embedding, then freeze and fit the predictor" pipeline. At tens of
thousands of labeled traces the embedding is best grounded directly in runtime rather than in a
proxy objective, and the consistency loss (source-GPU invariance) rides along in the same
backward pass. No self-supervised pretraining needed at this data scale (MLM stays a cheap
optional bonus only if lots of *unlabeled* traces accumulate).

### 5. Evaluation
- Report **MAPE / relative error**, not absolute ms.
- **Held-out GPUs** (transfer to an unseen target) and **held-out models** (transfer to unseen
  architectures) as the two generalization axes that matter.

## Guardrails / gotchas

- **No target leakage:** never feed measured per-kernel *durations* into the input — total
  runtime ≈ their sum, so the model would learn a useless identity. Inputs = kernel identity +
  shapes + launch config + GPU; target = measured time.
- **Flat sequence is lossless for single-stream runtime:** PyTorch eager launches all kernels on
  one CUDA stream serially, so the profiler's order is a valid topological serialization and
  runtime = Σ(durations) is order-invariant. Modeling graph parallelism would only matter for
  peak memory or multi-stream overlap — neither is in scope (memory is handled by torchinfo).
- **Precision fairness:** fp16 vs fp32 runs share the identical `zero_grad → forward → backward →
  step` sequence; only `autocast` differs (no GradScaler), keeping the compute-time comparison
  apples-to-apples.
- **Pin the torch *wheel*, not just the version:** kernel names are produced by the cuDNN /
  cuBLAS / cutlass libraries **bundled inside the torch wheel**, not by the system CUDA toolkit.
  Install the exact same build (e.g. `torch==2.x.y+cuXXX`) on every GPU and those libraries — and
  thus the kernel vocabulary — are identical everywhere, so a kernel-name difference is
  attributable to *hardware*, not a software-version confound. The system CUDA toolkit version is
  mostly a red herring (torch ships its own runtime); only the **driver** must clear the bundled
  runtime's minimum. Record `torch.__version__`, `torch.version.cuda`,
  `torch.backends.cudnn.version()`, and driver version per trace anyway, as cheap insurance and to
  segment any GPU forced onto a different wheel (e.g. a too-new arch the pinned wheel can't
  support — keep those out of v1 or confine them to the held-out test set).
- **cuDNN autotuner nondeterminism:** with `torch.backends.cudnn.benchmark=True`, cuDNN picks
  kernels by runtime timing, so the *same* workload on the *same* GPU can yield different kernel
  sequences across runs — variance injected into the model's *input*. Set `benchmark=False` during
  dataset collection for deterministic, reproducible kernel selection (heuristic-chosen).

## Status

- [x] `model_profiler.py` — kernel sequence extraction (forward + train), demangled
- [ ] Trace collection loop (sequences + runtimes + GPU features → dataset)
- [ ] Tokenizer / vocab builder + kernel-name normalizer
- [ ] Dataset (source→target pairing) + GPU feature table
- [ ] Transformer encoder + cost head + training loop
- [ ] Evaluation (held-out GPUs, held-out models)


## Docker build commands:
Push to git first.

sudo docker build --build-arg CACHE_BUST=$(date +%s) -t arge23/gpu-profiling:parallel .
sudo docker push arge23/gpu-profiling:parallel