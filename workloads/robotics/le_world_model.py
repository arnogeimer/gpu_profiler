"""JEPA-style world-model training kernel (LeJEPA / le-wm shape).

The core forward/loss is lifted from https://github.com/lucas-maes/le-wm — encode the
video sequence, split into context (first ctx_len frames) and target (remaining n_preds
frames, no-grad), predict target embeddings from context + actions, L2 loss in embedding
space, EMA-update target encoder. SIGReg regularizer is omitted (it stabilizes long
training; for per-batch GPU timing it adds noise without changing the shape of compute)."""
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import torch
import torch.nn as nn
import timm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler.cuda_monitor import CUDAMonitor, build_row

warnings.filterwarnings("ignore")


WARMUP_BATCHES = 3
TIMED_BATCHES = 10

INFER_WARMUP_BATCHES = 1
INFER_TIMED_BATCHES = 5

EMA_DECAY = 0.99


@dataclass
class Hyperparams:
    encoder: str = "vit_tiny_patch16_224"   # timm model name
    img_size: int = 224
    context_length: int = 8                  # past frames used as context
    num_predictions: int = 4                 # future frames whose embeddings we predict
    batch_size: int = 4
    action_dim: int = 6                      # typical robotics action vector size
    precision: str = "fp16"                  # fp32 | fp16


class Predictor(nn.Module):
    """Small transformer: (context embeddings + per-step action embeddings) -> predicted target embeddings."""
    def __init__(self, emb_dim: int, action_dim: int, num_predictions: int, depth: int = 2, heads: int = 4):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, emb_dim)
        self.pred_tokens = nn.Parameter(torch.randn(num_predictions, emb_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, batch_first=True, dim_feedforward=emb_dim * 4)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.out_proj = nn.Linear(emb_dim, emb_dim)

    def forward(self, ctx_embs: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        bs, n_preds = target_actions.shape[:2]
        action_embs = self.action_proj(target_actions)
        pred_init = self.pred_tokens.unsqueeze(0).expand(bs, -1, -1) + action_embs
        inp = torch.cat([ctx_embs, pred_init], dim=1)
        out = self.transformer(inp)
        return self.out_proj(out[:, -n_preds:])


def _encode(model: nn.Module, frames: torch.Tensor) -> torch.Tensor:
    """Per-frame encoding -> single embedding vector per frame. Pools whatever
    forward_features returns (works for ViTs giving [B,N,D] and CNNs giving [B,C,H,W])."""
    feats = model.forward_features(frames)
    if feats.ndim == 4:
        return feats.flatten(2).mean(-1)
    if feats.ndim == 3:
        return feats.mean(dim=1)
    return feats


@torch.no_grad()
def _ema_update(target: nn.Module, online: nn.Module, decay: float = EMA_DECAY) -> None:
    for tp, op in zip(target.parameters(), online.parameters()):
        tp.data.mul_(decay).add_(op.data, alpha=1 - decay)


def run(hyperparams: Hyperparams) -> list[dict]:
    """Run one config: JEPA training (train phase) + single-shot embedding rollout (infer phase)."""
    rows: list[dict] = []
    device = torch.device("cuda")
    monitor = CUDAMonitor(interval_ms=10)
    seq_len = hyperparams.context_length + hyperparams.num_predictions

    try:
        try:
            online_encoder = timm.create_model(hyperparams.encoder, pretrained=False, num_classes=0,
                                               img_size=hyperparams.img_size)
        except TypeError:
            online_encoder = timm.create_model(hyperparams.encoder, pretrained=False, num_classes=0)
        with torch.no_grad():
            probe = torch.zeros(1, 3, hyperparams.img_size, hyperparams.img_size)
            emb_dim = _encode(online_encoder, probe).shape[-1]

        try:
            target_encoder = timm.create_model(hyperparams.encoder, pretrained=False, num_classes=0,
                                               img_size=hyperparams.img_size)
        except TypeError:
            target_encoder = timm.create_model(hyperparams.encoder, pretrained=False, num_classes=0)
        target_encoder.load_state_dict(online_encoder.state_dict())
        for p in target_encoder.parameters():
            p.requires_grad_(False)

        predictor = Predictor(emb_dim=emb_dim, action_dim=hyperparams.action_dim,
                              num_predictions=hyperparams.num_predictions)
    except Exception as e:
        rows.append(build_row(hyperparams, "setup", error=f"model_creation_failed: {e}"))
        return rows

    online_encoder = online_encoder.to(device).train()
    target_encoder = target_encoder.to(device).eval()
    predictor = predictor.to(device).train()

    optimizer = torch.optim.AdamW(
        list(online_encoder.parameters()) + list(predictor.parameters()),
        lr=1e-4,
    )
    use_fp16 = hyperparams.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda") if use_fp16 else None

    bs = hyperparams.batch_size
    ctx_len = hyperparams.context_length
    n_preds = hyperparams.num_predictions

    def make_batch():
        frames = torch.randn(bs, seq_len, 3, hyperparams.img_size, hyperparams.img_size, device=device)
        actions = torch.randn(bs, n_preds, hyperparams.action_dim, device=device)
        return frames, actions

    def jepa_forward(frames, actions):
        flat = frames.view(bs * seq_len, 3, hyperparams.img_size, hyperparams.img_size)
        online_emb = _encode(online_encoder, flat).view(bs, seq_len, -1)
        ctx_emb = online_emb[:, :ctx_len]
        with torch.no_grad():
            target_emb = _encode(target_encoder, flat).view(bs, seq_len, -1)[:, ctx_len:]
        pred_emb = predictor(ctx_emb, actions)
        return (pred_emb - target_emb).pow(2).mean()

    def step(frames, actions):
        optimizer.zero_grad()
        if use_fp16:
            with torch.autocast("cuda", dtype=torch.float16):
                loss = jepa_forward(frames, actions)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss = jepa_forward(frames, actions)
            loss.backward()
            optimizer.step()
        _ema_update(target_encoder, online_encoder)

    # === Train warmup ===
    try:
        for _ in range(WARMUP_BATCHES):
            step(*make_batch())
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, "train", error=f"forward_failed: {e}"))
        return rows

    # === Train timed ===
    monitor.start()
    times_ms = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            frames, actions = make_batch()
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            step(frames, actions)
            end_evt.record()
            torch.cuda.synchronize()
            times_ms.append(start_evt.elapsed_time(end_evt))
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        train_metrics = monitor.stop(oom=oom)
        train_avg_ms = sum(times_ms) / len(times_ms) if times_ms else None
        rows.append(build_row(hyperparams, "train", metrics=train_metrics, avg_ms=train_avg_ms))
    torch.cuda.empty_cache()

    # === Inference (single-shot rollout: predict future embeddings from context) ===
    online_encoder.eval()
    predictor.eval()

    def rollout():
        with torch.no_grad():
            frames, actions = make_batch()
            flat = frames[:, :ctx_len].reshape(bs * ctx_len, 3, hyperparams.img_size, hyperparams.img_size)
            ctx_emb = _encode(online_encoder, flat).view(bs, ctx_len, -1)
            if use_fp16:
                with torch.autocast("cuda", dtype=torch.float16):
                    return predictor(ctx_emb, actions)
            return predictor(ctx_emb, actions)

    try:
        for _ in range(INFER_WARMUP_BATCHES):
            rollout()
        torch.cuda.synchronize()
    except RuntimeError as e:
        rows.append(build_row(hyperparams, "infer", error=f"forward_failed: {e}"))
        return rows

    monitor.start()
    inf_times_ms = []
    oom = False
    try:
        for _ in range(INFER_TIMED_BATCHES):
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
            rollout()
            end_evt.record()
            torch.cuda.synchronize()
            inf_times_ms.append(start_evt.elapsed_time(end_evt))
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        infer_metrics = monitor.stop(oom=oom)
        inf_avg_ms = sum(inf_times_ms) / len(inf_times_ms) if inf_times_ms else None
        rows.append(build_row(hyperparams, "infer", metrics=infer_metrics, avg_ms=inf_avg_ms))

    del online_encoder, target_encoder, predictor, optimizer
    def _fmt(avg_ms, batch):
        return "OOM" if avg_ms is None else f"{avg_ms / batch:.2f} ms/sample"
    print(f"train: {_fmt(train_avg_ms, bs)} | inference: {_fmt(inf_avg_ms, bs)}  ({hyperparams.encoder} | {hyperparams.img_size}x{hyperparams.img_size} | T={seq_len} | bs={bs} | {hyperparams.precision})")
    return rows


# A ViT-flavored subset of image_classification's MODELS for cross-workload comparability.
ENCODERS = [
    'vit_tiny_patch16_224',
    'vit_small_patch16_224',
    'deit_tiny_patch16_224',
    'deit_small_patch16_224',
    'efficientformer_l1',
    'twins_pcpvt_small',
    'xcit_tiny_12_p16_224',
]
IMG_SIZES = [224]
CONTEXT_LENGTHS = [8]
NUM_PREDICTIONS = [4]
BATCH_SIZES = [2, 4]
ACTION_DIMS = [6]
PRECISIONS = ['fp32', 'fp16']


UPLOAD_EVERY_N_MODELS = 5


def run_all(upload_fn: Optional[Callable[[pd.DataFrame], None]] = None,
            only_models: Optional[set] = None) -> pd.DataFrame:
    """Iterate every config; upload after every N encoders and once at the end."""
    all_rows: list[dict] = []
    selected = [m for m in ENCODERS if only_models is None or m in only_models]
    for i, encoder in enumerate(selected):
        for img_size, ctx_len, n_preds, bs, act_dim, precision in product(
            IMG_SIZES, CONTEXT_LENGTHS, NUM_PREDICTIONS, BATCH_SIZES, ACTION_DIMS, PRECISIONS,
        ):
            params = {
                "encoder": encoder,
                "img_size": img_size,
                "context_length": ctx_len,
                "num_predictions": n_preds,
                "batch_size": bs,
                "action_dim": act_dim,
                "precision": precision,
            }
            try:
                all_rows.extend(run(Hyperparams(**params)))
            except Exception as e:
                print(f"FAIL ({type(e).__name__}): {params} - {e}")
                all_rows.append({**params, "phase": "config_failed", "error": f"{type(e).__name__}: {e}"})
            finally:
                torch.cuda.empty_cache()

        is_milestone = (i + 1) % UPLOAD_EVERY_N_MODELS == 0
        is_last = i == len(selected) - 1
        if upload_fn and (is_milestone or is_last):
            try:
                upload_fn(pd.DataFrame(all_rows))
            except Exception as e:
                print(f"  upload after {encoder} failed: {e}")

    return pd.DataFrame(all_rows)
