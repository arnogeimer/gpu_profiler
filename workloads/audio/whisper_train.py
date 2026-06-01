import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import transformers
from transformers import WhisperConfig, WhisperForConditionalGeneration, GenerationConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from profiler import gpu_info
from profiler.cuda_monitor import CUDAMonitor, record_run

transformers.logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


WARMUP_BATCHES = 5
TIMED_BATCHES = 20

MEL_FRAMES = 3000         # whisper's fixed encoder input (30s of audio padded)
TOKENS_PER_SECOND = 3     # rough avg English tokens/sec, used to scale decoder sequence length

_MODELS = {
    "small":    "openai/whisper-small",
    "medium":   "openai/whisper-medium",
    "large-v3": "openai/whisper-large-v3",
}

_PRECISION = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


@dataclass
class Hyperparams:
    model: str = "small"      # small | medium | large-v3
    sample_length: int = 30   # actual speech content in seconds (encoder always pads to 30s)
    batch_size: int = 4
    precision: str = "fp16"   # fp32 | fp16 | bf16


def run(hyperparams: Hyperparams) -> dict:
    device = torch.device("cuda")

    csv_path = Path(__file__).parent / f"{gpu_info.get_gpu_name()}_metrics.csv"
    monitor = CUDAMonitor(interval_ms=20)

    config = WhisperConfig.from_pretrained(_MODELS[hyperparams.model])
    model = WhisperForConditionalGeneration(config).to(device).train()
    model.gradient_checkpointing_enable()
    model.generation_config = GenerationConfig.from_pretrained(_MODELS[hyperparams.model])

    n_mels = config.num_mel_bins
    vocab_size = config.vocab_size
    seq_tokens = hyperparams.sample_length * TOKENS_PER_SECOND

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    amp_dtype = _PRECISION[hyperparams.precision]
    use_autocast = hyperparams.precision != "fp32"
    scaler = torch.amp.GradScaler("cuda") if hyperparams.precision == "fp16" else None

    def make_batch():
        mel = torch.randn(hyperparams.batch_size, n_mels, MEL_FRAMES, device=device)
        labels = torch.randint(0, vocab_size, (hyperparams.batch_size, seq_tokens), device=device)
        return mel, labels

    def step(mel, labels):
        optimizer.zero_grad()
        if use_autocast:
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = model(input_features=mel, labels=labels).loss
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
        else:
            loss = model(input_features=mel, labels=labels).loss
            loss.backward()
            optimizer.step()

    for _ in range(WARMUP_BATCHES):
        step(*make_batch())
    torch.cuda.synchronize()

    monitor.start()
    times_ms = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            mel, labels = make_batch()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            step(mel, labels)
            end.record()
            torch.cuda.synchronize()
            times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True
        raise
    finally:
        train_metrics = monitor.stop(oom=oom)
        train_avg_ms = sum(times_ms) / len(times_ms) if times_ms else None
        record_run(csv_path, hyperparams, "train", metrics=train_metrics, avg_ms=train_avg_ms)

    model.eval()
    mel_inf = torch.randn(hyperparams.batch_size, n_mels, MEL_FRAMES, device=device)

    def infer(mel):
        with torch.no_grad():
            if use_autocast:
                with torch.autocast("cuda", dtype=amp_dtype):
                    return model.generate(
                        input_features=mel,
                        max_new_tokens=seq_tokens,
                        min_new_tokens=seq_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
            else:
                return model.generate(
                    input_features=mel,
                    max_new_tokens=seq_tokens,
                    min_new_tokens=seq_tokens,
                    do_sample=False,
                    num_beams=1,
                )

    for _ in range(WARMUP_BATCHES):
        infer(mel_inf)
    torch.cuda.synchronize()

    monitor.start()
    inf_times_ms = []
    oom = False
    try:
        for _ in range(TIMED_BATCHES):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            infer(mel_inf)
            end.record()
            torch.cuda.synchronize()
            inf_times_ms.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError:
        oom = True
        raise
    finally:
        infer_metrics = monitor.stop(oom=oom)
        inf_avg_ms = sum(inf_times_ms) / len(inf_times_ms) if inf_times_ms else None
        record_run(csv_path, hyperparams, "infer", metrics=infer_metrics, avg_ms=inf_avg_ms)

    del model, optimizer, mel_inf
    print(f"train: {train_avg_ms / hyperparams.batch_size:.3f} ms/sample | inference: {inf_avg_ms / hyperparams.batch_size:.3f} ms/sample  ({hyperparams.model} | {hyperparams.sample_length}s | bs={hyperparams.batch_size} | {hyperparams.precision})")
    return {
        "avg_batch_time_ms": train_avg_ms,
        "batch_times_ms": times_ms,
        "avg_inference_time_ms": inf_avg_ms,
        "inference_times_ms": inf_times_ms,
    }


if __name__ == "__main__":
    config_path = Path(__file__).parent / "whisper_train_config.json"
    for params in json.loads(config_path.read_text()):
        try:
            run(Hyperparams(**params))
        except torch.cuda.OutOfMemoryError:
            print(f"OOM: {params}")
        finally:
            torch.cuda.empty_cache()
