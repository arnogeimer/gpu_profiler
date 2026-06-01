# Workload Table

## Implemented

| Domain | Workload | Model(s) | Input size | Batch sizes | Precision | Type |
|---|---|---|---|---|---|---|
| **Computer Vision** | `classification_train` | ResNet-50, EfficientNetV2-M, ConvNeXt-Base, DeiT-B, ViT-B/16, MaxViT-T, ViT-L/16 | 32², 128², 224² (synthetic) | 32, 64, 128 | fp16, bf16, fp32 | Training + Inference |
| **Audio** | `whisper_train` | Whisper small, medium, large-v3 | 5s, 15s, 30s (synthetic, encoder always pads to 30s) | 1, 2, 4 | fp16, bf16, fp32 | Training + Inference |

**Training details:**
- Vision: SGD (momentum 0.9), `num_classes=16`, model `pretrained=False`
- Whisper: SGD (momentum 0.9), gradient checkpointing enabled, random-init weights
- Mixed precision via `torch.autocast`; fp16 also uses `GradScaler`

**Inference details:**
- Vision: forward pass on a fixed 128-image batch, `argmax` for top-1
- Whisper: autoregressive `.generate()`, greedy, forced length = `sample_length × 3` tokens

**Per-config artifacts:** every run writes `{workload_dir}/{gpu_name}_smi/{tag}_{train|infer}.json` containing nvidia-smi `max_*` metrics (gpu util, mem util, power, mem used, temperature) sampled at 100ms while the timed loop runs. OOMs are caught, recorded with `"oom": true`, and the run continues.

## Planned / stubs

| Domain | Workload | Model(s) | Type |
|---|---|---|---|
| **NLP** | `transformer_inference` | Llama 3.2 3B, Phi-3-mini, Gemma 2 2B | Inference |
| **NLP** | `llm_finetune` | Llama 3.1 8B (QLoRA), Phi-3-mini | Training |
| **Generative** | `diffusion_inference` | SD 1.5, SDXL-Turbo, FLUX.1-schnell | Inference |
| **Multimodal** | `vlm_inference` | LLaVA-1.5 7B, PaliGemma 3B | Inference |
| **Robotics** | `le_world_model` | LeWorldModel (JEPA, ~15M) | Training + Inference |
