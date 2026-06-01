from dataclasses import dataclass


@dataclass
class Hyperparams:
    model: str = "runwayml/stable-diffusion-v1-5"  # runwayml/stable-diffusion-v1-5 | stabilityai/sdxl-turbo | black-forest-labs/FLUX.1-schnell
    num_inference_steps: int = 20
    image_size: int = 512                           # 512 for SD1.5, 1024 for SDXL/FLUX
    batch_size: int = 1
    precision: str = "fp16"                         # fp32 | fp16


def run(hyperparams: Hyperparams) -> dict:
    """Run diffusion model inference and return raw metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    import json
    import sys

    params = json.loads(sys.argv[1])
    metrics = run(Hyperparams(**params))
    print(json.dumps(metrics))
