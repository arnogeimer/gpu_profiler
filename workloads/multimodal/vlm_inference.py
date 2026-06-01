from dataclasses import dataclass


@dataclass
class Hyperparams:
    model: str = "llava-hf/llava-1.5-7b-hf"  # llava-hf/llava-1.5-7b-hf | google/paligemma-3b-pt-224
    dataset: str = "vqav2"                     # vqav2 | coco-captions
    batch_size: int = 4
    max_new_tokens: int = 64
    precision: str = "fp16"                    # fp32 | fp16 | int4


def run(hyperparams: Hyperparams) -> dict:
    """Run VLM inference on a VQA dataset and return raw metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    import json
    import sys

    params = json.loads(sys.argv[1])
    metrics = run(Hyperparams(**params))
    print(json.dumps(metrics))
