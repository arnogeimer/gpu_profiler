from dataclasses import dataclass


@dataclass
class Hyperparams:
    batch_size: int = 8
    sequence_length: int = 512
    precision: str = "fp16"  # fp32 | fp16


def run(hyperparams: Hyperparams) -> dict:
    """Run transformer inference benchmark and return raw metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    import json
    import sys

    params = json.loads(sys.argv[1])
    metrics = run(Hyperparams(**params))
    print(json.dumps(metrics))
