from dataclasses import dataclass


@dataclass
class Hyperparams:
    dataset: str = "pusht"        # pusht | cube | reacher | two_rooms
    batch_size: int = 32
    context_length: int = 8       # number of past frames fed to the predictor
    precision: str = "fp32"       # fp32 | fp16


def run(hyperparams: Hyperparams) -> dict:
    """Run LeWorldModel training or planning rollout and return raw metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    import json
    import sys

    params = json.loads(sys.argv[1])
    metrics = run(Hyperparams(**params))
    print(json.dumps(metrics))
