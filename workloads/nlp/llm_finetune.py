from dataclasses import dataclass


@dataclass
class Hyperparams:
    model: str = "meta-llama/Llama-3.1-8B"   # meta-llama/Llama-3.1-8B | microsoft/Phi-3-mini-4k-instruct
    dataset: str = "alpaca"                    # alpaca | dolly-15k
    lora_rank: int = 16
    lora_alpha: int = 32
    batch_size: int = 4
    precision: str = "fp16"                    # fp16 | bf16
    grad_accumulation_steps: int = 4


def run(hyperparams: Hyperparams) -> dict:
    """QLoRA fine-tune an LLM and return raw metrics."""
    raise NotImplementedError


if __name__ == "__main__":
    import json
    import sys

    params = json.loads(sys.argv[1])
    metrics = run(Hyperparams(**params))
    print(json.dumps(metrics))
