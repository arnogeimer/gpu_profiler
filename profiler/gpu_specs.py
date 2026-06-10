"""Stock manufacturer TDPs for the GPUs in our dataset.

Used at startup to detect Salad hosts that have configured a custom (sub-spec)
power limit — measurements from those hosts under-represent the silicon's
capability and would pollute the dataset."""

# torch.cuda.get_device_name(0).replace(" ", "_") -> stock TDP (W)
STOCK_TDP_W: dict[str, float] = {
    # Turing (20xx)
    "NVIDIA_GeForce_RTX_2070":             175,
    "NVIDIA_GeForce_RTX_2070_SUPER":       215,
    "NVIDIA_GeForce_RTX_2080":             215,
    "NVIDIA_GeForce_RTX_2080_SUPER":       250,
    "NVIDIA_GeForce_RTX_2080_Ti":          250,
    # Ampere (30xx)
    "NVIDIA_GeForce_RTX_3050":             130,
    "NVIDIA_GeForce_RTX_3060":             170,
    "NVIDIA_GeForce_RTX_3060_Ti":          200,
    "NVIDIA_GeForce_RTX_3070":             220,
    "NVIDIA_GeForce_RTX_3070_Ti":          290,
    "NVIDIA_GeForce_RTX_3080":             320,
    "NVIDIA_GeForce_RTX_3080_Ti":          350,
    "NVIDIA_GeForce_RTX_3090":             350,
    "NVIDIA_GeForce_RTX_3090_Ti":          450,
    # Ada (40xx)
    "NVIDIA_GeForce_RTX_4060":             115,
    "NVIDIA_GeForce_RTX_4060_Ti":          165,
    "NVIDIA_GeForce_RTX_4070":             200,
    "NVIDIA_GeForce_RTX_4070_SUPER":       220,
    "NVIDIA_GeForce_RTX_4070_Ti":          285,
    "NVIDIA_GeForce_RTX_4070_Ti_SUPER":    285,
    "NVIDIA_GeForce_RTX_4080":             320,
    "NVIDIA_GeForce_RTX_4080_SUPER":       320,
    "NVIDIA_GeForce_RTX_4090":             450,
    # Blackwell (50xx)
    "NVIDIA_GeForce_RTX_5060":             145,
    "NVIDIA_GeForce_RTX_5060_Ti":          180,
    "NVIDIA_GeForce_RTX_5070":             250,
    "NVIDIA_GeForce_RTX_5070_Ti":          300,
    "NVIDIA_GeForce_RTX_5080":             360,
    "NVIDIA_GeForce_RTX_5090":             575,
    "NVIDIA_GeForce_RTX_5080_Laptop_GPU":  150,
    "NVIDIA_GeForce_RTX_5090_Laptop_GPU":  150,
    # Workstation
    "NVIDIA_RTX_A5000":                    230,
    "NVIDIA_RTX_A6000":                    300,
}


def check_full_power(gpu_name: str, measured_power_limit_w: float | None,
                     threshold: float = 0.95) -> tuple[bool, str]:
    """Returns (ok, reason). ok=False means this host is running below threshold of stock TDP.
    Unknown GPU models (not in STOCK_TDP_W) pass through as ok=True with a 'spec unknown' note."""
    if measured_power_limit_w is None:
        return True, "no power limit reported by NVML — cannot verify"
    spec = STOCK_TDP_W.get(gpu_name)
    if spec is None:
        return True, f"stock TDP unknown for {gpu_name} — proceeding"
    pct = measured_power_limit_w / spec
    if pct < threshold:
        return False, (f"measured power_limit_w={measured_power_limit_w:.0f}W is "
                       f"{pct*100:.1f}% of stock TDP {spec:.0f}W (threshold {threshold*100:.0f}%)")
    return True, f"power_limit_w={measured_power_limit_w:.0f}W is {pct*100:.1f}% of stock TDP {spec:.0f}W"
