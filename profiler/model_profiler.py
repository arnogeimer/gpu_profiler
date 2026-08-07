import torch
from torch.profiler import profile, ProfilerActivity


def profile_kernels(model, img_size=None, batch_size=1, device="cuda", train=False):
    """Run one pass and return the ordered CUDA kernel names — including the arch/tile
    'version' encoded in each name (e.g. 'ampere_sgemm_128x64_nn') — as dispatched on
    this GPU. train=True profiles a forward+backward step (gradient convs, BatchNorm
    training kernels, ...); otherwise a no_grad forward pass."""
    if img_size is None:
        img_size = getattr(model, "default_cfg", {}).get("input_size", (3, 224, 224))[-1]
    model = model.to(device).train(train)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)

    def run():
        if train:
            model(x).sum().backward()
        else:
            with torch.no_grad():
                model(x)

    run()  # warmup so cuDNN autotune/init kernels don't pollute the trace
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run()
        torch.cuda.synchronize()

    # demangle raw C++ symbols (e.g. '_ZN17cutlass...') so the same kernel isn't
    # split across vocab entries; already-readable names pass through unchanged.
    return [torch._C._demangle(k.name) for evt in prof.events() for k in evt.kernels]
