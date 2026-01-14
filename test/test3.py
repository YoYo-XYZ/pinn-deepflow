
import torch, torch.nn as nn
from torch.utils import benchmark

# Example model with some elementwise ops for fusion
class M(nn.Module):
    def __init__(self, d=1024):
        super().__init__()
        self.lin1 = nn.Linear(d, d)
        self.lin2 = nn.Linear(d, d)
    def forward(self, x):
        x = self.lin1(x)
        x = torch.relu(x)
        # chain of elementwise ops that may fuse
        x = x * 1.2345 + 0.9876
        return self.lin2(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True if device.type == "cuda" else False

m = M().to(device).eval()
x = torch.randn(2048, 1024, device=device)

# Baseline eager
with torch.inference_mode():
    for _ in range(10): m(x)  # warmup
t_eager = benchmark.Timer(stmt="m(x)",
                          globals={"m": m, "x": x}).blocked_autorange(min_run_time=2.0)

# TorchScript + freeze
scripted = torch.jit.script(m)
scripted = torch.jit.freeze(scripted)
with torch.inference_mode():
    for _ in range(10): scripted(x)
t_script = benchmark.Timer(stmt="scripted(x)",
                           globals={"scripted": scripted, "x": x}).blocked_autorange(min_run_time=2.0)

# Mixed precision (GPU example)
t_amp = None
if device.type == "cuda":
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        for _ in range(10): m(x)
    t_amp = benchmark.Timer(
        stmt="m(x)",
        globals={"m": m, "x": x,
                 "torch": torch},
        setup="from __main__ import m, x, torch\n"
              "from contextlib import nullcontext\n"
              "amp = torch.autocast(device_type='cuda', dtype=torch.float16)\n"
    ).blocked_autorange(min_run_time=2.0)

print("Eager:\n", t_eager)
print("TorchScript:\n", t_script)
if t_amp: print("Eager + autocast fp16:\n", t_amp)
