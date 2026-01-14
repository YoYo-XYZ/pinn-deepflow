
import torch

# ----- Implementations -----
def loop_sum(residual_fields):
    # Initialize with a tensor (avoids Python int promotion)
    residuals_field = torch.zeros_like(residual_fields[0])
    for residual_field in residual_fields:
        residuals_field += residual_field.square()
    return residuals_field.mean()

def generator_sum(residual_fields):
    return (sum(rf.square() for rf in residual_fields)).mean()

# Fused: stack all, square once, sum over dim=0, then mean
def stacked_sum1(residual_fields):
    return torch.stack(residual_fields, dim=0).pow(2).sum(dim=0).mean()

def stacked_sum2(residual_fields):
    return torch.stack(residual_fields, dim=0).square().sum(dim=0).mean()

def stacked_sum3(residual_fields):
    return torch.stack(residual_fields, dim=0).square().sum(dim=0).mean()

# Specialized for exactly 3 tensors: square per tensor + add (no stack)
def specialized3(residual_fields):
    a, b, c = residual_fields
    return (a.square() + b.square() + c.square()).mean()

# Optional in-place variant for 3 tensors (mutates inputs!)
def specialized3_inplace(residual_fields):
    a, b, c = residual_fields
    a.square_(); b.square_(); c.square_()
    return (a + b + c).mean()

# ----- Benchmark utility with CUDA events -----
def bench(fn, residual_fields, iters=100000, label="fn"):
    assert residual_fields[0].is_cuda, "Use CUDA device for GPU timing."

    # Warmup
    for _ in range(10):
        _ = fn(residual_fields)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        out = fn(residual_fields)
        # lightly touch the result to keep the computation (no host transfer)
        _ = out * 1.0
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event) / iters
    print(f"{label}: {elapsed_ms:.4f} ms/iter")

# ----- Prepare test data -----
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(69)

shape = (3000,)     # 1D size
residual_fields = [torch.randn(shape, device=device) for _ in range(3)]

# Ensure on CUDA for fair comparison
if device != "cuda":
    print("CUDA not available; results will reflect CPU timings.")
else:
    bench(loop_sum, residual_fields, label="loop_sum")
    bench(generator_sum, residual_fields, label="generator_sum")
    bench(stacked_sum1, residual_fields, label="stacked_sum1 (pow)")
    bench(stacked_sum2, residual_fields, label="stacked_sum2 (square)")
    bench(stacked_sum3, residual_fields, label="stacked_sum3 (compute then mean)")
    bench(specialized3, residual_fields, label="specialized3 (no stack)")
    # Uncomment if mutation is allowed:
    # bench(specialized3_inplace, residual_fields, label="specialized3_inplace (mutates)")
