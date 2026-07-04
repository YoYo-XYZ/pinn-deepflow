#!/usr/bin/env python3
"""
Numerical equivalence test: batched loss vs per-geometry loss.

Verifies that the new ``_batched_loss`` path (one forward pass per
physics_type) produces the same loss values as the old per-geometry
``calc_loss`` loop (one forward pass per geometry).
"""
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "src"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import deepflow as df

df.manual_seed(69)

# ---------------------------------------------------------------------------
# Channel-flow setup (mirrors benchmark_deepflow.py / common_config.py)
# ---------------------------------------------------------------------------
Lx, Ly = 5.0, 1.0
rect = df.geometry.rectangle([0, Lx], [0, Ly])
domain = df.domain(rect)

domain.bound_list[0].define_bc({"u": 1, "v": 0})    # inflow
domain.bound_list[1].define_bc({"u": 0, "v": 0})    # wall
domain.bound_list[2].define_bc({"p": 0})             # outflow
domain.bound_list[3].define_bc({"u": 0, "v": 0})    # wall
domain.area_list[0].define_pde(
    df.pde.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000)
)

domain.sampling_random([100, 500, 100, 500], [2000])

model = df.PINN(
    width=32, length=4,
    input_vars=["x", "y"],
    output_vars=["u", "v", "p"],
)
model = model.to(df.get_device())

# ---------------------------------------------------------------------------
# Old per-geometry loss (one forward pass per geometry)
# ---------------------------------------------------------------------------
old_loss_dict = {"pde_loss": 0.0, "bc_loss": 0.0, "ic_loss": 0.0}
for geometry in domain:
    old_loss_dict[f'{geometry.physics_type.lower()}_loss'] += geometry.calc_loss(model)
old_loss_dict["total_loss"] = sum(
    v for k, v in old_loss_dict.items() if k != "total_loss"
)

# ---------------------------------------------------------------------------
# New batched loss (one forward pass per physics_type)
# ---------------------------------------------------------------------------
new_loss_dict = df.calc_loss_simple(domain)(model)

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
def _val(v):
    return v.item() if isinstance(v, torch.Tensor) else v

print("Old (per-geometry):", {k: _val(v) for k, v in old_loss_dict.items()})
print("New (batched):     ", {k: _val(v) for k, v in new_loss_dict.items()})
print()

all_ok = True
for key in old_loss_dict:
    old_val = _val(old_loss_dict[key])
    new_val = _val(new_loss_dict[key])
    diff = abs(old_val - new_val)
    ok = diff < 1e-5
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {key:12s}: old={old_val:.10f}  new={new_val:.10f}  diff={diff:.2e}")
    if not ok:
        all_ok = False

print()
if all_ok:
    print("All losses match — batched path is numerically equivalent.")
    sys.exit(0)
else:
    print("MISMATCH detected — batched path differs from per-geometry path.")
    sys.exit(1)
