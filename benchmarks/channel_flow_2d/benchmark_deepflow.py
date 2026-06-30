#!/usr/bin/env python3
"""
Benchmark: 2D steady channel flow using DeepFlow.

Matches the setup in ``static/quickstart/code.ipynb``.

Results are saved to ``results/deepflow_results.npz``.
"""

import os
import sys
import time

# ---------------------------------------------------------------------------
# Ensure imports resolve regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Ensure we can resolve the ``deepflow`` package from the project root.
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "src"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

import deepflow as df

from common_config import (
    Lx,
    Ly,
    WIDTH,
    DEPTH,
    LR,
    EPOCHS,
    BOUNDARY_POINTS,
    INTERIOR_POINTS,
    EVAL_GRID,
    SEED,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
df.manual_seed(SEED)

# ---------------------------------------------------------------------------
# Results directory
# ---------------------------------------------------------------------------
_results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_results_dir, exist_ok=True)

print("=" * 60)
print("DeepFlow  -  2D Steady Channel Flow Benchmark")
print("=" * 60)

# ===========================================================================
# 1. Geometry
# ===========================================================================
rect = df.geometry.rectangle([0, Lx], [0, Ly])
domain = df.domain(rect)

# polygon(p1, p2, p3, p4) creates boundaries in this order:
#   bound_list[0]  left   (inflow:  x=0, y∈[0,1])
#   bound_list[1]  bottom (wall:    y=0, x∈[0,5])
#   bound_list[2]  right  (outflow: x=5, y∈[0,1])
#   bound_list[3]  top    (wall:    y=1, x∈[0,5])

# ===========================================================================
# 2. Boundary Conditions
# ===========================================================================
domain.bound_list[0].define_bc({"u": 1, "v": 0})   # inflow:  u=1, v=0
domain.bound_list[1].define_bc({"u": 0, "v": 0})   # wall:    u=0, v=0 (no-slip)
domain.bound_list[2].define_bc({"p": 0})            # outflow: p=0
domain.bound_list[3].define_bc({"u": 0, "v": 0})   # wall:    u=0, v=0 (no-slip)

# ===========================================================================
# 3. PDE – Navier-Stokes (non-dimensional, Re = 100)
#
#    The DeepFlow NavierStokes PDE uses the formulation:
#        continuity:         ∂u/∂x + ∂v/∂y = 0
#        x-momentum:   u ∂u/∂x + v ∂u/∂y + ∂p/∂x − (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
#        y-momentum:   u ∂v/∂x + v ∂v/∂y + ∂p/∂y − (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
#
#    The parameters U=0.0001, L=1, mu=0.001, rho=1000 give Re = 100.
# ===========================================================================
domain.area_list[0].define_pde(
    df.pde.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000)
)

# ===========================================================================
# 4. Training-point sampling
# ===========================================================================
# To keep the comparison apples-to-apples, DeepFlow uses the same *total*
# boundary count as DeepXDE, but distributes it proportionally to side length
# (this is how DeepXDE's Rectangle.random_boundary_points works).
# For a [0,5] x [0,1] rectangle with perimeter 12, that gives:
#   left/right (length 1 each)  -> 1200 * 1/12 = 100 points each
#   bottom/top (length 5 each)  -> 1200 * 5/12 = 500 points each
perimeter = 2 * (Lx + Ly)
total_boundary = sum(BOUNDARY_POINTS)
BOUNDARY_POINTS_DEEPFLOW = [
    int(total_boundary * Ly / perimeter),   # left
    int(total_boundary * Lx / perimeter),   # bottom
    int(total_boundary * Ly / perimeter),   # right
    int(total_boundary * Lx / perimeter),   # top
]

# Use random sampling (DeepXDE default for PDE training points is also random,
# i.e. train_distribution="Hammersley" is a quasi-random sequence). Random is
# closer to DeepXDE's default behaviour.
domain.sampling_random(BOUNDARY_POINTS_DEEPFLOW, [INTERIOR_POINTS])

# ===========================================================================
# 5. Network
# ===========================================================================
model0 = df.PINN(
    width=WIDTH, length=DEPTH,
    input_vars=["x", "y"],
    output_vars=["u", "v", "p"],
)

print(f"\nNetwork        : {WIDTH}x{DEPTH}  Tanh  ->  (u, v, p)")
print(f"Sampling       : BOUNDARY_POINTS={BOUNDARY_POINTS_DEEPFLOW} (total={sum(BOUNDARY_POINTS_DEEPFLOW)}), INTERIOR={INTERIOR_POINTS}")
print(f"Training       : Adam, lr={LR}, {EPOCHS} epochs\n")

# ===========================================================================
# 6. Train
# ===========================================================================
t_start = time.perf_counter()
model1, model1_best = model0.train_adam(
    calc_loss=df.calc_loss_simple(domain),
    learning_rate=LR,
    epochs=EPOCHS,
)
train_time_s = time.perf_counter() - t_start

print(f"\nTraining time  : {train_time_s:.2f} s")

# ===========================================================================
# 7. Evaluate on a uniform grid
# ===========================================================================
print("Evaluating on uniform grid ...")
prediction = domain.area_list[0].evaluate(model1_best)
prediction.sampling_area(EVAL_GRID)  # [500, 100]

data = prediction.data_dict

# Final loss values
loss_history = data.get("loss_history", {})
_final_total = float(data["total_loss"][-1]) if data.get("total_loss") is not None else float("nan")
_final_bc    = float(data["bc_loss"][-1]) if data.get("bc_loss") is not None else float("nan")
_final_pde   = float(data["pde_loss"][-1]) if data.get("pde_loss") is not None else float("nan")

print(f"Final loss     : total={_final_total:.6f}  bc={_final_bc:.6f}  pde={_final_pde:.6f}")

# ===========================================================================
# 8. Save to NPZ
# ===========================================================================
np.savez(
    os.path.join(_results_dir, "deepflow_results.npz"),
    # Fields on evaluation grid
    x=np.asarray(data.get("x", [])),
    y=np.asarray(data.get("y", [])),
    u=np.asarray(data.get("u", [])),
    v=np.asarray(data.get("v", [])),
    p=np.asarray(data.get("p", [])),
    # PDE residuals (raw fields from PDE.compute_residuals)
    continuity_residual=np.asarray(data.get("continuity_residual", [])),
    x_momentum_residual=np.asarray(data.get("x_momentum_residual", [])),
    y_momentum_residual=np.asarray(data.get("y_momentum_residual", [])),
    # Loss history
    total_loss=np.asarray(data.get("total_loss", [])),
    bc_loss=np.asarray(data.get("bc_loss", [])),
    pde_loss=np.asarray(data.get("pde_loss", [])),
    # Summary metrics
    train_time_s=np.float64(train_time_s),
    final_total_loss=np.float64(_final_total),
    final_bc_loss=np.float64(_final_bc),
    final_pde_loss=np.float64(_final_pde),
    boundary_points=np.asarray(BOUNDARY_POINTS_DEEPFLOW),
)

print("Results saved to results/deepflow_results.npz")
print("=" * 60)
