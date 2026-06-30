#!/usr/bin/env python3
"""
Benchmark: 2D steady channel flow using DeepXDE with PyTorch backend.

Matches the setup in ``static/quickstart/code.ipynb`` for a fair comparison
with the DeepFlow implementation.

Results are saved to ``results/deepxde_results.npz``.
"""

import os
import sys
import time
import warnings

# ---------------------------------------------------------------------------
# Ensure imports resolve regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# ---------------------------------------------------------------------------
# Set DeepXDE backend to PyTorch **before** importing deepxde
# ---------------------------------------------------------------------------
os.environ["DDE_BACKEND"] = "pytorch"

import numpy as np
import torch
import deepxde as dde

from common_config import (
    Lx,
    Ly,
    Re,
    WIDTH,
    DEPTH,
    ACTIVATION,
    LR,
    EPOCHS,
    BOUNDARY_POINTS,
    INTERIOR_POINTS,
    EVAL_GRID,
    SEED,
)

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    device_str = "cuda"
    try:
        torch.set_default_device(device_str)
    except AttributeError:
        # Older PyTorch versions: set via torch.device context
        pass
else:
    device_str = "cpu"

print("=" * 60)
print("DeepXDE (PyTorch)  -  2D Steady Channel Flow Benchmark")
print(f"Device            : {device_str}")
print("=" * 60)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
np.random.seed(SEED)
torch.manual_seed(SEED)

# ===========================================================================
# 1. Geometry
# ===========================================================================
geom = dde.geometry.Rectangle(xmin=[0, 0], xmax=[Lx, Ly])

# ===========================================================================
# 2. PDE — Non-dimensional Navier-Stokes (Re = 100)
#
#    Continuity:         ∂u/∂x + ∂v/∂y = 0
#    X-momentum:   u ∂u/∂x + v ∂u/∂y + ∂p/∂x − (1/Re)(∂²u/∂x² + ∂²u/∂y²) = 0
#    Y-momentum:   u ∂v/∂x + v ∂v/∂y + ∂p/∂y − (1/Re)(∂²v/∂x² + ∂²v/∂y²) = 0
# ===========================================================================
def pde(x, y):
    """
    Args:
        x: Input tensor of shape (N, 2) — columns are (x, y).
        y: Output tensor of shape (N, 3) — columns are (u, v, p).

    Returns:
        List of three residual tensors [continuity, x_momentum, y_momentum].
    """
    u = y[:, 0:1]
    v = y[:, 1:2]
    p = y[:, 2:3]

    # First derivatives
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)

    # Second derivatives
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    nu_inv = 1.0 / Re  # 1/Re = 0.01

    continuity    = u_x + v_y
    x_momentum    = u * u_x + v * u_y + p_x - nu_inv * (u_xx + u_yy)
    y_momentum    = u * v_x + v * v_y + p_y - nu_inv * (v_xx + v_yy)

    return [continuity, x_momentum, y_momentum]

# ===========================================================================
# 3. Boundary Conditions
# ===========================================================================
def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)

def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], Lx)

def boundary_wall(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0.0) or np.isclose(x[1], Ly))

# Inflow: u=1, v=0
bc_inlet_u = dde.icbc.DirichletBC(geom, lambda x: 1.0, boundary_left, component=0)
bc_inlet_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_left, component=1)

# Walls (no-slip): u=0, v=0
bc_wall_u = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_wall, component=0)
bc_wall_v = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_wall, component=1)

# Outflow: p=0
bc_outlet_p = dde.icbc.DirichletBC(geom, lambda x: 0.0, boundary_right, component=2)

bcs = [bc_inlet_u, bc_inlet_v, bc_wall_u, bc_wall_v, bc_outlet_p]

# ===========================================================================
# 4. PDE Data
# ===========================================================================
data = dde.data.PDE(
    geom,
    pde,
    bcs,
    num_domain=INTERIOR_POINTS,
    num_boundary=sum(BOUNDARY_POINTS),
    num_test=5000,
)

# ===========================================================================
# 5. Network — FNN with same architecture as DeepFlow
# ===========================================================================
layer_sizes = [2] + [WIDTH] * DEPTH + [3]
net = dde.nn.FNN(layer_sizes, ACTIVATION, "Glorot normal")

model = dde.Model(data, net)

# ===========================================================================
# 6. Compile & Train
# ===========================================================================
model.compile("adam", lr=LR)

print(f"\nNetwork        : {WIDTH}x{DEPTH}  {ACTIVATION}  ->  (u, v, p)")
print(f"Sampling       : BOUNDARY_POINTS={BOUNDARY_POINTS}, INTERIOR={INTERIOR_POINTS}")
print(f"Training       : Adam, lr={LR}, {EPOCHS} iterations\n")

t_start = time.perf_counter()
losshistory, train_state = model.train(iterations=EPOCHS, display_every=200)
train_time_s = time.perf_counter() - t_start

print(f"\nTraining time  : {train_time_s:.2f} s")
print(f"Best train loss: {train_state.best_loss_train:.6f}")
print(f"Best test  loss: {train_state.best_loss_test:.6f}")

# ===========================================================================
# 7. Evaluate on uniform grid (matching DeepFlow EVAL_GRID)
# ===========================================================================
print("Evaluating on uniform grid ...")
x_lin = np.linspace(0, Lx, EVAL_GRID[0])
y_lin = np.linspace(0, Ly, EVAL_GRID[1])
X_grid, Y_grid = np.meshgrid(x_lin, y_lin, indexing="ij")
X_pred = np.hstack([X_grid.reshape(-1, 1), Y_grid.reshape(-1, 1)])

# Predict (u, v, p)
Y_pred = model.predict(X_pred)
u_pred = Y_pred[:, 0]
v_pred = Y_pred[:, 1]
p_pred = Y_pred[:, 2]

# PDE residuals on the same grid
continuity_res, x_momentum_res, y_momentum_res = None, None, None
try:
    residual_pred = model.predict(X_pred, operator=pde)
    if isinstance(residual_pred, (list, tuple)):
        continuity_res = np.asarray(residual_pred[0]).ravel()
        x_momentum_res = np.asarray(residual_pred[1]).ravel()
        y_momentum_res = np.asarray(residual_pred[2]).ravel()
    else:
        continuity_res = np.asarray(residual_pred[:, 0]).ravel()
        x_momentum_res = np.asarray(residual_pred[:, 1]).ravel()
        y_momentum_res = np.asarray(residual_pred[:, 2]).ravel()
except Exception:
    # Fallback: compute residuals using a fresh torch graph
    warnings.warn("Direct operator prediction failed; computing residuals manually.")
    xt = torch.tensor(X_pred, dtype=torch.float32, requires_grad=True)
    yt = model.net(xt) if hasattr(model, "net") else model.predict(X_pred)
    if not isinstance(yt, torch.Tensor):
        yt = torch.tensor(yt, dtype=torch.float32, requires_grad=True)
    res_t = pde(xt, yt)
    continuity_res = res_t[0].detach().cpu().numpy().ravel()
    x_momentum_res = res_t[1].detach().cpu().numpy().ravel()
    y_momentum_res = res_t[2].detach().cpu().numpy().ravel()

if continuity_res is None:
    continuity_res = np.array([])
if x_momentum_res is None:
    x_momentum_res = np.array([])
if y_momentum_res is None:
    y_momentum_res = np.array([])

# Loss history
loss_train = np.array(losshistory.loss_train) if hasattr(losshistory, "loss_train") else None
loss_test  = np.array(losshistory.loss_test) if hasattr(losshistory, "loss_test") else None
loss_steps = np.array(losshistory.steps) if hasattr(losshistory, "steps") else None

loss_train_total = None
loss_test_total = None
if loss_train is not None and loss_train.ndim == 2:
    # loss_train shape: (n_iterations, n_loss_terms) — take sum per iteration
    loss_train_total = loss_train.sum(axis=1)
if loss_test is not None and loss_test.ndim == 2:
    loss_test_total = loss_test.sum(axis=1)

# Final losses
final_total_loss = float(loss_train_total[-1]) if loss_train_total is not None else float("nan")

# ===========================================================================
# 8. Save to NPZ
# ===========================================================================
_results_dir = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(_results_dir, exist_ok=True)

np.savez(
    os.path.join(_results_dir, "deepxde_results.npz"),
    # Fields on evaluation grid
    x=X_grid.ravel(),
    y=Y_grid.ravel(),
    u=u_pred,
    v=v_pred,
    p=p_pred,
    # PDE residuals
    continuity_residual=continuity_res,
    x_momentum_residual=x_momentum_res,
    y_momentum_residual=y_momentum_res,
    # Loss history
    loss_train=loss_train_total if loss_train_total is not None else np.array([]),
    loss_test=loss_test_total if loss_test_total is not None else np.array([]),
    loss_steps=loss_steps if loss_steps is not None else np.array([]),
    # Summary metrics
    train_time_s=np.float64(train_time_s),
    final_total_loss=np.float64(final_total_loss),
    best_loss_train=np.float64(train_state.best_loss_train),
    best_loss_test=np.float64(train_state.best_loss_test),
    best_step=np.int64(train_state.best_step),
)

print("Results saved to results/deepxde_results.npz")
print("=" * 60)
