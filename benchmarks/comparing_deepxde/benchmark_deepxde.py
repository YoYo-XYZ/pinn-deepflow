#!/usr/bin/env python3
"""Benchmark the 2D steady channel flow problem with DeepXDE."""

import os
import time
import warnings

import numpy as np

from benchmark_common import RESULTS_DIR, evaluation_grid

# DeepXDE reads its backend during import.
os.environ["DDE_BACKEND"] = "pytorch"

import deepxde as dde  # noqa: E402
import torch  # noqa: E402

from common_config import (  # noqa: E402
    ACTIVATION,
    BOUNDARY_POINTS,
    DEPTH,
    EPOCHS,
    EVAL_GRID,
    INTERIOR_POINTS,
    Lx,
    Ly,
    LR,
    Re,
    SEED,
    WIDTH,
)


def pde(x, y):
    """Return continuity, x-momentum, and y-momentum residuals."""
    u, v = y[:, 0:1], y[:, 1:2]
    u_x = dde.grad.jacobian(y, x, i=0, j=0)
    u_y = dde.grad.jacobian(y, x, i=0, j=1)
    v_x = dde.grad.jacobian(y, x, i=1, j=0)
    v_y = dde.grad.jacobian(y, x, i=1, j=1)
    p_x = dde.grad.jacobian(y, x, i=2, j=0)
    p_y = dde.grad.jacobian(y, x, i=2, j=1)
    u_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    u_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    v_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    v_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    viscosity = 1.0 / Re
    return [
        u_x + v_y,
        u * u_x + v * u_y + p_x - viscosity * (u_xx + u_yy),
        u * v_x + v * v_y + p_y - viscosity * (v_xx + v_yy),
    ]


def boundary_left(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0)


def boundary_right(x, on_boundary):
    return on_boundary and np.isclose(x[0], Lx)


def boundary_wall(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0.0) or np.isclose(x[1], Ly))


def build_model():
    """Build the DeepXDE model and its training data."""
    geometry = dde.geometry.Rectangle(xmin=[0, 0], xmax=[Lx, Ly])
    bcs = [
        dde.icbc.DirichletBC(geometry, lambda x: 1.0, boundary_left, component=0),
        dde.icbc.DirichletBC(geometry, lambda x: 0.0, boundary_left, component=1),
        dde.icbc.DirichletBC(geometry, lambda x: 0.0, boundary_wall, component=0),
        dde.icbc.DirichletBC(geometry, lambda x: 0.0, boundary_wall, component=1),
        dde.icbc.DirichletBC(geometry, lambda x: 0.0, boundary_right, component=2),
    ]
    data = dde.data.PDE(
        geometry,
        pde,
        bcs,
        num_domain=INTERIOR_POINTS,
        num_boundary=sum(BOUNDARY_POINTS),
        num_test=5000,
    )
    layers = [2] + [WIDTH] * DEPTH + [3]
    return dde.Model(data, dde.nn.FNN(layers, ACTIVATION, "Glorot normal"))


def _residuals(model, points):
    """Evaluate PDE residuals, with a PyTorch fallback for older DeepXDE."""
    try:
        values = model.predict(points, operator=pde)
        if isinstance(values, (list, tuple)):
            return [np.asarray(value).ravel() for value in values]
        return [np.asarray(values[:, index]).ravel() for index in range(3)]
    except Exception:
        warnings.warn("Direct operator prediction failed; computing residuals manually.")
        net = getattr(model, "net", None)
        if net is None:
            inputs = torch.tensor(points, dtype=torch.float32, requires_grad=True)
            outputs = torch.tensor(model.predict(points), dtype=torch.float32)
        else:
            device = next(net.parameters()).device
            inputs = torch.tensor(
                points, dtype=torch.float32, device=device, requires_grad=True
            )
            outputs = net(inputs)
        values = pde(inputs, outputs)
        return [value.detach().cpu().numpy().ravel() for value in values]


def _total_losses(history, attribute):
    values = np.asarray(getattr(history, attribute, []))
    return values.sum(axis=1) if values.ndim == 2 else np.array([])


def main():
    if torch.cuda.is_available():
        device = "cuda"
        try:
            torch.set_default_device(device)
        except AttributeError:
            pass
    else:
        device = "cpu"

    np.random.seed(SEED)
    torch.manual_seed(SEED)
    print("=" * 60)
    print("DeepXDE (PyTorch)  -  2D Steady Channel Flow Benchmark")
    print(f"Device            : {device}")
    print("=" * 60)

    model = build_model()
    model.compile("adam", lr=LR)
    print(f"\nNetwork        : {WIDTH}x{DEPTH}  {ACTIVATION}  ->  (u, v, p)")
    print(f"Sampling       : BOUNDARY_POINTS={BOUNDARY_POINTS}, INTERIOR={INTERIOR_POINTS}")
    print(f"Training       : Adam, lr={LR}, {EPOCHS} iterations\n")

    start = time.perf_counter()
    history, state = model.train(iterations=EPOCHS, display_every=200)
    train_time_s = time.perf_counter() - start
    print(f"\nTraining time  : {train_time_s:.2f} s")
    print(f"Best train loss: {state.best_loss_train:.6f}")
    print(f"Best test  loss: {state.best_loss_test:.6f}")

    print("Evaluating on uniform grid ...")
    x_grid, y_grid, points = evaluation_grid(Lx, Ly, EVAL_GRID)
    prediction = model.predict(points)
    residuals = _residuals(model, points)
    loss_train = _total_losses(history, "loss_train")
    loss_test = _total_losses(history, "loss_test")
    loss_steps = np.asarray(getattr(history, "steps", []))
    final_total_loss = float(loss_train[-1]) if len(loss_train) else float("nan")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        RESULTS_DIR / "deepxde_results.npz",
        x=x_grid.ravel(),
        y=y_grid.ravel(),
        u=prediction[:, 0],
        v=prediction[:, 1],
        p=prediction[:, 2],
        continuity_residual=residuals[0],
        x_momentum_residual=residuals[1],
        y_momentum_residual=residuals[2],
        loss_train=loss_train,
        loss_test=loss_test,
        loss_steps=loss_steps,
        train_time_s=np.float64(train_time_s),
        final_total_loss=np.float64(final_total_loss),
        best_loss_train=np.float64(state.best_loss_train),
        best_loss_test=np.float64(state.best_loss_test),
        best_step=np.int64(state.best_step),
    )
    print("Results saved to results/deepxde_results.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
