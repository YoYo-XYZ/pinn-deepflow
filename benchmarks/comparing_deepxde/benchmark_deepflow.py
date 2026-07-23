#!/usr/bin/env python3
"""
Benchmark: 2D steady channel flow using DeepFlow.

Matches the setup in ``static/quickstart/code.ipynb``.

Results are saved to ``results/deepflow_results.npz``.
"""

import time

import numpy as np

from benchmark_common import RESULTS_DIR
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

def _boundary_points():
    """Match DeepXDE's perimeter-weighted boundary sampling."""
    perimeter = 2 * (Lx + Ly)
    total = sum(BOUNDARY_POINTS)
    return [
        int(total * Ly / perimeter),
        int(total * Lx / perimeter),
        int(total * Ly / perimeter),
        int(total * Lx / perimeter),
    ]


def build_domain():
    """Build and sample the channel-flow domain."""
    domain = df.domain(df.geometry.rectangle([0, Lx], [0, Ly]))
    domain.bound_list[0].define_bc({"u": 1, "v": 0})
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.bound_list[3].define_bc({"u": 0, "v": 0})
    domain.area_list[0].define_pde(
        df.pde.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000)
    )
    boundary_points = _boundary_points()
    domain.sampling_random(boundary_points, [INTERIOR_POINTS])
    return domain, boundary_points


def build_model():
    return df.PINN(
        width=WIDTH,
        length=DEPTH,
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
    )


def _loss_value(losses, name):
    return float(losses[name].detach().cpu().item())


def main():
    df.manual_seed(SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("DeepFlow  -  2D Steady Channel Flow Benchmark")
    print("=" * 60)

    domain, boundary_points = build_domain()
    model = build_model()
    print(f"\nNetwork        : {WIDTH}x{DEPTH}  Tanh  ->  (u, v, p)")
    print(f"Sampling       : BOUNDARY_POINTS={boundary_points} (total={sum(boundary_points)}), INTERIOR={INTERIOR_POINTS}")
    print(f"Training       : Adam, lr={LR}, {EPOCHS} epochs\n")

    calc_loss = df.calc_loss_simple(domain)
    start = time.perf_counter()
    _, best_model = model.train_adam(
        calc_loss=calc_loss,
        learning_rate=LR,
        epochs=EPOCHS,
    )
    train_time_s = time.perf_counter() - start
    print(f"\nTraining time  : {train_time_s:.2f} s")

    best_model.eval()
    best_loss = calc_loss(best_model)
    print("Evaluating on uniform grid ...")
    prediction = domain.area_list[0].evaluate(best_model)
    prediction.sampling_area(EVAL_GRID)
    data = prediction.data_dict
    final_total = _loss_value(best_loss, "total_loss")
    final_bc = _loss_value(best_loss, "bc_loss")
    final_pde = _loss_value(best_loss, "pde_loss")
    print(f"Final loss     : total={final_total:.6f}  bc={final_bc:.6f}  pde={final_pde:.6f}")

    np.savez(
        RESULTS_DIR / "deepflow_results.npz",
        x=np.asarray(data.get("x", [])),
        y=np.asarray(data.get("y", [])),
        u=np.asarray(data.get("u", [])),
        v=np.asarray(data.get("v", [])),
        p=np.asarray(data.get("p", [])),
        continuity_residual=np.asarray(data.get("continuity_residual", [])),
        x_momentum_residual=np.asarray(data.get("x_momentum_residual", [])),
        y_momentum_residual=np.asarray(data.get("y_momentum_residual", [])),
        total_loss=np.asarray(data.get("total_loss", [])),
        bc_loss=np.asarray(data.get("bc_loss", [])),
        pde_loss=np.asarray(data.get("pde_loss", [])),
        train_time_s=np.float64(train_time_s),
        final_total_loss=np.float64(final_total),
        final_bc_loss=np.float64(final_bc),
        final_pde_loss=np.float64(final_pde),
        boundary_points=np.asarray(boundary_points),
    )
    print("Results saved to results/deepflow_results.npz")
    print("=" * 60)


if __name__ == "__main__":
    main()
