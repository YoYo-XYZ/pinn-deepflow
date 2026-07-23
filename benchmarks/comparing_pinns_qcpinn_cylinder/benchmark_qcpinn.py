#!/usr/bin/env python3
"""
Benchmark: QCPINN (Quantum-Classical PINN) training for 2D steady cylinder flow.

Trains a QCPINN (pre=[50], post=[50], nqubits=4 → ~769 params) on the same
Navier-Stokes cylinder problem as the classical PINN baseline. Results are
saved to ``results/qcpinn_results.npz`` for later comparison by ``compare.py``.

Usage:
    python benchmark_qcpinn.py                          # default: 3 runs
    python benchmark_qcpinn.py --num_runs 1             # single run
    python benchmark_qcpinn.py --epochs_adam 100        # quick smoke test
"""

import os
import sys
import argparse
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# PennyLane is a hard dependency for QCPINN.
try:
    import pennylane as qml  # noqa: F401  (imported for side-effect check)
    _PENNYLANE_AVAILABLE = True
except ImportError:
    _PENNYLANE_AVAILABLE = False

import deepflow as df
from common_config import (
    CHANNEL_X,
    CHANNEL_Y,
    CYLINDER_CX,
    CYLINDER_CY,
    CYLINDER_R,
    U_INF,
    L_CHAR,
    MU,
    RHO,
    BOUNDARY_POINTS,
    INTERIOR_POINTS,
    RESAMPLE_EVERY,
    LR_ADAM,
    EPOCHS_ADAM,
    THRESHOLD_ADAM,
    EPOCHS_LBFGS,
    THRESHOLD_LBFGS,
    EVAL_GRID,
    OUTLET_LINE_POINTS,
    QC_PRE,
    QC_POST,
    QC_NQUBITS,
    QC_ITERATIONS,
    SEEDS,
    RESULTS_DIR,
    QCPINN_RESULTS_FILE,
)

if not _PENNYLANE_AVAILABLE:
    print(
        "[ERROR] PennyLane is required for QCPINN but is not installed.\n"
        "        Install it with:  pip install pennylane\n"
        "        Then re-run this benchmark."
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Problem construction (identical to benchmark_pinn.py)
# ---------------------------------------------------------------------------

def build_domain():
    circle = df.geometry.circle(CYLINDER_CX, CYLINDER_CY, CYLINDER_R)
    rectangle = df.geometry.rectangle(list(CHANNEL_X), list(CHANNEL_Y))
    area = rectangle - circle
    domain = df.domain(area, circle.bound_list)

    domain.area_list[0].define_pde(
        df.pde.NavierStokes(U=U_INF, L=L_CHAR, mu=MU, rho=RHO)
    )

    channel_height = CHANNEL_Y[1]
    domain.bound_list[0].define_bc({
        "u": ["y", lambda y: 4 * U_INF * y * (channel_height - y) / channel_height ** 2],
        "v": 0,
    })
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.bound_list[3].define_bc({"u": 0, "v": 0})
    domain.bound_list[4].define_bc({"u": 0, "v": 0})
    domain.bound_list[5].define_bc({"u": 0, "v": 0})

    domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)
    return domain


def do_randomr(epoch, model):
    """'randomr' callback: full LHS resampling every RESAMPLE_EVERY L-BFGS epochs."""
    if epoch > 0 and epoch % RESAMPLE_EVERY == 0:
        domain = model._current_domain
        domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)


# ---------------------------------------------------------------------------
# Per-run training
# ---------------------------------------------------------------------------

def _count_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_one(seed, epochs_adam, epochs_lbfgs):
    """Train one QCPINN with the given seed."""
    print(f"\n--- QCPINN run (seed {seed}) ---")
    df.manual_seed(seed)

    domain = build_domain()
    calc_loss = df.calc_loss_simple(domain)

    model0 = df.QCPINN(
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        hidden_layer_pre=QC_PRE,
        hidden_layer_post=QC_POST,
        nqubits=QC_NQUBITS,
        q_layer_iterations=QC_ITERATIONS,
    )
    n_params = _count_params(model0)
    print(
        f"  Model: QCPINN(pre={QC_PRE}, post={QC_POST}, nqubits={QC_NQUBITS},"
        f" iter={QC_ITERATIONS}) — {n_params} trainable params"
    )

    model0._current_domain = domain

    # Phase 1: Adam
    t0 = time.perf_counter()
    model_adam, model_adam_best = model0.train_adam(
        calc_loss=calc_loss,
        learning_rate=LR_ADAM,
        epochs=epochs_adam,
        threshold_loss=THRESHOLD_ADAM,
        do_between_epochs=None,
        print_every=max(1, epochs_adam // 10),
    )
    t_adam = time.perf_counter() - t0

    # Phase 2: L-BFGS
    t0 = time.perf_counter()
    model_final, model_best = model_adam_best.train_lbfgs(
        calc_loss=calc_loss,
        epochs=epochs_lbfgs,
        threshold_loss=THRESHOLD_LBFGS,
        do_between_epochs=do_randomr,
        print_every=max(1, epochs_lbfgs // 10),
    )
    t_lbfgs = time.perf_counter() - t0

    area_eval = domain.area_list[0].evaluate(model_best)
    area_eval.sampling_area(EVAL_GRID)
    data = area_eval.data_dict

    outlet_eval = domain.bound_list[2].evaluate(model_best)
    outlet_eval.sampling_line(OUTLET_LINE_POINTS)
    outlet_data = outlet_eval.data_dict

    return {
        "seed": seed,
        "n_params": n_params,
        "adam_time_s": float(t_adam),
        "lbfgs_time_s": float(t_lbfgs),
        "total_time_s": float(t_adam + t_lbfgs),
        "final_total_loss": float(data["total_loss"][-1]),
        "final_bc_loss": float(data["bc_loss"][-1]),
        "final_pde_loss": float(data["pde_loss"][-1]),
        "max_continuity": float(np.max(np.abs(data["continuity_residual"]))),
        "max_x_momentum": float(np.max(np.abs(data["x_momentum_residual"]))),
        "max_y_momentum": float(np.max(np.abs(data["y_momentum_residual"]))),
        "mean_abs_continuity": float(np.mean(np.abs(data["continuity_residual"]))),
        "mean_abs_x_momentum": float(np.mean(np.abs(data["x_momentum_residual"]))),
        "mean_abs_y_momentum": float(np.mean(np.abs(data["y_momentum_residual"]))),
        "x": np.asarray(data["x"]),
        "y": np.asarray(data["y"]),
        "u": np.asarray(data["u"]),
        "v": np.asarray(data["v"]),
        "p": np.asarray(data["p"]),
        "continuity_residual": np.asarray(data["continuity_residual"]),
        "x_momentum_residual": np.asarray(data["x_momentum_residual"]),
        "y_momentum_residual": np.asarray(data["y_momentum_residual"]),
        "total_loss_history": np.asarray(
            model_best.loss_history["total_loss"], dtype=np.float64
        ),
        "bc_loss_history": np.asarray(
            model_best.loss_history["bc_loss"], dtype=np.float64
        ),
        "pde_loss_history": np.asarray(
            model_best.loss_history["pde_loss"], dtype=np.float64
        ),
        "outlet_y": np.asarray(outlet_data["y"]),
        "outlet_u": np.asarray(outlet_data["u"]),
    }


# ---------------------------------------------------------------------------
# Aggregation (identical to benchmark_pinn.py)
# ---------------------------------------------------------------------------

def _agg(arr, num_runs):
    arr = np.asarray(arr, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if num_runs > 1 else 0.0
    return mean, std


def aggregate(per_run, num_runs, epochs_adam, epochs_lbfgs):
    final_total = np.array([r["final_total_loss"] for r in per_run])
    median_idx = int(np.argsort(final_total)[len(final_total) // 2])

    out = {
        "label": "QCPINN",
        "n_params": per_run[0]["n_params"],
        "num_runs": num_runs,
        "epochs_adam": epochs_adam,
        "epochs_lbfgs": epochs_lbfgs,
        "median_run_idx": median_idx,
    }
    scalar_keys = [
        "final_total_loss", "final_bc_loss", "final_pde_loss",
        "max_continuity", "max_x_momentum", "max_y_momentum",
        "mean_abs_continuity", "mean_abs_x_momentum", "mean_abs_y_momentum",
        "adam_time_s", "lbfgs_time_s", "total_time_s",
    ]
    for key in scalar_keys:
        m, s = _agg([r[key] for r in per_run], num_runs)
        out[f"{key}_mean"] = m
        out[f"{key}_std"] = s

    out["final_total_loss_runs"] = final_total
    out["total_time_s_runs"] = np.array([r["total_time_s"] for r in per_run])

    med = per_run[median_idx]
    for k in (
        "x", "y", "u", "v", "p",
        "continuity_residual", "x_momentum_residual", "y_momentum_residual",
        "total_loss_history", "bc_loss_history", "pde_loss_history",
        "outlet_y", "outlet_u",
    ):
        out[k] = med[k]
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: QCPINN for 2D steady cylinder flow (Re=50)."
    )
    parser.add_argument("--num_runs", type=int, default=len(SEEDS),
                        help=f"Number of independent runs (default: {len(SEEDS)}).")
    parser.add_argument("--epochs_adam", type=int, default=EPOCHS_ADAM,
                        help=f"Adam epochs (default: {EPOCHS_ADAM}).")
    parser.add_argument("--epochs_lbfgs", type=int, default=EPOCHS_LBFGS,
                        help=f"L-BFGS epochs (default: {EPOCHS_LBFGS}).")
    args = parser.parse_args()

    num_runs = args.num_runs
    print("=" * 80)
    print("QCPINN Benchmark — 2D Steady Cylinder Flow (Re=50)")
    print("=" * 80)
    print(f"Device:           {df.device}")
    print(f"Runs:             {num_runs}")
    print(f"Adam epochs:      {args.epochs_adam}")
    print(f"L-BFGS epochs:    {args.epochs_lbfgs}")
    print(f"Base seed:        {SEEDS[0]}")
    print(
        f"Network:          QCPINN(pre={QC_PRE}, post={QC_POST},"
        f" nqubits={QC_NQUBITS}, iter={QC_ITERATIONS})"
    )

    per_run = []
    for i in range(num_runs):
        seed = SEEDS[i] if i < len(SEEDS) else SEEDS[-1] + (i - len(SEEDS) + 1)
        per_run.append(train_one(seed, args.epochs_adam, args.epochs_lbfgs))

    results = aggregate(per_run, num_runs, args.epochs_adam, args.epochs_lbfgs)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(QCPINN_RESULTS_FILE, **results)
    print(f"\nResults saved to: {QCPINN_RESULTS_FILE}")

    print("\n" + "-" * 60)
    print(f"QCPINN  ({results['n_params']} params, {num_runs} runs)")
    print(f"  final total loss : {results['final_total_loss_mean']:.4e} ± {results['final_total_loss_std']:.4e}")
    print(f"  final PDE loss   : {results['final_pde_loss_mean']:.4e} ± {results['final_pde_loss_std']:.4e}")
    print(f"  max |continuity| : {results['max_continuity_mean']:.4e} ± {results['max_continuity_std']:.4e}")
    print("-" * 60)


if __name__ == "__main__":
    main()
