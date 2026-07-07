#!/usr/bin/env python3
"""
Benchmark: FP32 vs FP64 precision for the 1D Burgers equation using DeepFlow.

This script trains identical PINNs with single (FP32) and double (FP64) precision
and reports whether FP64 yields measurably better accuracy for this problem.

Usage
-----
Run from the repository root::

    python benchmarks/comparing_precision/benchmark_precision.py

For more statistically robust results, average over multiple runs::

    python benchmarks/comparing_precision/benchmark_precision.py --num_runs 5
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import sin, pi

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import deepflow as df
from common_config import (
    X_RANGE,
    Y_RANGE,
    NU,
    WIDTH,
    DEPTH,
    EPOCHS,
    SEED,
    BOUNDARY_POINTS,
    INTERIOR_POINTS,
    EVAL_GRID,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def build_domain():
    """Build the Burgers-equation domain (geometry, PDE, BCs)."""
    area = df.geometry.rectangle(list(X_RANGE), list(Y_RANGE))
    line_ic = df.geometry.line_horizontal(y=Y_RANGE[0], range_x=list(X_RANGE))
    line_bc1 = df.geometry.line_vertical(x=X_RANGE[0], range_y=list(Y_RANGE))
    line_bc2 = df.geometry.line_vertical(x=X_RANGE[1], range_y=list(Y_RANGE))
    domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)

    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})

    domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)
    return domain


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _to_numpy_dict(data_dict):
    """Convert a dictionary of tensors/lists to numpy arrays."""
    out = {}
    for key, value in data_dict.items():
        try:
            if isinstance(value, torch.Tensor):
                out[key] = value.detach().cpu().numpy()
            elif isinstance(value, list):
                out[key] = np.asarray(value)
            elif isinstance(value, np.ndarray):
                out[key] = value
            else:
                out[key] = np.asarray(value)
        except Exception as e:
            print(f"Warning: could not convert key '{key}' to numpy: {e}")
            out[key] = value
    return out


def train_one(dtype, seed, epochs):
    """Train one PINN with the requested floating-point dtype."""
    label = "FP32" if dtype == torch.float32 else "FP64"
    print(f"\n--- Training {label} (seed {seed}) ---")

    # Set global precision for this run
    df.dtype = dtype

    # Reproducibility
    df.manual_seed(seed)

    # Build problem and model
    domain = build_domain()
    model0 = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=WIDTH,
        length=DEPTH,
    )

    calc_loss = df.calc_loss_simple(domain)

    # Train
    t_start = time.perf_counter()
    model, model_best = model0.train_lbfgs(
        epochs=epochs,
        calc_loss=calc_loss,
        print_every=max(1, epochs // 10),
    )
    train_time_s = time.perf_counter() - t_start

    # Evaluate on a uniform grid
    prediction = domain.area_list[0].evaluate(model_best)
    prediction.sampling_area(EVAL_GRID)
    data = prediction.data_dict

    return {
        "label": label,
        "dtype": str(dtype),
        "train_time_s": float(train_time_s),
        "final_total_loss": float(data["total_loss"][-1]),
        "final_bc_loss": float(data["bc_loss"][-1]),
        "final_pde_loss": float(data["pde_loss"][-1]),
        "max_pde_residual": float(np.max(np.abs(data["pde_residual"]))),
        "mean_abs_pde_residual": float(np.mean(np.abs(data["pde_residual"]))),
        "x": np.asarray(data["x"]),
        "y": np.asarray(data["y"]),
        "u": np.asarray(data["u"]),
        "total_loss_history": np.asarray(model.loss_history["total_loss"], dtype=np.float64),
        "bc_loss_history": np.asarray(model.loss_history["bc_loss"], dtype=np.float64),
        "pde_loss_history": np.asarray(model.loss_history["pde_loss"], dtype=np.float64),
    }


def run_precision_benchmark(dtype, num_runs, epochs):
    """Run multiple training runs for one precision and aggregate results."""
    per_run = [train_one(dtype, SEED + i, epochs) for i in range(num_runs)]

    times = np.asarray([r["train_time_s"] for r in per_run], dtype=np.float64)
    final_total = np.asarray([r["final_total_loss"] for r in per_run], dtype=np.float64)
    final_bc = np.asarray([r["final_bc_loss"] for r in per_run], dtype=np.float64)
    final_pde = np.asarray([r["final_pde_loss"] for r in per_run], dtype=np.float64)
    max_res = np.asarray([r["max_pde_residual"] for r in per_run], dtype=np.float64)
    mean_res = np.asarray([r["mean_abs_pde_residual"] for r in per_run], dtype=np.float64)

    # Use median-loss run for representative fields and histories
    median_idx = int(np.argsort(final_total)[len(final_total) // 2])

    return {
        "label": per_run[0]["label"],
        "dtype": per_run[0]["dtype"],
        "num_runs": num_runs,
        "epochs": epochs,
        "train_time_mean": float(times.mean()),
        "train_time_std": float(times.std(ddof=1)) if num_runs > 1 else 0.0,
        "final_total_loss_mean": float(final_total.mean()),
        "final_total_loss_std": float(final_total.std(ddof=1)) if num_runs > 1 else 0.0,
        "final_bc_loss_mean": float(final_bc.mean()),
        "final_bc_loss_std": float(final_bc.std(ddof=1)) if num_runs > 1 else 0.0,
        "final_pde_loss_mean": float(final_pde.mean()),
        "final_pde_loss_std": float(final_pde.std(ddof=1)) if num_runs > 1 else 0.0,
        "max_pde_residual_mean": float(max_res.mean()),
        "max_pde_residual_std": float(max_res.std(ddof=1)) if num_runs > 1 else 0.0,
        "mean_abs_pde_residual_mean": float(mean_res.mean()),
        "mean_abs_pde_residual_std": float(mean_res.std(ddof=1)) if num_runs > 1 else 0.0,
        "median_run_idx": median_idx,
        "x": per_run[median_idx]["x"],
        "y": per_run[median_idx]["y"],
        "u": per_run[median_idx]["u"],
        "total_loss_history": per_run[median_idx]["total_loss_history"],
        "bc_loss_history": per_run[median_idx]["bc_loss_history"],
        "pde_loss_history": per_run[median_idx]["pde_loss_history"],
        "train_time_s": times,
        "final_total_loss_runs": final_total,
    }


# ---------------------------------------------------------------------------
# Reporting / plotting helpers
# ---------------------------------------------------------------------------

def _format_mean_std(mean, std):
    if std is not None and std > 0:
        return f"{mean:.6e} ± {std:.6e}"
    return f"{mean:.6e}"


def _pct_delta(fp32, fp64):
    if fp32 == 0:
        return float("nan")
    return (fp64 - fp32) / fp32 * 100.0


def print_summary(res32, res64):
    print("\n" + "=" * 90)
    print("FP32 vs FP64 Precision Benchmark Summary")
    print("=" * 90)
    print(f"{'Metric':<30} {'FP32':>24} {'FP64':>24} {'Delta':>10}")
    print("-" * 90)

    rows = [
        (
            "Final total loss",
            res32["final_total_loss_mean"],
            res32["final_total_loss_std"],
            res64["final_total_loss_mean"],
            res64["final_total_loss_std"],
        ),
        (
            "Final BC loss",
            res32["final_bc_loss_mean"],
            res32["final_bc_loss_std"],
            res64["final_bc_loss_mean"],
            res64["final_bc_loss_std"],
        ),
        (
            "Final PDE loss",
            res32["final_pde_loss_mean"],
            res32["final_pde_loss_std"],
            res64["final_pde_loss_mean"],
            res64["final_pde_loss_std"],
        ),
        (
            "Max |PDE residual|",
            res32["max_pde_residual_mean"],
            res32["max_pde_residual_std"],
            res64["max_pde_residual_mean"],
            res64["max_pde_residual_std"],
        ),
        (
            "Mean |PDE residual|",
            res32["mean_abs_pde_residual_mean"],
            res32["mean_abs_pde_residual_std"],
            res64["mean_abs_pde_residual_mean"],
            res64["mean_abs_pde_residual_std"],
        ),
        (
            "Train time (s)",
            res32["train_time_mean"],
            res32["train_time_std"],
            res64["train_time_mean"],
            res64["train_time_std"],
        ),
    ]

    for label, m32, s32, m64, s64 in rows:
        col32 = _format_mean_std(m32, s32)
        col64 = _format_mean_std(m64, s64)
        delta = _pct_delta(m32, m64)
        delta_str = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
        print(f"{label:<30} {col32:>24} {col64:>24} {delta_str:>10}")

    print("=" * 90)
    print("Interpretation: negative Delta for losses means FP64 is lower/better.")
    print("Positive Delta for time means FP64 is slower.")
    print("=" * 90)


def plot_results(res32, res64):
    """Generate loss-curve and u-field comparison figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Loss curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    histories = [
        ("total_loss_history", "Total loss"),
        ("bc_loss_history", "BC loss"),
        ("pde_loss_history", "PDE loss"),
    ]
    for ax, (key, title) in zip(axes, histories):
        ax.semilogy(res32[key], label="FP32")
        ax.semilogy(res64[key], label="FP64")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    loss_path = RESULTS_DIR / "loss_curves.png"
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    print(f"Loss-curve figure saved to: {loss_path}")

    # u-field comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    u_range = (
        float(min(res32["u"].min(), res64["u"].min())),
        float(max(res32["u"].max(), res64["u"].max())),
    )

    coords = [res32, res64]
    for ax, label, res in zip(axes[:2], ["FP32", "FP64"], coords):
        sc = ax.scatter(res["x"], res["y"], c=res["u"], s=1, cmap="jet", marker="s")
        sc.set_clim(u_range)
        ax.set_title(f"u – {label}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(sc, ax=ax, shrink=0.8)

    diff = res64["u"] - res32["u"]
    ax = axes[2]
    sc = ax.scatter(res64["x"], res64["y"], c=diff, s=1, cmap="RdBu_r", marker="s")
    ax.set_title("u difference (FP64 − FP32)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, shrink=0.8)

    plt.tight_layout()
    field_path = RESULTS_DIR / "u_field_comparison.png"
    fig.savefig(field_path, dpi=150)
    plt.close(fig)
    print(f"u-field figure saved to: {field_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP32 vs FP64 precision for DeepFlow PINNs."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of independent runs to average over for each precision. Default: 1",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Number of LBFGS epochs. Default: {EPOCHS}",
    )
    args = parser.parse_args()

    print("=" * 90)
    print("DeepFlow FP32 vs FP64 Precision Benchmark")
    print("=" * 90)
    print(f"Device: {df.device}")
    print(f"Runs per precision: {args.num_runs}")
    print(f"Epochs per run: {args.epochs}")
    print(f"Base seed: {SEED}")

    # FP32 first, then FP64
    res32 = run_precision_benchmark(torch.float32, args.num_runs, args.epochs)
    res64 = run_precision_benchmark(torch.float64, args.num_runs, args.epochs)

    # Summary table
    print_summary(res32, res64)

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for res in (res32, res64):
        out_path = RESULTS_DIR / f"{res['label'].lower()}_results.npz"
        np.savez(out_path, **res)
        print(f"Saved {res['label']} results to: {out_path}")

    # Figures
    plot_results(res32, res64)

    # Final verdict
    better_total = res64["final_total_loss_mean"] < res32["final_total_loss_mean"]
    better_pde = res64["mean_abs_pde_residual_mean"] < res32["mean_abs_pde_residual_mean"]
    print("\nVerdict:")
    print(f"  FP64 total loss is {'LOWER (better)' if better_total else 'HIGHER (worse)'} than FP32.")
    print(f"  FP64 mean PDE residual is {'LOWER (better)' if better_pde else 'HIGHER (worse)'} than FP32.")


if __name__ == "__main__":
    main()
