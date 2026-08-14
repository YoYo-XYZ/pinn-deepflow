#!/usr/bin/env python3
"""
Benchmark: FP32 vs FP64 precision for the 1D Burgers equation using DeepFlow.

This script trains identical PINNs with single (FP32) and double (FP64) precision
and compares their training losses and PDE residuals.

Usage
-----
Run from the repository root::

    python benchmarks/comparing_precision/benchmark_precision.py

For more statistically robust results, average over multiple runs::

    python benchmarks/comparing_precision/benchmark_precision.py --num_runs 5
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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

import deepflow as df  # noqa: E402
from common_config import (  # noqa: E402
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


def _positive_int(value):
    """Parse a strictly positive command-line integer."""
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------

def _unique_geometries(domain):
    """Iterate over each sampled geometry once (areas also own their bounds)."""
    geometries = []
    seen = set()
    for geometry in domain.bound_list + domain.area_list:
        if id(geometry) not in seen:
            seen.add(id(geometry))
            geometries.append(geometry)
    return geometries


def _capture_coordinates(domain):
    """Capture sampled coordinates as CPU FP32 tensors for both precision runs."""
    return tuple(
        (
            geometry.X.detach().cpu().clone(),
            geometry.Y.detach().cpu().clone(),
        )
        for geometry in _unique_geometries(domain)
    )


def _apply_coordinates(domain, coordinates, dtype):
    """Restore baseline coordinates and prepare them in the requested dtype."""
    geometries = _unique_geometries(domain)
    if len(geometries) != len(coordinates):
        raise ValueError("Baseline and target domains have different geometry layouts.")

    for geometry, (x, y) in zip(geometries, coordinates):
        geometry.set_coordinates(
            x.to(dtype=dtype).clone(),
            y.to(dtype=dtype).clone(),
        )
    for geometry in geometries:
        geometry.process_coordinates()


def build_domain(sampled_coordinates=None):
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

    if sampled_coordinates is None:
        domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)
    else:
        _apply_coordinates(domain, sampled_coordinates, df.dtype)
        domain.sampling_option = "paired_baseline"
    return domain


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _seed(seed):
    """Seed every RNG used by the benchmark and request deterministic kernels."""
    df.manual_seed(seed, deterministic=True)


def _synchronize_cuda():
    """Synchronize only when this benchmark is actually running on CUDA."""
    if torch.cuda.is_available() and str(df.device).startswith("cuda"):
        torch.cuda.synchronize()


def _build_baseline(seed):
    """Create one FP32 domain/model baseline for a paired seed."""
    df.dtype = torch.float32
    _seed(seed)
    domain = build_domain()
    model = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=WIDTH,
        length=DEPTH,
    )
    return {
        "coordinates": _capture_coordinates(domain),
        "model_state": {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        },
    }


def _build_model(dtype, model_state):
    """Build a model and load the shared FP32 state cast to ``dtype``."""
    model = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=WIDTH,
        length=DEPTH,
    )
    cast_state = {
        key: value.to(dtype=dtype) if value.is_floating_point() else value.clone()
        for key, value in model_state.items()
    }
    model.load_state_dict(cast_state)
    return model


def build_evaluation_grid():
    """Create one uniform FP32 grid whose coordinates are shared by both dtypes."""
    previous_dtype = df.dtype
    df.dtype = torch.float32
    try:
        area = df.geometry.rectangle(list(X_RANGE), list(Y_RANGE))
        area.sampling_area(EVAL_GRID)
        return (
            area.X.detach().cpu().clone(),
            area.Y.detach().cpu().clone(),
        )
    finally:
        df.dtype = previous_dtype


def _set_evaluation_grid(domain, evaluation_grid, dtype):
    """Put the canonical grid on the evaluation area in the requested dtype."""
    area = domain.area_list[0]
    area.set_coordinates(
        evaluation_grid[0].to(dtype=dtype).clone(),
        evaluation_grid[1].to(dtype=dtype).clone(),
    )
    area.process_coordinates()


def train_one(dtype, seed, epochs, baseline=None, evaluation_grid=None):
    """Train one PINN with the requested floating-point dtype."""
    label = "FP32" if dtype == torch.float32 else "FP64"
    print(f"\n--- Training {label} (seed {seed}) ---")

    if baseline is None:
        baseline = _build_baseline(seed)
    if evaluation_grid is None:
        evaluation_grid = build_evaluation_grid()

    # Set global precision and restore the shared baseline in this dtype.
    df.dtype = dtype
    _seed(seed)

    # Build problem and model
    domain = build_domain(sampled_coordinates=baseline["coordinates"])
    model0 = _build_model(dtype, baseline["model_state"])

    calc_loss = df.calc_loss_simple(domain)

    # Train
    _synchronize_cuda()
    t_start = time.perf_counter()
    model, model_best = model0.train_lbfgs(
        epochs=epochs,
        calc_loss=calc_loss,
        print_every=max(1, epochs // 10),
    )
    _synchronize_cuda()
    train_time_s = time.perf_counter() - t_start

    # Recompute reported losses on the model selected by train_lbfgs.
    model_best.eval()
    best_losses = calc_loss(model_best)

    # Evaluate both dtypes on the same canonical grid.
    _set_evaluation_grid(domain, evaluation_grid, dtype)
    prediction = domain.area_list[0].evaluate(model_best)
    data = prediction.data_dict

    return {
        "label": label,
        "dtype": str(dtype),
        "seed": int(seed),
        "train_time_s": float(train_time_s),
        "final_total_loss": float(best_losses["total_loss"].detach().cpu().item()),
        "final_bc_loss": float(best_losses["bc_loss"].detach().cpu().item()),
        "final_pde_loss": float(best_losses["pde_loss"].detach().cpu().item()),
        "max_pde_residual": float(np.max(np.abs(data["pde_residual"]))),
        "mean_abs_pde_residual": float(np.mean(np.abs(data["pde_residual"]))),
        "x": np.asarray(data["x"]),
        "y": np.asarray(data["y"]),
        "u": np.asarray(data["u"]),
        "total_loss_history": np.asarray(model.loss_history["total_loss"], dtype=np.float64),
        "bc_loss_history": np.asarray(model.loss_history["bc_loss"], dtype=np.float64),
        "pde_loss_history": np.asarray(model.loss_history["pde_loss"], dtype=np.float64),
    }


def _representative_run_idx(per_run, representative_run_idx=None):
    if representative_run_idx is None:
        final_total = np.asarray(
            [run["final_total_loss"] for run in per_run], dtype=np.float64
        )
        representative_run_idx = int(np.argsort(final_total)[len(per_run) // 2])
    if not 0 <= representative_run_idx < len(per_run):
        raise ValueError("representative_run_idx is outside the available runs.")
    return representative_run_idx


def run_precision_benchmark(
    dtype,
    num_runs,
    epochs,
    paired_runs=None,
    evaluation_grid=None,
    representative_run_idx=None,
):
    """Run multiple training runs for one precision and aggregate results."""
    if paired_runs is None:
        if evaluation_grid is None:
            evaluation_grid = build_evaluation_grid()
        per_run = []
        for i in range(num_runs):
            seed = SEED + i
            baseline = _build_baseline(seed)
            per_run.append(train_one(dtype, seed, epochs, baseline, evaluation_grid))
    else:
        dtype_key = "fp32" if dtype == torch.float32 else "fp64"
        per_run = [paired_run[dtype_key] for paired_run in paired_runs]
        if len(per_run) != num_runs:
            raise ValueError("paired_runs does not match num_runs.")

    metric_names = {
        "train_time_s": "train_time",
        "final_total_loss": "final_total_loss",
        "final_bc_loss": "final_bc_loss",
        "final_pde_loss": "final_pde_loss",
        "max_pde_residual": "max_pde_residual",
        "mean_abs_pde_residual": "mean_abs_pde_residual",
    }
    metrics = {
        name: np.asarray([run[name] for run in per_run], dtype=np.float64)
        for name in metric_names
    }

    # Use one caller-selected run for both dtypes when runs are paired.
    median_idx = _representative_run_idx(per_run, representative_run_idx)

    result = {
        "label": per_run[0]["label"],
        "dtype": per_run[0]["dtype"],
        "num_runs": num_runs,
        "epochs": epochs,
        "median_run_idx": median_idx,
        "representative_run_idx": median_idx,
        "representative_seed": per_run[median_idx]["seed"],
        "run_seeds": np.asarray([run["seed"] for run in per_run], dtype=np.int64),
        "run_indices": np.arange(num_runs, dtype=np.int64),
        "x": per_run[median_idx]["x"],
        "y": per_run[median_idx]["y"],
        "u": per_run[median_idx]["u"],
        "total_loss_history": per_run[median_idx]["total_loss_history"],
        "bc_loss_history": per_run[median_idx]["bc_loss_history"],
        "pde_loss_history": per_run[median_idx]["pde_loss_history"],
    }
    for name, output_name in metric_names.items():
        values = metrics[name]
        result[f"{output_name}_mean"] = float(values.mean())
        result[f"{output_name}_std"] = (
            float(values.std(ddof=1)) if num_runs > 1 else 0.0
        )

    result["train_time_s"] = metrics["train_time_s"]
    result["final_total_loss_runs"] = metrics["final_total_loss"]
    return result


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

    metrics = [
        ("Final total loss", "final_total_loss"),
        ("Final BC loss", "final_bc_loss"),
        ("Final PDE loss", "final_pde_loss"),
        ("Max |PDE residual|", "max_pde_residual"),
        ("Mean |PDE residual|", "mean_abs_pde_residual"),
        ("Train time (s)", "train_time"),
    ]

    for label, key in metrics:
        m32, s32 = res32[f"{key}_mean"], res32[f"{key}_std"]
        m64, s64 = res64[f"{key}_mean"], res64[f"{key}_std"]
        col32, col64 = _format_mean_std(m32, s32), _format_mean_std(m64, s64)
        delta = _pct_delta(m32, m64)
        delta_str = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
        print(f"{label:<30} {col32:>24} {col64:>24} {delta_str:>10}")

    print("=" * 90)
    print("Interpretation: negative Delta for training losses/residuals means FP64 is lower/better.")
    print("These metrics measure training and PDE residual behavior, not independent solution accuracy.")
    print("Positive Delta for time means FP64 is slower.")
    print("=" * 90)


def plot_results(res32, res64):
    """Generate loss-curve and u-field comparison figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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
        type=_positive_int,
        default=1,
        help="Number of independent runs to average over for each precision. Default: 1",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
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

    # Build one FP32 baseline per seed, then run both dtypes from each baseline.
    evaluation_grid = build_evaluation_grid()
    paired_runs = []
    for i in range(args.num_runs):
        seed = SEED + i
        baseline = _build_baseline(seed)
        paired_runs.append(
            {
                "fp32": train_one(
                    torch.float32, seed, args.epochs, baseline, evaluation_grid
                ),
                "fp64": train_one(
                    torch.float64, seed, args.epochs, baseline, evaluation_grid
                ),
            }
        )

    paired_final_total = np.asarray(
        [
            (run["fp32"]["final_total_loss"] + run["fp64"]["final_total_loss"])
            / 2.0
            for run in paired_runs
        ],
        dtype=np.float64,
    )
    representative_run_idx = int(
        np.argsort(paired_final_total)[args.num_runs // 2]
    )
    res32 = run_precision_benchmark(
        torch.float32,
        args.num_runs,
        args.epochs,
        paired_runs=paired_runs,
        evaluation_grid=evaluation_grid,
        representative_run_idx=representative_run_idx,
    )
    res64 = run_precision_benchmark(
        torch.float64,
        args.num_runs,
        args.epochs,
        paired_runs=paired_runs,
        evaluation_grid=evaluation_grid,
        representative_run_idx=representative_run_idx,
    )

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
