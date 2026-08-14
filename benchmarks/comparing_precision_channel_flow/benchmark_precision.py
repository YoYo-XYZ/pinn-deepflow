#!/usr/bin/env python3
"""
Benchmark: FP32 vs FP64 precision for 2D steady channel flow using DeepFlow.

This mirrors ``benchmarks/comparing_precision/benchmark_precision.py``, but uses
steady Navier-Stokes channel flow instead of Burgers.

The reported quantities measure training losses and PDE residuals; no independent
reference solution is used to measure solution accuracy.

Usage
-----
Run from the repository root::

    python benchmarks/comparing_precision_channel_flow/benchmark_precision.py

For more statistically robust results, average over multiple runs::

    python benchmarks/comparing_precision_channel_flow/benchmark_precision.py --num_runs 5
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
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

import deepflow as df  # noqa: E402
from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    DEPTH,
    EPOCHS,
    EVAL_GRID,
    INTERIOR_POINTS,
    L,
    MU,
    RESULTS_DIR,
    RHO,
    SEED,
    U,
    WIDTH,
    X_RANGE,
    Y_RANGE,
)


RESIDUAL_KEYS = (
    "pde_residual",
    "continuity_residual",
    "x_momentum_residual",
    "y_momentum_residual",
)
AGGREGATE_KEYS = (
    "train_time_s",
    "final_total_loss",
    "final_bc_loss",
    "final_pde_loss",
    "max_pde_residual",
    "mean_abs_pde_residual",
    "max_continuity_residual",
    "mean_abs_continuity_residual",
    "max_x_momentum_residual",
    "mean_abs_x_momentum_residual",
    "max_y_momentum_residual",
    "mean_abs_y_momentum_residual",
)
SUMMARY_ROWS = (
    ("Final total loss", "final_total_loss"),
    ("Final BC loss", "final_bc_loss"),
    ("Final PDE loss", "final_pde_loss"),
    ("Max |PDE residual|", "max_pde_residual"),
    ("Mean |PDE residual|", "mean_abs_pde_residual"),
    ("Max |continuity|", "max_continuity_residual"),
    ("Mean |continuity|", "mean_abs_continuity_residual"),
    ("Max |x-momentum|", "max_x_momentum_residual"),
    ("Mean |x-momentum|", "mean_abs_x_momentum_residual"),
    ("Max |y-momentum|", "max_y_momentum_residual"),
    ("Mean |y-momentum|", "mean_abs_y_momentum_residual"),
    ("Train time (s)", "train_time"),
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
    """Build the steady 2D channel-flow domain (geometry, PDE, BCs)."""
    rect = df.geometry.rectangle(list(X_RANGE), list(Y_RANGE))
    domain = df.domain(rect)

    # Boundary order from rectangle(): left, bottom, right, top.
    domain.bound_list[0].define_bc({"u": 1, "v": 0})
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.bound_list[3].define_bc({"u": 0, "v": 0})

    domain.area_list[0].define_pde(df.pde.NavierStokes(U=U, L=L, mu=MU, rho=RHO))
    if sampled_coordinates is None:
        domain.sampling_random(BOUNDARY_POINTS, INTERIOR_POINTS)
    else:
        _apply_coordinates(domain, sampled_coordinates, df.dtype)
        domain.sampling_option = "paired_baseline"
    return domain


# ---------------------------------------------------------------------------
# Training / evaluation helpers
# ---------------------------------------------------------------------------

def _as_np(values):
    """Convert tensors without erasing their native floating-point dtype."""
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def _history(model, key):
    return _as_np(model.loss_history.get(key, []))


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
        output_vars=["u", "v", "p"],
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
        output_vars=["u", "v", "p"],
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
    """Train one channel-flow PINN with the requested floating-point dtype."""
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

    residuals = {key: _as_np(data[key]) for key in RESIDUAL_KEYS}
    residual_metrics = {
        metric: value
        for key, values in residuals.items()
        for metric, value in (
            (f"max_{key}", float(np.max(np.abs(values)))),
            (f"mean_abs_{key}", float(np.mean(np.abs(values)))),
        )
    }

    return {
        "label": label,
        "dtype": str(dtype),
        "seed": int(seed),
        "train_time_s": float(train_time_s),
        "final_total_loss": float(best_losses["total_loss"].detach().cpu().item()),
        "final_bc_loss": float(best_losses["bc_loss"].detach().cpu().item()),
        "final_pde_loss": float(best_losses["pde_loss"].detach().cpu().item()),
        **residual_metrics,
        "x": _as_np(data["x"]),
        "y": _as_np(data["y"]),
        "u": _as_np(data["u"]),
        "v": _as_np(data["v"]),
        "p": _as_np(data["p"]),
        **residuals,
        "total_loss_history": _history(model, "total_loss"),
        "bc_loss_history": _history(model, "bc_loss"),
        "pde_loss_history": _history(model, "pde_loss"),
    }


def _representative_run_idx(per_run, representative_run_idx=None):
    if representative_run_idx is None:
        final_total = _as_np([run["final_total_loss"] for run in per_run])
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

    aggregates = {}
    for key in AGGREGATE_KEYS:
        values = _as_np([run[key] for run in per_run])
        base = key[:-2] if key.endswith("_s") else key
        aggregates[f"{base}_mean"] = float(values.mean())
        aggregates[f"{base}_std"] = float(values.std(ddof=1)) if num_runs > 1 else 0.0
        aggregates[f"{key}_runs"] = values

    median_idx = _representative_run_idx(per_run, representative_run_idx)
    representative = per_run[median_idx]

    return {
        "label": representative["label"],
        "dtype": representative["dtype"],
        "num_runs": num_runs,
        "epochs": epochs,
        "median_run_idx": median_idx,
        "representative_run_idx": median_idx,
        "representative_seed": representative["seed"],
        "run_seeds": np.asarray([run["seed"] for run in per_run], dtype=np.int64),
        "run_indices": np.arange(num_runs, dtype=np.int64),
        **aggregates,
        **{
            key: representative[key]
            for key in [
                "x",
                "y",
                "u",
                "v",
                "p",
                "pde_residual",
                "continuity_residual",
                "x_momentum_residual",
                "y_momentum_residual",
                "total_loss_history",
                "bc_loss_history",
                "pde_loss_history",
            ]
        },
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
    print("\n" + "=" * 96)
    print("FP32 vs FP64 2D Channel-Flow Precision Benchmark Summary")
    print("=" * 96)
    print(f"{'Metric':<34} {'FP32':>24} {'FP64':>24} {'Delta':>10}")
    print("-" * 96)

    for label, key in SUMMARY_ROWS:
        m32, s32 = res32[f"{key}_mean"], res32[f"{key}_std"]
        m64, s64 = res64[f"{key}_mean"], res64[f"{key}_std"]
        delta = _pct_delta(m32, m64)
        delta_str = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
        print(
            f"{label:<34} "
            f"{_format_mean_std(m32, s32):>24} "
            f"{_format_mean_std(m64, s64):>24} "
            f"{delta_str:>10}"
        )

    print("=" * 96)
    print("Interpretation: negative Delta for training losses/residuals means FP64 is lower/better.")
    print("These metrics measure training and PDE residual behavior, not independent solution accuracy.")
    print("Positive Delta for time means FP64 is slower.")
    print("=" * 96)


def _shared_range(*arrays):
    values = np.concatenate([np.asarray(array).ravel() for array in arrays])
    return float(np.nanmin(values)), float(np.nanmax(values))


def _scatter(ax, x, y, field, title, cmap="jet", vrange=None):
    sc = ax.scatter(x, y, c=field, s=1, cmap=cmap, marker="s")
    if vrange is not None:
        sc.set_clim(vrange)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, shrink=0.8)


def _save_figure(filename, message, draw):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    draw(axes)
    plt.tight_layout()
    path = RESULTS_DIR / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"{message}: {path}")


def _plot_loss_curves(axes, res32, res64):
    for ax, (key, title) in zip(
        axes,
        (
            ("total_loss_history", "Total loss"),
            ("bc_loss_history", "BC loss"),
            ("pde_loss_history", "PDE loss"),
        ),
    ):
        ax.semilogy(res32[key], label="FP32")
        ax.semilogy(res64[key], label="FP64")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, which="both", ls="--", alpha=0.5)


def _plot_field_comparison(axes, res32, res64, field32, field64, titles):
    value_range = _shared_range(field32, field64)
    _scatter(axes[0], res32["x"], res32["y"], field32, titles[0], vrange=value_range)
    _scatter(axes[1], res64["x"], res64["y"], field64, titles[1], vrange=value_range)
    _scatter(
        axes[2],
        res64["x"],
        res64["y"],
        field64 - field32,
        titles[2],
        cmap="RdBu_r",
    )


def _plot_residual_differences(axes, res32, res64):
    for ax, (key, title) in zip(
        axes,
        (
            ("continuity_residual", "|continuity|"),
            ("x_momentum_residual", "|x-momentum|"),
            ("y_momentum_residual", "|y-momentum|"),
        ),
    ):
        _scatter(
            ax,
            res64["x"],
            res64["y"],
            np.abs(res64[key]) - np.abs(res32[key]),
            f"{title} difference (FP64 - FP32)",
            cmap="RdBu_r",
        )


def plot_results(res32, res64):
    """Generate loss, velocity, pressure, and residual comparison figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    _save_figure(
        "loss_curves.png",
        "Loss-curve figure saved to",
        lambda axes: _plot_loss_curves(axes, res32, res64),
    )

    speed32 = np.sqrt(res32["u"] ** 2 + res32["v"] ** 2)
    speed64 = np.sqrt(res64["u"] ** 2 + res64["v"] ** 2)
    _save_figure(
        "velocity_magnitude_comparison.png",
        "Velocity-magnitude figure saved to",
        lambda axes: _plot_field_comparison(
            axes,
            res32,
            res64,
            speed32,
            speed64,
            ("|v| - FP32", "|v| - FP64", "|v| difference (FP64 - FP32)"),
        ),
    )

    _save_figure(
        "pressure_comparison.png",
        "Pressure figure saved to",
        lambda axes: _plot_field_comparison(
            axes,
            res32,
            res64,
            res32["p"],
            res64["p"],
            ("p - FP32", "p - FP64", "p difference (FP64 - FP32)"),
        ),
    )

    _save_figure(
        "residual_difference.png",
        "Residual-difference figure saved to",
        lambda axes: _plot_residual_differences(axes, res32, res64),
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FP32 vs FP64 precision for DeepFlow 2D channel flow."
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

    print("=" * 96)
    print("DeepFlow FP32 vs FP64 2D Channel-Flow Precision Benchmark")
    print("=" * 96)
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

    print_summary(res32, res64)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for res in (res32, res64):
        out_path = RESULTS_DIR / f"{res['label'].lower()}_results.npz"
        np.savez(out_path, **res)
        print(f"Saved {res['label']} results to: {out_path}")

    plot_results(res32, res64)

    better_total = res64["final_total_loss_mean"] < res32["final_total_loss_mean"]
    better_residual = (
        res64["mean_abs_pde_residual_mean"] < res32["mean_abs_pde_residual_mean"]
    )
    print("\nVerdict:")
    print(f"  FP64 total loss is {'LOWER (better)' if better_total else 'HIGHER (worse)'} than FP32.")
    print(
        "  FP64 mean PDE residual is "
        f"{'LOWER (better)' if better_residual else 'HIGHER (worse)'} than FP32."
    )


if __name__ == "__main__":
    main()
