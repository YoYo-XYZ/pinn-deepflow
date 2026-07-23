#!/usr/bin/env python3
"""
Benchmark the 1D Burgers equation with DeepFlow.

The script intentionally uses only the public API shared by the legacy and
current DeepFlow versions.  It can be copied with ``common_config_burgers.py``
to a legacy checkout that does not contain the repository's benchmarks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Resolve local configuration and the source checkout regardless of cwd.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import numpy as np  # noqa: E402
from torch import pi, sin  # noqa: E402

import deepflow as df  # noqa: E402

from common_config_burgers import (  # noqa: E402
    BOUNDARY_POINTS,
    DEPTH,
    EPOCHS,
    INTERIOR_POINTS,
    LR,
    NU,
    RESULTS_DIR,
    SEED,
    WIDTH,
    X_RANGE,
    Y_RANGE,
)


def _get_commit_metadata() -> dict:
    """Return the active repository's git hash and commit date."""
    repo_root = Path(__file__).resolve().parents[2]
    metadata = {"commit_hash": "unknown", "commit_date": "unknown"}
    try:
        metadata["commit_hash"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
        metadata["commit_date"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%ci", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    return metadata


def _build_domain():
    """Create and sample a fresh Burgers domain for one independent run."""
    area = df.geometry.rectangle(list(X_RANGE), list(Y_RANGE))
    initial_condition = df.geometry.line_horizontal(
        y=Y_RANGE[0], range_x=list(X_RANGE)
    )
    left_boundary = df.geometry.line_vertical(
        x=X_RANGE[0], range_y=list(Y_RANGE)
    )
    right_boundary = df.geometry.line_vertical(
        x=X_RANGE[1], range_y=list(Y_RANGE)
    )
    domain = df.domain(
        area.area_list, initial_condition, left_boundary, right_boundary
    )
    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})
    domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)
    return domain


def _run_once(run_idx: int, num_runs: int) -> dict:
    run_seed = SEED + run_idx
    df.manual_seed(run_seed)
    domain = _build_domain()
    model = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=WIDTH,
        length=DEPTH,
    )

    print(f"\n--- Run {run_idx + 1}/{num_runs} (seed {run_seed}) ---")
    start = time.perf_counter()
    model, _ = model.train_adam(
        calc_loss=df.calc_loss_simple(domain),
        learning_rate=LR,
        epochs=EPOCHS,
        print_every=200,
    )
    elapsed = time.perf_counter() - start
    history = model.loss_history
    return {
        "time": elapsed,
        "first_loss": float(history["total_loss"][0]),
        "final_loss": float(history["total_loss"][-1]),
        "total_loss": np.asarray(history["total_loss"], dtype=np.float64),
        "bc_loss": np.asarray(history["bc_loss"], dtype=np.float64),
        "pde_loss": np.asarray(history["pde_loss"], dtype=np.float64),
    }


def _mean_and_std(values: np.ndarray) -> tuple[float, float]:
    """Return mean and sample standard deviation for one or more runs."""
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0


def _aggregate_results(runs: list[dict], commit_metadata: dict) -> dict:
    times = np.asarray([run["time"] for run in runs], dtype=np.float64)
    final_losses = np.asarray(
        [run["final_loss"] for run in runs], dtype=np.float64
    )
    first_losses = np.asarray(
        [run["first_loss"] for run in runs], dtype=np.float64
    )
    median_idx = int(np.argsort(final_losses)[len(runs) // 2])
    time_mean, time_std = _mean_and_std(times)
    final_mean, final_std = _mean_and_std(final_losses)
    first_mean, first_std = _mean_and_std(first_losses)
    median_run = runs[median_idx]

    return {
        "train_time_s": times,
        "final_total_loss": final_losses,
        "first_total_loss": first_losses,
        "total_loss": median_run["total_loss"],
        "bc_loss": median_run["bc_loss"],
        "pde_loss": median_run["pde_loss"],
        "train_time_mean": time_mean,
        "train_time_std": time_std,
        "final_loss_mean": final_mean,
        "final_loss_std": final_std,
        "first_loss_mean": first_mean,
        "first_loss_std": first_std,
        "num_runs": len(runs),
        "median_run_idx": median_idx,
        "commit_hash": commit_metadata["commit_hash"],
        "commit_date": commit_metadata["commit_date"],
        "epochs": EPOCHS,
        "width": WIDTH,
        "depth": DEPTH,
        "lr": LR,
        "seed": SEED,
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 1D Burgers equation with DeepFlow."
    )
    parser.add_argument(
        "--num_runs",
        type=int,
        default=1,
        help="Number of independent training runs to average over. Default: 1",
    )
    args = parser.parse_args()
    if args.num_runs < 1:
        parser.error("--num_runs must be at least 1")

    print("=" * 60)
    print(f"DeepFlow  -  1D Burgers Equation Benchmark  ({args.num_runs} run(s))")
    print("=" * 60)
    print(f"Device: {df.device}")
    print(f"Base seed: {SEED}")
    print(f"Width: {WIDTH}, Depth: {DEPTH}, Epochs: {EPOCHS}, LR: {LR}")

    results = _aggregate_results(
        [_run_once(run_idx, args.num_runs) for run_idx in range(args.num_runs)],
        _get_commit_metadata(),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / (
        f"burgers_benchmark_{results['commit_hash'][:7]}.npz"
    )
    np.savez(output_path, **results)

    print("\n" + "=" * 60)
    print("Benchmark complete")
    print("=" * 60)
    print(f"Commit:            {results['commit_hash']}")
    print(f"Commit date:       {results['commit_date']}")
    print(f"Number of runs:    {results['num_runs']}")
    print(
        f"Train time (s):    {results['train_time_mean']:.4f} "
        f"± {results['train_time_std']:.4f}"
    )
    print(
        f"First total loss:  {results['first_loss_mean']:.6e} "
        f"± {results['first_loss_std']:.6e}"
    )
    print(
        f"Final total loss:  {results['final_loss_mean']:.6e} "
        f"± {results['final_loss_std']:.6e}"
    )
    print(f"Median run index:  {results['median_run_idx'] + 1}")
    print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
