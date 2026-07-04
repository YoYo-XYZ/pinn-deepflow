#!/usr/bin/env python3
"""
Benchmark: 1D Burgers equation using DeepFlow.

This script uses only the public DeepFlow API that is stable between the
origin/dev baseline (dd3efd08) and the current dev HEAD. It times a fixed
Adam-only training run, records the loss history, and saves the results to
an NPZ file for later comparison between framework versions.

Usage
-----
Run from the repository root on the version you want to benchmark::

    python benchmarks/burgers_eq/benchmark_burgers.py

To run against an older checkout that does not contain the ``benchmarks/``
directory, copy this file together with ``common_config_burgers.py`` to an
external location and execute from there.
"""

import os
import sys
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Parse command line arguments
# ---------------------------------------------------------------------------
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
NUM_RUNS = args.num_runs

# ---------------------------------------------------------------------------
# Ensure imports resolve regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# Ensure we can resolve the ``deepflow`` package from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
from torch import sin, pi

import deepflow as df

from common_config_burgers import (
    X_RANGE,
    Y_RANGE,
    NU,
    WIDTH,
    DEPTH,
    LR,
    EPOCHS,
    SEED,
    BOUNDARY_POINTS,
    INTERIOR_POINTS,
    RESULTS_DIR,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
df.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Commit metadata
# ---------------------------------------------------------------------------
def _get_commit_metadata() -> dict:
    """Return the current git hash and committer date from the active repo."""
    cwd = Path(__file__).resolve().parents[2]
    meta = {
        "commit_hash": "unknown",
        "commit_date": "unknown",
    }
    try:
        meta["commit_hash"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("utf-8")
            .strip()
        )
        meta["commit_date"] = (
            subprocess.check_output(
                ["git", "log", "-1", "--format=%ci", "HEAD"], cwd=cwd
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass
    return meta


# ---------------------------------------------------------------------------
# Build the problem (geometry/PDE/BC are shared across runs)
# ---------------------------------------------------------------------------
print("=" * 60)
print(f"DeepFlow  -  1D Burgers Equation Benchmark  ({NUM_RUNS} run(s))")
print("=" * 60)
print(f"Device: {df.device}")
print(f"Base seed: {SEED}")
print(f"Width: {WIDTH}, Depth: {DEPTH}, Epochs: {EPOCHS}, LR: {LR}")

area = df.geometry.rectangle(list(X_RANGE), list(Y_RANGE))
line_ic = df.geometry.line_horizontal(y=Y_RANGE[0], range_x=list(X_RANGE))
line_bc1 = df.geometry.line_vertical(x=X_RANGE[0], range_y=list(Y_RANGE))
line_bc2 = df.geometry.line_vertical(x=X_RANGE[1], range_y=list(Y_RANGE))
base_domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)

base_domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
base_domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
base_domain.bound_list[1].define_bc({"u": 0})
base_domain.bound_list[2].define_bc({"u": 0})

# ---------------------------------------------------------------------------
# Run multiple independent training runs
# ---------------------------------------------------------------------------
per_run_times = []
per_run_final_losses = []
per_run_first_losses = []
per_run_total_losses = []
per_run_bc_losses = []
per_run_pde_losses = []

for run_idx in range(NUM_RUNS):
    run_seed = SEED + run_idx
    df.manual_seed(run_seed)

    # Fresh domain/model for each run so weights and samples are independent
    domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)
    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})
    domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)

    model0 = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=WIDTH,
        length=DEPTH,
    )

    calc_loss = df.calc_loss_simple(domain)

    print(f"\n--- Run {run_idx + 1}/{NUM_RUNS} (seed {run_seed}) ---")
    t_start = time.perf_counter()
    model, model_best = model0.train_adam(
        calc_loss=calc_loss,
        learning_rate=LR,
        epochs=EPOCHS,
        print_every=200,
    )
    train_time_s = time.perf_counter() - t_start

    per_run_times.append(train_time_s)
    per_run_final_losses.append(float(model.loss_history["total_loss"][-1]))
    per_run_first_losses.append(float(model.loss_history["total_loss"][0]))
    per_run_total_losses.append(
        np.asarray(model.loss_history["total_loss"], dtype=np.float64)
    )
    per_run_bc_losses.append(
        np.asarray(model.loss_history["bc_loss"], dtype=np.float64)
    )
    per_run_pde_losses.append(
        np.asarray(model.loss_history["pde_loss"], dtype=np.float64)
    )

# ---------------------------------------------------------------------------
# Aggregate results
# ---------------------------------------------------------------------------
per_run_times_arr = np.asarray(per_run_times, dtype=np.float64)
per_run_final_losses_arr = np.asarray(per_run_final_losses, dtype=np.float64)
per_run_first_losses_arr = np.asarray(per_run_first_losses, dtype=np.float64)

train_time_mean = float(per_run_times_arr.mean())
train_time_std = float(per_run_times_arr.std(ddof=1)) if NUM_RUNS > 1 else 0.0
final_loss_mean = float(per_run_final_losses_arr.mean())
final_loss_std = float(per_run_final_losses_arr.std(ddof=1)) if NUM_RUNS > 1 else 0.0
first_loss_mean = float(per_run_first_losses_arr.mean())
first_loss_std = float(per_run_first_losses_arr.std(ddof=1)) if NUM_RUNS > 1 else 0.0

commit_meta = _get_commit_metadata()

# Use the median-loss run for representative loss curves
median_idx = int(np.argsort(per_run_final_losses_arr)[len(per_run_final_losses_arr) // 2])

results = {
    "train_time_s": per_run_times_arr,
    "final_total_loss": per_run_final_losses_arr,
    "first_total_loss": per_run_first_losses_arr,
    "total_loss": per_run_total_losses[median_idx],
    "bc_loss": per_run_bc_losses[median_idx],
    "pde_loss": per_run_pde_losses[median_idx],
    "train_time_mean": train_time_mean,
    "train_time_std": train_time_std,
    "final_loss_mean": final_loss_mean,
    "final_loss_std": final_loss_std,
    "first_loss_mean": first_loss_mean,
    "first_loss_std": first_loss_std,
    "num_runs": NUM_RUNS,
    "median_run_idx": median_idx,
    "commit_hash": commit_meta["commit_hash"],
    "commit_date": commit_meta["commit_date"],
    "epochs": EPOCHS,
    "width": WIDTH,
    "depth": DEPTH,
    "lr": LR,
    "seed": SEED,
    "run_timestamp": datetime.now(timezone.utc).isoformat(),
}

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
output_name = f"burgers_benchmark_{commit_meta['commit_hash'][:7]}.npz"
output_path = RESULTS_DIR / output_name
np.savez(output_path, **results)

print("\n" + "=" * 60)
print("Benchmark complete")
print("=" * 60)
print(f"Commit:            {commit_meta['commit_hash']}")
print(f"Commit date:       {commit_meta['commit_date']}")
print(f"Number of runs:    {NUM_RUNS}")
print(f"Train time (s):    {train_time_mean:.4f} ± {train_time_std:.4f}")
print(f"First total loss:  {first_loss_mean:.6e} ± {first_loss_std:.6e}")
print(f"Final total loss:  {final_loss_mean:.6e} ± {final_loss_std:.6e}")
print(f"Median run index:  {median_idx + 1}")
print(f"Results written to: {output_path}")
