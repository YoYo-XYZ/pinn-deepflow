#!/usr/bin/env python3
"""
Compare two DeepFlow Burgers-equation benchmark results and generate a report.

Loads the NPZ files produced by ``benchmark_burgers.py`` for the old and new
versions of the framework, prints a comparison table, plots the loss curves,
and writes a Markdown report.

Usage
-----
After running ``benchmark_burgers.py`` on both versions (and renaming the
result files to ``burgers_benchmark_old.npz`` and ``burgers_benchmark_new.npz``
inside ``results/``), execute::

    python benchmarks/burgers_eq/compare_versions.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Ensure imports resolve regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from common_config_burgers import (
    X_RANGE,
    WIDTH,
    DEPTH,
    LR,
    EPOCHS,
    SEED,
    BOUNDARY_POINTS,
    INTERIOR_POINTS,
    OLD_RESULTS_FILE,
    NEW_RESULTS_FILE,
    REPORT_FILE,
    LOSS_CURVES_FILE,
)


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def _load_results(npz_path: Path, label: str) -> Dict[str, Any]:
    if not npz_path.is_file():
        print(f"[ERROR] {label} results not found: {npz_path}")
        sys.exit(1)
    data = np.load(npz_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
old_data = _load_results(OLD_RESULTS_FILE, "Old")
new_data = _load_results(NEW_RESULTS_FILE, "New")

old_num_runs = int(old_data.get("num_runs", 1))
new_num_runs = int(new_data.get("num_runs", 1))

old_time = float(old_data["train_time_mean"]) if "train_time_mean" in old_data else float(old_data["train_time_s"])
new_time = float(new_data["train_time_mean"]) if "train_time_mean" in new_data else float(new_data["train_time_s"])
old_time_std = float(old_data["train_time_std"]) if "train_time_std" in old_data else 0.0
new_time_std = float(new_data["train_time_std"]) if "train_time_std" in new_data else 0.0

old_final = float(old_data["final_loss_mean"]) if "final_loss_mean" in old_data else float(old_data["final_total_loss"])
new_final = float(new_data["final_loss_mean"]) if "final_loss_mean" in new_data else float(new_data["final_total_loss"])
old_final_std = float(old_data["final_loss_std"]) if "final_loss_std" in old_data else 0.0
new_final_std = float(new_data["final_loss_std"]) if "final_loss_std" in new_data else 0.0

old_first = float(old_data["first_loss_mean"]) if "first_loss_mean" in old_data else float(old_data["first_total_loss"])
new_first = float(new_data["first_loss_mean"]) if "first_loss_mean" in new_data else float(new_data["first_total_loss"])
old_first_std = float(old_data["first_loss_std"]) if "first_loss_std" in old_data else 0.0
new_first_std = float(new_data["first_loss_std"]) if "first_loss_std" in new_data else 0.0

old_commit = str(old_data["commit_hash"].item())
new_commit = str(new_data["commit_hash"].item())
old_date = str(old_data["commit_date"].item())
new_date = str(new_data["commit_date"].item())


epochs = int(old_data["epochs"])
if int(new_data["epochs"]) != epochs:
    print("[WARNING] Epoch counts differ between old and new results.")

time_per_epoch_old = old_time / epochs * 1000.0  # ms
time_per_epoch_new = new_time / epochs * 1000.0  # ms
speedup = old_time / new_time if new_time > 0 else float("inf")


# ---------------------------------------------------------------------------
# Print comparison table
# ---------------------------------------------------------------------------
def _fmt_float(value: float, fmt: str = ".4f") -> str:
    return f"{value:{fmt}}"


print("=" * 70)
print("DeepFlow 1D Burgers Equation Benchmark - Version Comparison")
print("=" * 70)

rows = [
    ("Version", old_commit[:7], new_commit[:7]),
    ("Commit date", old_date, new_date),
    ("Number of runs", str(old_num_runs), str(new_num_runs)),
    ("Train time (s)", f"{old_time:.4f} ± {old_time_std:.4f}", f"{new_time:.4f} ± {new_time_std:.4f}"),
    ("Time per epoch (ms)", f"{time_per_epoch_old:.4f}", f"{time_per_epoch_new:.4f}"),
    ("First total loss", f"{old_first:.6e} ± {old_first_std:.6e}", f"{new_first:.6e} ± {new_first_std:.6e}"),
    ("Final total loss", f"{old_final:.6e} ± {old_final_std:.6e}", f"{new_final:.6e} ± {new_final_std:.6e}"),
    ("Speedup (old / new)", "1.00x", f"{speedup:.2f}x"),
]

print(f"{'Metric':<25} {'Old':<22} {'New':<22}")
print("-" * 70)
for metric, old_val, new_val in rows:
    print(f"{metric:<25} {old_val:<22} {new_val:<22}")
print("=" * 70)


# ---------------------------------------------------------------------------
# Plot loss curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(10, 6))

plt.semilogy(
    np.arange(1, epochs + 1),
    old_data["total_loss"],
    label=f"Old total loss ({old_commit[:7]})",
    linewidth=2,
)
plt.semilogy(
    np.arange(1, epochs + 1),
    new_data["total_loss"],
    label=f"New total loss ({new_commit[:7]})",
    linewidth=2,
)

plt.xlabel("Epoch")
plt.ylabel("Total loss")
plt.title("DeepFlow 1D Burgers Equation - Loss Curves")
plt.legend()
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(LOSS_CURVES_FILE, dpi=150)
plt.close()

print(f"Loss curve plot saved to: {LOSS_CURVES_FILE}")


# ---------------------------------------------------------------------------
# Write Markdown report
# ---------------------------------------------------------------------------
report_lines = [
    "# DeepFlow 1D Burgers Equation Benchmark Report",
    "",
    "## Setup",
    "",
    f"- **PDE**: 1D Burgers equation,  $u_t + u u_x = \\nu u_{{xx}}$  with  $\\nu = \\frac{{0.01}}{{\\pi}}$",
    f"- **Spatial domain**:  $x \\in [{X_RANGE[0]}, {X_RANGE[1]}]$",
    f"- **Network**:  {WIDTH}x{DEPTH} fully-connected network with Tanh activation",
    f"- **Optimizer**: Adam, learning rate = {LR}",
    f"- **Epochs**: {EPOCHS}",
    f"- **Seed**: {SEED}",
    f"- **Boundary sampling**: {BOUNDARY_POINTS}",
    f"- **Interior sampling**: {INTERIOR_POINTS}",
    "",
    "## Results",
    "",
    "| Metric | Old | New |",
    "|---|---|---|",
    f"| Commit hash | `{old_commit}` | `{new_commit}` |",
    f"| Commit date | {old_date} | {new_date} |",
    f"| Number of runs | {old_num_runs} | {new_num_runs} |",
    f"| Train time (s) | {old_time:.4f} ± {old_time_std:.4f} | {new_time:.4f} ± {new_time_std:.4f} |",
    f"| Time per epoch (ms) | {time_per_epoch_old:.4f} | {time_per_epoch_new:.4f} |",
    f"| First total loss | {old_first:.6e} ± {old_first_std:.6e} | {new_first:.6e} ± {new_first_std:.6e} |",
    f"| Final total loss | {old_final:.6e} ± {old_final_std:.6e} | {new_final:.6e} ± {new_final_std:.6e} |",
    f"| Speedup (old / new) | 1.00x | {speedup:.2f}x |",
    "",
    "## Loss curves",
    "",
    f"![Loss curves]({LOSS_CURVES_FILE.name})",
    "",
    "## Notes",
    "",
    "- Both versions used the same public DeepFlow API (`df.geometry.rectangle`, `df.pde.BurgersEquation1D`,",
    "  `df.calc_loss_simple`, `df.PINN`, `model.train_adam`).",
    "- The implementation of `calc_loss_simple` differs between these versions: the old version evaluates",
    "  geometry losses in a loop, while the new version uses a batched forward pass. Small differences in final",
    "  loss are expected due to this and to floating-point accumulation order.",
    "- The old `train_adam` copies the full model every epoch to track the best model; the new version caches",
    "  `state_dict` instead, which can itself reduce overhead.",
    "- To improve reliability, consider repeating each run 3–5 times and reporting mean ± standard deviation.",
    "",
]

REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
REPORT_FILE.write_text("\n".join(report_lines), encoding="utf-8")

print(f"Report written to: {REPORT_FILE}")
print("Done.")
