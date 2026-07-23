#!/usr/bin/env python3
"""Compare two Burgers benchmark results and generate a report and plot."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_config_burgers import (  # noqa: E402
    BOUNDARY_POINTS,
    DEPTH,
    EPOCHS,
    INTERIOR_POINTS,
    LOSS_CURVES_FILE,
    LR,
    NEW_RESULTS_FILE,
    OLD_RESULTS_FILE,
    REPORT_FILE,
    SEED,
    WIDTH,
    X_RANGE,
)


def _load_results(npz_path: Path, label: str) -> dict:
    if not npz_path.is_file():
        print(f"[ERROR] {label} results not found: {npz_path}")
        sys.exit(1)
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _summary(data: dict) -> dict:
    """Normalize fields from both single-run and multi-run result files."""
    def value(summary_key: str, raw_key: str, default=None):
        if summary_key in data:
            return float(data[summary_key])
        if raw_key in data:
            return float(data[raw_key])
        return default

    return {
        "num_runs": int(data.get("num_runs", 1)),
        "time": value("train_time_mean", "train_time_s"),
        "time_std": value("train_time_std", "train_time_s", 0.0),
        "final": value("final_loss_mean", "final_total_loss"),
        "final_std": value("final_loss_std", "final_total_loss", 0.0),
        "first": value("first_loss_mean", "first_total_loss"),
        "first_std": value("first_loss_std", "first_total_loss", 0.0),
        "commit": str(data["commit_hash"].item()),
        "date": str(data["commit_date"].item()),
    }


def _print_comparison(old: dict, new: dict, epochs: int) -> None:
    old_time_per_epoch = old["time"] / epochs * 1000.0
    new_time_per_epoch = new["time"] / epochs * 1000.0
    speedup = old["time"] / new["time"] if new["time"] > 0 else float("inf")
    rows = [
        ("Version", old["commit"][:7], new["commit"][:7]),
        ("Commit date", old["date"], new["date"]),
        ("Number of runs", str(old["num_runs"]), str(new["num_runs"])),
        (
            "Train time (s)",
            f"{old['time']:.4f} ± {old['time_std']:.4f}",
            f"{new['time']:.4f} ± {new['time_std']:.4f}",
        ),
        (
            "Time per epoch (ms)",
            f"{old_time_per_epoch:.4f}",
            f"{new_time_per_epoch:.4f}",
        ),
        (
            "First total loss",
            f"{old['first']:.6e} ± {old['first_std']:.6e}",
            f"{new['first']:.6e} ± {new['first_std']:.6e}",
        ),
        (
            "Final total loss",
            f"{old['final']:.6e} ± {old['final_std']:.6e}",
            f"{new['final']:.6e} ± {new['final_std']:.6e}",
        ),
        ("Speedup (old / new)", "1.00x", f"{speedup:.2f}x"),
    ]

    print("=" * 70)
    print("DeepFlow 1D Burgers Equation Benchmark - Version Comparison")
    print("=" * 70)
    print(f"{'Metric':<25} {'Old':<22} {'New':<22}")
    print("-" * 70)
    for metric, old_value, new_value in rows:
        print(f"{metric:<25} {old_value:<22} {new_value:<22}")
    print("=" * 70)


def _plot_loss_curves(
    old_data: dict, new_data: dict, old: dict, new: dict, epochs: int
) -> None:
    plt.figure(figsize=(10, 6))
    for data, summary, label in (
        (old_data, old, "Old"),
        (new_data, new, "New"),
    ):
        plt.semilogy(
            np.arange(1, epochs + 1),
            data["total_loss"],
            label=f"{label} total loss ({summary['commit'][:7]})",
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


def _write_report(old: dict, new: dict, epochs: int) -> None:
    old_time_per_epoch = old["time"] / epochs * 1000.0
    new_time_per_epoch = new["time"] / epochs * 1000.0
    speedup = old["time"] / new["time"] if new["time"] > 0 else float("inf")
    report_lines = [
        "# DeepFlow 1D Burgers Equation Benchmark Report",
        "",
        "## Setup",
        "",
        "- **PDE**: 1D Burgers equation,  $u_t + u u_x = \\nu u_{xx}$  with  $\\nu = \\frac{0.01}{\\pi}$",
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
        f"| Commit hash | `{old['commit']}` | `{new['commit']}` |",
        f"| Commit date | {old['date']} | {new['date']} |",
        f"| Number of runs | {old['num_runs']} | {new['num_runs']} |",
        f"| Train time (s) | {old['time']:.4f} ± {old['time_std']:.4f} | {new['time']:.4f} ± {new['time_std']:.4f} |",
        f"| Time per epoch (ms) | {old_time_per_epoch:.4f} | {new_time_per_epoch:.4f} |",
        f"| First total loss | {old['first']:.6e} ± {old['first_std']:.6e} | {new['first']:.6e} ± {new['first_std']:.6e} |",
        f"| Final total loss | {old['final']:.6e} ± {old['final_std']:.6e} | {new['final']:.6e} ± {new['final_std']:.6e} |",
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


def main() -> None:
    old_data = _load_results(OLD_RESULTS_FILE, "Old")
    new_data = _load_results(NEW_RESULTS_FILE, "New")
    old = _summary(old_data)
    new = _summary(new_data)
    epochs = int(old_data["epochs"])
    if int(new_data["epochs"]) != epochs:
        print("[WARNING] Epoch counts differ between old and new results.")

    _print_comparison(old, new, epochs)
    _plot_loss_curves(old_data, new_data, old, new, epochs)
    print(f"Loss curve plot saved to: {LOSS_CURVES_FILE}")
    _write_report(old, new, epochs)
    print(f"Report written to: {REPORT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
