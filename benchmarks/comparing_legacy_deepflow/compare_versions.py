#!/usr/bin/env python3
"""Compare two Burgers benchmark results and generate a report and plot."""

from __future__ import annotations

import argparse
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
    Y_RANGE,
)


REQUIRED_METADATA = (
    "commit_hash",
    "commit_date",
    "epochs",
    "width",
    "depth",
    "lr",
    "seed",
)
CONFIG_METADATA = ("epochs", "width", "depth", "lr", "seed")


def _load_results(npz_path: Path, label: str) -> dict:
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"{label} results not found: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _metadata_scalar(data: dict, key: str, label: str):
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise ValueError(f"{label} metadata '{key}' must be scalar")
    return value.item()


def _metadata_text(data: dict, key: str, default: str) -> str:
    if key not in data:
        return default
    return str(_metadata_scalar(data, key, "Result"))


def _validate_results(data: dict, label: str) -> None:
    missing = [key for key in REQUIRED_METADATA if key not in data]
    if missing:
        raise ValueError(
            f"{label} results are missing required metadata: {', '.join(missing)}"
        )
    if "total_loss" not in data:
        raise ValueError(f"{label} results are missing required field: total_loss")

    epochs = int(_metadata_scalar(data, "epochs", label))
    if epochs < 1:
        raise ValueError(f"{label} metadata 'epochs' must be at least 1")
    total_loss = np.asarray(data["total_loss"])
    if total_loss.ndim != 1 or len(total_loss) != epochs:
        raise ValueError(
            f"{label} total_loss has length {len(total_loss)}; expected {epochs}"
        )


def _validate_config(old_data: dict, new_data: dict) -> int:
    for key in CONFIG_METADATA:
        old_value = _metadata_scalar(old_data, key, "Old")
        new_value = _metadata_scalar(new_data, key, "New")
        if key == "lr":
            matches = np.isclose(float(old_value), float(new_value))
        else:
            matches = old_value == new_value
        if not matches:
            raise ValueError(
                f"Configuration mismatch for '{key}': "
                f"old={old_value!r}, new={new_value!r}"
            )

    epochs = int(_metadata_scalar(old_data, "epochs", "Old"))
    new_epochs = int(_metadata_scalar(new_data, "epochs", "New"))
    if new_epochs != epochs:
        raise ValueError(
            f"Configuration mismatch for 'epochs': old={epochs}, new={new_epochs}"
        )
    return epochs


def _numeric_summary(
    data: dict, summary_key: str, raw_key: str, statistic: str, default=None
):
    if summary_key in data:
        value = np.asarray(data[summary_key])
        if value.size != 1:
            raise ValueError(f"Result field '{summary_key}' must contain one value")
        return float(value.reshape(-1)[0])
    if raw_key not in data:
        return default

    values = np.asarray(data[raw_key], dtype=np.float64).reshape(-1)
    if values.size == 0:
        return default
    if statistic == "mean":
        return float(values.mean())
    if statistic == "std":
        return float(values.std(ddof=1)) if values.size > 1 else 0.0
    raise ValueError(f"Unknown result statistic: {statistic}")


def _summary(data: dict) -> dict:
    """Normalize fields from both single-run and multi-run result files."""
    return {
        "num_runs": int(_metadata_scalar(data, "num_runs", "Result"))
        if "num_runs" in data
        else 1,
        "time": _numeric_summary(data, "train_time_mean", "train_time_s", "mean"),
        "time_std": _numeric_summary(
            data, "train_time_std", "train_time_s", "std", 0.0
        ),
        "final": _numeric_summary(
            data, "final_loss_mean", "final_total_loss", "mean"
        ),
        "final_std": _numeric_summary(
            data, "final_loss_std", "final_total_loss", "std", 0.0
        ),
        "first": _numeric_summary(
            data, "first_loss_mean", "first_total_loss", "mean"
        ),
        "first_std": _numeric_summary(
            data, "first_loss_std", "first_total_loss", "std", 0.0
        ),
        "commit": _metadata_text(data, "commit_hash", "unknown"),
        "date": _metadata_text(data, "commit_date", "unknown"),
        "initialization_protocol": _metadata_text(
            data, "initialization_protocol", "not recorded"
        ),
    }


def _print_comparison(old: dict, new: dict, epochs: int) -> None:
    old_time_per_epoch = old["time"] / epochs * 1000.0
    new_time_per_epoch = new["time"] / epochs * 1000.0
    speedup = old["time"] / new["time"] if new["time"] > 0 else float("inf")
    rows = [
        ("Version", old["commit"][:7], new["commit"][:7]),
        ("Commit date", old["date"], new["date"]),
        (
            "Initialization protocol",
            old["initialization_protocol"],
            new["initialization_protocol"],
        ),
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
        "- **Coordinates**: `x` is the spatial coordinate; `y` is time ($t$ in the PDE notation).",
        f"- **Domain**:  $x \\in [{X_RANGE[0]}, {X_RANGE[1]}]$,  $y=t \\in [{Y_RANGE[0]}, {Y_RANGE[1]}]$",
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
        f"| Initialization protocol | {old['initialization_protocol']} | {new['initialization_protocol']} |",
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
    parser = argparse.ArgumentParser(
        description="Compare two Burgers benchmark results and generate a report."
    )
    parser.add_argument(
        "--old",
        type=Path,
        default=OLD_RESULTS_FILE,
        help=f"Old result file (default: {OLD_RESULTS_FILE})",
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=NEW_RESULTS_FILE,
        help=f"New result file (default: {NEW_RESULTS_FILE})",
    )
    args = parser.parse_args()

    try:
        old_data = _load_results(args.old, "Old")
        new_data = _load_results(args.new, "New")
        _validate_results(old_data, "Old")
        _validate_results(new_data, "New")
        epochs = _validate_config(old_data, new_data)
        old = _summary(old_data)
        new = _summary(new_data)
        for label, summary in (("Old", old), ("New", new)):
            for field in ("time", "final", "first"):
                if summary[field] is None:
                    raise ValueError(
                        f"{label} results do not contain required {field} data"
                    )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    _print_comparison(old, new, epochs)
    _plot_loss_curves(old_data, new_data, old, new, epochs)
    print(f"Loss curve plot saved to: {LOSS_CURVES_FILE}")
    _write_report(old, new, epochs)
    print(f"Report written to: {REPORT_FILE}")
    print("Done.")


if __name__ == "__main__":
    main()
