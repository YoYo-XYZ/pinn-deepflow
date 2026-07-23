#!/usr/bin/env python3
"""Compare saved DeepFlow and DeepXDE channel-flow benchmark results."""

import sys

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from benchmark_common import RESULTS_DIR
from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    DEPTH,
    EPOCHS,
    INTERIOR_POINTS,
    Lx,
    Ly,
    LR,
    Re,
    WIDTH,
)

try:
    _trapz = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _trapz = np.trapz

METHODS = (("DeepFlow", "C0"), ("DeepXDE", "C1"))
RESIDUAL_FIELDS = (
    "continuity_residual",
    "x_momentum_residual",
    "y_momentum_residual",
)
METRIC_ROWS = (
    ("Train time", "train_time_s", "{:.2f}", "s"),
    ("Final total loss", "final_total_loss", "{:.6e}", "–"),
    ("Best train loss", "best_loss_train", "{:.6e}", "–"),
    ("Best test loss", "best_loss_test", "{:.6e}", "–"),
    ("Mean |u|", "mean_u", "{:.6f}", "–"),
    ("Mean |v|", "mean_v", "{:.6f}", "–"),
    ("Max continuity|", "max_continuity_residual", "{:.6e}", "–"),
    ("Max x-momentum|", "max_x_momentum_residual", "{:.6e}", "–"),
    ("Max y-momentum|", "max_y_momentum_residual", "{:.6e}", "–"),
    ("Mass flux inlet", "mass_flux_inlet", "{:.6f}", "–"),
    ("Mass flux outlet", "mass_flux_outlet", "{:.6f}", "–"),
    ("Mass flux rel.err", "mass_flux_rel_error", "{:.6e}", "–"),
)


def _load_results():
    """Load available NPZ files into plain dictionaries."""
    results = {}
    for label, _ in METHODS:
        path = RESULTS_DIR / f"{label.lower()}_results.npz"
        if not path.is_file():
            print(f"[WARN] {path} not found.")
            results[label] = None
            continue
        with np.load(path) as archive:
            results[label] = {key: archive[key] for key in archive.files}
    return results


def _array(data, key):
    if data is None or key not in data:
        return np.array([])
    return np.asarray(data[key])


def _scalar(data, key):
    values = _array(data, key)
    return float(values.flat[0]) if values.size else float("nan")


def _max_abs(values):
    return float(np.max(np.abs(values))) if len(values) else float("nan")


def _mass_flux(data, x_target):
    x = _array(data, "x")
    y = _array(data, "y")
    u = _array(data, "u")
    mask = np.abs(x - x_target) < 0.01
    if not np.any(mask):
        return float("nan")
    order = np.argsort(y[mask])
    return float(_trapz(u[mask][order], y[mask][order]))


def _metrics(results):
    metrics = {}
    for label, _ in METHODS:
        data = results[label]
        if data is None:
            metrics[label] = {}
            continue
        entry = {
            key: _scalar(data, key)
            for key in ("train_time_s", "final_total_loss", "best_loss_train", "best_loss_test")
        }
        entry["mean_u"] = float(np.mean(np.abs(_array(data, "u"))))
        entry["mean_v"] = float(np.mean(np.abs(_array(data, "v"))))
        entry.update({f"max_{key}": _max_abs(_array(data, key)) for key in RESIDUAL_FIELDS})

        inlet = _mass_flux(data, 0.0)
        outlet = _mass_flux(data, Lx)
        entry.update(
            mass_flux_inlet=inlet,
            mass_flux_outlet=outlet,
            mass_flux_rel_error=(
                abs(inlet - outlet) / abs(inlet)
                if np.isfinite(inlet) and np.isfinite(outlet) and abs(inlet) > 1e-15
                else float("nan")
            ),
        )
        metrics[label] = entry
    return metrics


def _format_value(value, format_spec):
    return "N/A" if not np.isfinite(value) else format_spec.format(value)


def _print_summary(metrics):
    print("=" * 100)
    print("Benchmark Comparison: DeepFlow vs DeepXDE (2D Steady Channel Flow)")
    print("=" * 100)
    header = f"{'Metric':<40} {'DeepFlow':>20} {'DeepXDE':>20} {'Unit':>15}"
    separator = "-" * len(header)
    print(separator)
    print(header)
    print(separator)
    for description, key, format_spec, unit in METRIC_ROWS:
        values = [
            _format_value(metrics[label].get(key, float("nan")), format_spec)
            for label, _ in METHODS
        ]
        print(f"{description:<40} {values[0]:>20} {values[1]:>20} {unit:>15}")
    print(separator)


def _field_data(data, key):
    return _array(data, "x"), _array(data, "y"), _array(data, key)


def _display_field(data, key, absolute=False):
    if key == "velocity_magnitude":
        _, _, u = _field_data(data, "u")
        _, _, v = _field_data(data, "v")
        values = np.sqrt(u**2 + v**2)
    else:
        values = _field_data(data, key)[2]
    return np.abs(values) if absolute else values


def _shared_range(*arrays):
    values = [np.asarray(array).ravel() for array in arrays if array is not None]
    finite = np.concatenate(values) if values else np.array([])
    finite = finite[np.isfinite(finite)]
    return (float(finite.min()), float(finite.max())) if len(finite) else None


def _scatter(ax, x, y, values, title, cmap, value_range):
    scatter = ax.scatter(x, y, c=values, s=1, cmap=cmap, marker="s")
    if value_range is not None:
        scatter.set_clim(value_range)
    ax.set(title=title, xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, shrink=0.8)


def _plot_field(results, key, title, filename, cmap="viridis", absolute=False):
    fields = {
        label: _display_field(results[label], key, absolute)
        if results[label] is not None
        else None
        for label, _ in METHODS
    }
    value_range = _shared_range(*fields.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (label, _) in zip(axes, METHODS):
        data = results[label]
        if data is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        x, y, _ = _field_data(data, "u")
        _scatter(ax, x, y, fields[label], f"{title} – {label}", cmap, value_range)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def _profile(data, coordinate, target):
    x = _array(data, "x")
    y = _array(data, "y")
    u = _array(data, "u")
    coordinate_values = x if coordinate == "x" else y
    profile_values = y if coordinate == "x" else x
    if not len(coordinate_values):
        return np.array([]), np.array([])
    nearest_index = int(np.argmin(np.abs(coordinate_values - target)))
    nearest_coordinate = coordinate_values[nearest_index]
    indices = np.where(
        np.isclose(coordinate_values, nearest_coordinate, rtol=0.0, atol=1e-12)
    )[0]
    if not len(indices):
        return np.array([]), np.array([])
    order = np.argsort(profile_values[indices])
    return profile_values[indices][order], u[indices][order]


def _plot_profile(results, coordinate, target, filename, title, xlabel, ylabel):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in METHODS:
        data = results[label]
        if data is None:
            continue
        coordinate_values, u_values = _profile(data, coordinate, target)
        if len(coordinate_values):
            if coordinate == "x":
                ax.plot(u_values, coordinate_values, label=label, color=color)
            else:
                ax.plot(coordinate_values, u_values, label=label, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def _plot_loss(results):
    fig, ax = plt.subplots(figsize=(8, 5))
    deepflow_loss = _array(results["DeepFlow"], "total_loss")
    if len(deepflow_loss):
        ax.semilogy(deepflow_loss, label="DeepFlow total", color="C0")
    deepxde_loss = _array(results["DeepXDE"], "loss_train")
    if len(deepxde_loss):
        steps = _array(results["DeepXDE"], "loss_steps")
        x = steps if len(steps) == len(deepxde_loss) else np.linspace(0, EPOCHS, len(deepxde_loss))
        ax.semilogy(x, deepxde_loss, label="DeepXDE train", color="C1", linestyle="--")
    ax.set(title="Training Loss Curves", xlabel="Iteration", ylabel="Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "compare_loss_curves.png", dpi=150)
    plt.close(fig)


def _write_report(metrics):
    lines = [
        "# Benchmark Report: DeepFlow vs DeepXDE",
        "",
        "## 2D Steady Channel Flow",
        "",
        f"- **Geometry**: rectangle [0, {Lx}] × [0, {Ly}]",
        f"- **Reynolds number**: Re = {Re}",
        f"- **Network**: input=2, output=3 (u,v,p), {WIDTH}×{DEPTH} Tanh",
        f"- **Optimizer**: Adam, lr={LR}, {EPOCHS} iterations",
        f"- **Sampling**: boundary {sum(BOUNDARY_POINTS)} points, interior {INTERIOR_POINTS} points",
        "",
        "## Summary Table",
        "",
        "| Metric | DeepFlow | DeepXDE | Unit |",
        "|--------|----------|---------|------|",
    ]
    for description, key, format_spec, unit in METRIC_ROWS:
        values = [
            _format_value(metrics[label].get(key, float("nan")), format_spec)
            for label, _ in METHODS
        ]
        lines.append(f"| {description} | {values[0]} | {values[1]} | {unit} |")
    lines.extend(
        [
            "",
            "## Generated Figures",
            "",
            "The following comparison plots have been saved to `results/`:",
            "",
            "- `compare_u_field.png` — u velocity field side-by-side",
            "- `compare_v_field.png` — v velocity field side-by-side",
            "- `compare_p_field.png` — pressure field side-by-side",
            "- `compare_velocity_magnitude.png` — |U| field side-by-side",
            "- `compare_continuity_residual.png` — |continuity residual| field",
            "- `compare_x_momentum_residual.png` — |x-momentum residual| field",
            "- `compare_loss_curves.png` — training loss curves",
            "- `compare_profile_u_y_at_x2.5.png` — u(y) profile at x = 2.5",
            "- `compare_profile_u_x_at_y0.5.png` — u(x) profile at y = 0.5",
            "",
            "---",
            "*Report generated by `compare.py`*",
        ]
    )
    report_path = RESULTS_DIR / "benchmark_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Report saved to {report_path}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = _load_results()
    if all(results[label] is None for label, _ in METHODS):
        print("Neither DeepFlow nor DeepXDE results found. Run the benchmarks first.")
        return 1

    metrics = _metrics(results)
    _print_summary(metrics)
    print("\nGenerating comparison plots ...")
    _plot_field(results, "u", "u", "compare_u_field.png", cmap="jet")
    _plot_field(results, "v", "v", "compare_v_field.png", cmap="jet")
    _plot_field(results, "p", "p", "compare_p_field.png", cmap="jet")
    _plot_field(results, "velocity_magnitude", "|U|", "compare_velocity_magnitude.png", cmap="jet")
    _plot_field(results, "continuity_residual", "|Continuity|", "compare_continuity_residual.png", cmap="hot", absolute=True)
    _plot_field(results, "x_momentum_residual", "|x-Momentum|", "compare_x_momentum_residual.png", cmap="hot", absolute=True)
    _plot_loss(results)
    _plot_profile(
        results,
        "x",
        2.5,
        "compare_profile_u_y_at_x2.5.png",
        "u(y) at x = 2.5",
        "u",
        "y",
    )
    _plot_profile(
        results,
        "y",
        0.5,
        "compare_profile_u_x_at_y0.5.png",
        "u(x) at y = 0.5",
        "x",
        "u",
    )
    print("Plots saved to results/")
    _write_report(metrics)
    print("=" * 100)
    return 0


if __name__ == "__main__":
    sys.exit(main())
