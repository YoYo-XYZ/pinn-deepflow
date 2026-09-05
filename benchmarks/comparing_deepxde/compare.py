#!/usr/bin/env python3
"""Compare a native DeepFlow model with the raw DeepXDE result archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# FLEX: these plots combine fields from two independent frameworks. The
# DeepFlow visualizer can render one evaluator, but cannot render a comparison.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.shared_harness import (  # noqa: E402
    BenchmarkConfig,
    collect_metrics,
    evaluate_area,
    load_model,
    plot_results,
    write_markdown_report,
)

try:  # Package execution.
    from .benchmark_common import RESULTS_DIR  # noqa: E402
    from .benchmark_deepflow import (  # noqa: E402
        DEFAULT_CONFIG,
        Lx,
        Ly,
        Re,
        SMOKE_CONFIG,
        build_domain,
    )
except ImportError:  # Direct script execution.
    from benchmark_common import RESULTS_DIR  # type: ignore  # noqa: E402
    from benchmark_deepflow import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG,
        Lx,
        Ly,
        Re,
        SMOKE_CONFIG,
        build_domain,
    )


METHODS = (("DeepFlow", "C0"), ("DeepXDE", "C1"))
REPORT_NAME = "REPORT.md"
DEFAULT_DEEPFLOW_MODEL_PATH = RESULTS_DIR / "deepflow.pkl"
DEFAULT_DEEPXDE_RESULTS_PATH = RESULTS_DIR / "deepxde_results.npz"

RESIDUAL_FIELDS = (
    "continuity_residual",
    "x_momentum_residual",
    "y_momentum_residual",
)
METRIC_ROWS = (
    ("Train time", "train_time_s", "{:.2f}", "s"),
    ("Final total loss", "final_total_loss", "{:.6e}", "-"),
    ("Best train loss", "best_loss_train", "{:.6e}", "-"),
    ("Best test loss", "best_loss_test", "{:.6e}", "-"),
    ("Mean |u|", "mean_u", "{:.6f}", "-"),
    ("Mean |v|", "mean_v", "{:.6f}", "-"),
    ("Max continuity|", "max_continuity_residual", "{:.6e}", "-"),
    ("Max x-momentum|", "max_x_momentum_residual", "{:.6e}", "-"),
    ("Max y-momentum|", "max_y_momentum_residual", "{:.6e}", "-"),
    ("Mass flux inlet", "mass_flux_inlet", "{:.6f}", "-"),
    ("Mass flux outlet", "mass_flux_outlet", "{:.6f}", "-"),
    ("Mass flux rel. error", "mass_flux_rel_error", "{:.6e}", "-"),
)

try:
    _trapz = np.trapezoid
except AttributeError:  # NumPy < 2.0
    _trapz = np.trapz


def _resolve_model_path(path: Path) -> Path:
    path = Path(path)
    return path if path.suffix == ".pkl" else Path(f"{path}.pkl")


def _result_data(result):
    if result is None:
        return None
    if isinstance(result, dict) and "data" in result:
        return result["data"]
    return result


def _load_results(
    deepflow_model_path: Path = DEFAULT_DEEPFLOW_MODEL_PATH,
    deepxde_results_path: Path = DEFAULT_DEEPXDE_RESULTS_PATH,
    config: BenchmarkConfig = DEFAULT_CONFIG,
):
    """Load the native DeepFlow model and raw DeepXDE archive."""
    results = {label: None for label, _ in METHODS}

    model_path = _resolve_model_path(deepflow_model_path)
    if model_path.is_file():
        model = load_model(model_path)
        domain = build_domain(config)
        evaluator = evaluate_area(domain, model, list(config.eval_grid))
        results["DeepFlow"] = {
            "data": evaluator.data_dict,
            "model": model,
            "domain": domain,
            "evaluator": evaluator,
            "metrics": collect_metrics(evaluator, model),
            "source_path": model_path,
        }
    else:
        print(f"[WARN] {model_path} not found.")

    deepxde_results_path = Path(deepxde_results_path)
    if deepxde_results_path.is_file():
        # FLEX: the untouched DeepXDE competitor writes this raw NPZ output,
        # which the comparison reader must consume as-is.
        with np.load(deepxde_results_path) as archive:
            results["DeepXDE"] = {
                "data": {key: archive[key] for key in archive.files},
                "source_path": deepxde_results_path,
            }
    else:
        print(f"[WARN] {deepxde_results_path} not found.")
    return results


def _array(data, key):
    if data is None or key not in data:
        return np.array([])
    return np.asarray(data[key])


def _scalar(data, key):
    values = _array(data, key)
    return float(values.flat[0]) if values.size else float("nan")


def _max_abs(values):
    values = np.asarray(values)
    return float(np.max(np.abs(values))) if values.size else float("nan")


def _mass_flux(data, x_target):
    x = _array(data, "x").reshape(-1)
    y = _array(data, "y").reshape(-1)
    u = _array(data, "u").reshape(-1)
    if not x.size or x.size != y.size or x.size != u.size:
        return float("nan")
    mask = np.abs(x - x_target) < 0.01
    if not np.any(mask):
        return float("nan")
    order = np.argsort(y[mask])
    return float(_trapz(u[mask][order], y[mask][order]))


def _metrics(results):
    """Derive comparison metrics from evaluator data and native histories."""
    metrics = {}
    for label, _ in METHODS:
        result = results[label]
        if result is None:
            metrics[label] = {}
            continue

        data = _result_data(result)
        entry = dict(result.get("metrics", {})) if isinstance(result, dict) else {}
        if label == "DeepXDE":
            entry.update(
                {
                    key: _scalar(data, key)
                    for key in (
                        "train_time_s",
                        "final_total_loss",
                        "best_loss_train",
                        "best_loss_test",
                    )
                }
            )
            loss_train = _array(data, "loss_train").reshape(-1)
            if loss_train.size:
                entry["history_last_total_loss"] = float(loss_train[-1])
        else:
            entry.setdefault("train_time_s", float("nan"))
            for key in (
                "final_total_loss",
                "best_loss_train",
                "best_loss_test",
            ):
                if key in data:
                    entry.setdefault(key, _scalar(data, key))
            if "history_last_total_loss" not in entry:
                history = _array(data, "total_loss").reshape(-1)
                if history.size:
                    entry["history_last_total_loss"] = float(history[-1])
            if "history_last_total_loss" in entry:
                entry.setdefault(
                    "final_total_loss", entry["history_last_total_loss"]
                )

        for field in ("u", "v"):
            values = _array(data, field)
            entry[f"mean_{field}"] = (
                float(np.mean(np.abs(values))) if values.size else float("nan")
            )
        entry.update(
            {
                f"max_{key}": _max_abs(_array(data, key))
                for key in RESIDUAL_FIELDS
            }
        )

        inlet = _mass_flux(data, 0.0)
        outlet = _mass_flux(data, Lx)
        entry.update(
            mass_flux_inlet=inlet,
            mass_flux_outlet=outlet,
            mass_flux_rel_error=(
                abs(inlet - outlet) / abs(inlet)
                if np.isfinite(inlet)
                and np.isfinite(outlet)
                and abs(inlet) > 1e-15
                else float("nan")
            ),
        )
        metrics[label] = entry
    return metrics


def _format_value(value, format_spec):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "N/A"
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
    values = [array[np.isfinite(array)] for array in values if array.size]
    if not values:
        return None
    finite = np.concatenate(values)
    return (float(finite.min()), float(finite.max())) if finite.size else None


def _scatter(ax, x, y, values, title, cmap, value_range):
    scatter = ax.scatter(x, y, c=values, s=1, cmap=cmap, marker="s")
    if value_range is not None:
        scatter.set_clim(value_range)
    ax.set(title=title, xlabel="x", ylabel="y")
    ax.set_aspect("equal")
    plt.colorbar(scatter, ax=ax, shrink=0.8)


def _plot_field(
    results,
    key,
    title,
    filename,
    cmap="viridis",
    absolute=False,
    output_dir: Path = RESULTS_DIR,
):
    fields = {
        label: _display_field(_result_data(results[label]), key, absolute)
        if results[label] is not None
        else None
        for label, _ in METHODS
    }
    value_range = _shared_range(*fields.values())
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for ax, (label, _) in zip(axes, METHODS):
        result = results[label]
        if result is None:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        data = _result_data(result)
        x, y, values = _field_data(data, "u")
        if not x.size or values.size != x.size or values.size != y.size:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            continue
        _scatter(ax, x, y, fields[label], f"{title} - {label}", cmap, value_range)
    fig.tight_layout()
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _profile(data, coordinate, target):
    x = _array(data, "x").reshape(-1)
    y = _array(data, "y").reshape(-1)
    u = _array(data, "u").reshape(-1)
    if not x.size or x.size != y.size or x.size != u.size:
        return np.array([]), np.array([])
    coordinate_values = x if coordinate == "x" else y
    profile_values = y if coordinate == "x" else x
    nearest_index = int(np.argmin(np.abs(coordinate_values - target)))
    nearest_coordinate = coordinate_values[nearest_index]
    indices = np.where(
        np.isclose(coordinate_values, nearest_coordinate, rtol=0.0, atol=1e-12)
    )[0]
    if not len(indices):
        return np.array([]), np.array([])
    order = np.argsort(profile_values[indices])
    return profile_values[indices][order], u[indices][order]


def _plot_profile(
    results,
    coordinate,
    target,
    filename,
    title,
    xlabel,
    ylabel,
    output_dir: Path = RESULTS_DIR,
):
    fig, ax = plt.subplots(figsize=(7, 5))
    for label, color in METHODS:
        result = results[label]
        if result is None:
            continue
        coordinate_values, u_values = _profile(_result_data(result), coordinate, target)
        if len(coordinate_values):
            if coordinate == "x":
                ax.plot(u_values, coordinate_values, label=label, color=color)
            else:
                ax.plot(coordinate_values, u_values, label=label, color=color)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(output_dir) / filename
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_loss(results, output_dir: Path = RESULTS_DIR):
    fig, ax = plt.subplots(figsize=(8, 5))
    deepflow_loss = _array(_result_data(results["DeepFlow"]), "total_loss")
    if len(deepflow_loss):
        ax.semilogy(deepflow_loss, label="DeepFlow total", color="C0")
    deepxde_loss = _array(_result_data(results["DeepXDE"]), "loss_train")
    if len(deepxde_loss):
        steps = _array(_result_data(results["DeepXDE"]), "loss_steps")
        x = (
            steps
            if len(steps) == len(deepxde_loss)
            else np.linspace(0, DEFAULT_CONFIG.epochs_adam, len(deepxde_loss))
        )
        ax.semilogy(x, deepxde_loss, label="DeepXDE train", color="C1", linestyle="--")
    ax.set(title="Training Loss Curves", xlabel="Iteration", ylabel="Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = Path(output_dir) / "compare_loss_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _comparison_plots(results, output_dir: Path):
    """Write cross-framework plots, which the single-model visualizer cannot make."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _plot_field(
            results,
            "u",
            "u",
            "compare_u_field.png",
            cmap="jet",
            output_dir=output_dir,
        ),
        _plot_field(
            results,
            "v",
            "v",
            "compare_v_field.png",
            cmap="jet",
            output_dir=output_dir,
        ),
        _plot_field(
            results,
            "p",
            "p",
            "compare_p_field.png",
            cmap="jet",
            output_dir=output_dir,
        ),
        _plot_field(
            results,
            "velocity_magnitude",
            "|U|",
            "compare_velocity_magnitude.png",
            cmap="jet",
            output_dir=output_dir,
        ),
        _plot_field(
            results,
            "continuity_residual",
            "|Continuity|",
            "compare_continuity_residual.png",
            cmap="hot",
            absolute=True,
            output_dir=output_dir,
        ),
        _plot_field(
            results,
            "x_momentum_residual",
            "|x-Momentum|",
            "compare_x_momentum_residual.png",
            cmap="hot",
            absolute=True,
            output_dir=output_dir,
        ),
        _plot_loss(results, output_dir=output_dir),
        _plot_profile(
            results,
            "x",
            2.5,
            "compare_profile_u_y_at_x2.5.png",
            "u(y) at x = 2.5",
            "u",
            "y",
            output_dir=output_dir,
        ),
        _plot_profile(
            results,
            "y",
            0.5,
            "compare_profile_u_x_at_y0.5.png",
            "u(x) at y = 0.5",
            "x",
            "u",
            output_dir=output_dir,
        ),
    ]


def _write_report(
    metrics,
    config: BenchmarkConfig = DEFAULT_CONFIG,
    artifacts=(),
    output_dir: Path = RESULTS_DIR / "comparison",
):
    report_metrics = {
        "methods": ", ".join(label for label, _ in METHODS),
        "geometry": f"rectangle [0, {Lx}] x [0, {Ly}]",
        "reynolds": Re,
        "width": config.width,
        "depth": config.depth,
        "epochs": config.epochs_adam,
        "interior_points": config.interior_points,
        "boundary_points": config.boundary_points,
    }
    for label, values in metrics.items():
        slug = label.lower()
        report_metrics.update({f"{slug}_{key}": value for key, value in values.items()})
    return write_markdown_report(
        Path(output_dir) / REPORT_NAME,
        "DeepFlow vs DeepXDE channel-flow comparison",
        config,
        report_metrics,
        artifacts,
    )


def run_comparison(
    deepflow_model_path: Path = DEFAULT_DEEPFLOW_MODEL_PATH,
    deepxde_results_path: Path = DEFAULT_DEEPXDE_RESULTS_PATH,
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR / "comparison",
):
    """Read both outputs, evaluate the native model, and write a report."""
    output_dir = Path(output_dir)
    results = _load_results(
        deepflow_model_path,
        deepxde_results_path,
        config,
    )
    if all(results[label] is None for label, _ in METHODS):
        raise FileNotFoundError("Neither DeepFlow nor DeepXDE results were found.")

    metrics = _metrics(results)
    _print_summary(metrics)
    artifacts = []
    native = results["DeepFlow"]
    if native is not None:
        artifacts.extend(plot_results(native["evaluator"], output_dir, prefix="deepflow"))
    artifacts.extend(_comparison_plots(results, output_dir))
    report = _write_report(metrics, config, artifacts, output_dir)
    return {
        "variants": results,
        "metrics": metrics,
        "report": report,
        "artifacts": artifacts,
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deepflow-model",
        type=Path,
        default=DEFAULT_DEEPFLOW_MODEL_PATH,
        help="Native DeepFlow model path, with or without .pkl.",
    )
    parser.add_argument(
        "--deepxde-results",
        type=Path,
        default=DEFAULT_DEEPXDE_RESULTS_PATH,
        help="DeepXDE result archive path.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the small shared-harness evaluation grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "comparison",
        help="Directory for comparison plots and report.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG
    result = run_comparison(
        args.deepflow_model,
        args.deepxde_results,
        config,
        args.output_dir,
    )
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
