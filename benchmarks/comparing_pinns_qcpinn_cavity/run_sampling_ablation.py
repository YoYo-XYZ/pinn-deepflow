#!/usr/bin/env python3
"""Run independent PINN sampling ablations for the cavity benchmark."""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from benchmark_common import aggregate, train_one  # noqa: E402
from benchmark_pinn import build_model  # noqa: E402
from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    CFD_REFERENCE_FILENAME,
    EPOCHS_ADAM,
    EPOCHS_LBFGS,
    INTERIOR_POINTS,
    PINN_LENGTH,
    PINN_WIDTH,
    RESULTS_DIR,
    SEEDS,
)

VARIANTS = ("baseline", "corner_excluded", "boundary_refined")
VARIANT_DESCRIPTIONS = {
    "baseline": "Original uniform boundaries including corner endpoints",
    "corner_excluded": "Uniform boundaries with lid/wall corner endpoints removed",
    "boundary_refined": "Uniform boundaries with twice as many points per edge",
}


def _model_grid(data, field):
    """Convert flattened DeepFlow evaluation data to a sorted 2D grid."""
    x = np.unique(np.asarray(data["x"]).reshape(-1))
    y = np.unique(np.asarray(data["y"]).reshape(-1))
    values = np.asarray(data[field]).reshape(-1)
    x_indices = np.searchsorted(x, np.asarray(data["x"]).reshape(-1))
    y_indices = np.searchsorted(y, np.asarray(data["y"]).reshape(-1))
    grid = np.full((y.size, x.size), np.nan, dtype=values.dtype)
    grid[y_indices, x_indices] = values
    return x, y, grid


def _reference_metrics(cfd, model):
    """Compute common-grid field and centerline errors against CFD."""
    model_x, model_y, _ = _model_grid(model, "u")
    points = np.stack(
        np.meshgrid(model_y, model_x, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    cfd_x = np.asarray(cfd["x"])
    cfd_y = np.asarray(cfd["y"])
    interior = (
        (model_x[None, :] >= cfd_x[0])
        & (model_x[None, :] <= cfd_x[-1])
        & (model_y[:, None] >= cfd_y[0])
        & (model_y[:, None] <= cfd_y[-1])
    )

    metrics = {}
    model_fields = {}
    reference_fields = {}
    for field in ("u", "v", "p"):
        _, _, model_values = _model_grid(model, field)
        interpolator = RegularGridInterpolator(
            (cfd_y, cfd_x),
            np.asarray(cfd[field]),
            bounds_error=False,
            fill_value=None,
        )
        reference_values = interpolator(points).reshape(model_values.shape)
        difference = model_values[interior] - reference_values[interior]
        denominator = max(float(np.linalg.norm(reference_values[interior])), 1.0e-14)
        metrics[f"l2_relative_{field}"] = float(
            np.linalg.norm(difference) / denominator
        )
        model_fields[field] = model_values
        reference_fields[field] = reference_values

    model_speed = np.sqrt(model_fields["u"] ** 2 + model_fields["v"] ** 2)
    reference_speed = np.sqrt(
        reference_fields["u"] ** 2 + reference_fields["v"] ** 2
    )
    metrics["l2_relative_speed"] = float(
        np.linalg.norm((model_speed - reference_speed)[interior])
        / max(float(np.linalg.norm(reference_speed[interior])), 1.0e-14)
    )

    for direction, coordinate_key, value_key, cfd_coordinate, cfd_value in (
        ("vertical", "vertical_y", "vertical_u", "vertical_y", "vertical_u"),
        ("horizontal", "horizontal_x", "horizontal_v", "horizontal_x", "horizontal_v"),
    ):
        model_coordinate = np.asarray(model[coordinate_key])
        model_value = np.asarray(model[value_key])
        reference_coordinate = np.asarray(cfd[cfd_coordinate])
        reference_value = np.interp(
            model_coordinate, reference_coordinate, np.asarray(cfd[cfd_value])
        )
        overlap = (
            (model_coordinate >= reference_coordinate[0])
            & (model_coordinate <= reference_coordinate[-1])
        )
        metrics[f"centerline_rmse_{direction}"] = float(
            np.sqrt(np.mean((model_value[overlap] - reference_value[overlap]) ** 2))
        )
    return metrics


def _format(value):
    return f"{value:.6e}"


def main():
    parser = argparse.ArgumentParser(
        description="PINN-only cavity sampling ablation against the CFD reference."
    )
    parser.add_argument("--num_runs", type=int, default=1)
    parser.add_argument("--epochs_adam", type=int, default=EPOCHS_ADAM)
    parser.add_argument("--epochs_lbfgs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=SEEDS[0])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    args = parser.parse_args()

    cfd_path = RESULTS_DIR / CFD_REFERENCE_FILENAME
    if not cfd_path.is_file():
        raise SystemExit(f"Missing CFD reference: {cfd_path}")
    cfd = np.load(cfd_path)
    summary = []

    for variant in args.variants:
        print("\n" + "=" * 80)
        print(f"Sampling ablation: {variant}")
        print(VARIANT_DESCRIPTIONS[variant])
        print("=" * 80)
        per_run = []
        for run_index in range(args.num_runs):
            seed = args.seed + run_index
            per_run.append(
                train_one(
                    seed=seed,
                    model_factory=build_model,
                    label=f"PINN-{variant}",
                    network_description=(
                        f"PINN(width={PINN_WIDTH}, length={PINN_LENGTH}), {variant}"
                    ),
                    epochs_adam=args.epochs_adam,
                    epochs_lbfgs=args.epochs_lbfgs,
                    sampling_mode=variant,
                    boundary_points=BOUNDARY_POINTS,
                    interior_points=INTERIOR_POINTS,
                )
            )

        results = aggregate(
            per_run,
            label=f"PINN-{variant}",
            num_runs=args.num_runs,
            epochs_adam=args.epochs_adam,
            epochs_lbfgs=args.epochs_lbfgs,
        )
        results["sampling_variant"] = np.asarray(variant)
        results["sampling_description"] = np.asarray(VARIANT_DESCRIPTIONS[variant])
        metrics = _reference_metrics(cfd, results)
        results.update(metrics)
        output_path = RESULTS_DIR / f"pinn_sampling_{variant}.npz"
        np.savez(output_path, **results)
        print(f"Saved: {output_path}")
        print(
            f"  total loss={_format(results['final_total_loss_mean'])}, "
            f"L2(u)={_format(metrics['l2_relative_u'])}, "
            f"L2(v)={_format(metrics['l2_relative_v'])}, "
            f"L2(speed)={_format(metrics['l2_relative_speed'])}, "
            f"L2(p)={_format(metrics['l2_relative_p'])}"
        )
        summary.append((variant, results, metrics))

    report_path = RESULTS_DIR / "sampling_ablation_report.md"
    lines = [
        "# PINN Cavity Sampling Ablation",
        "",
        "Independent PINN-only tests with the same seed, architecture, CFD reference, "
        f"and training budget ({args.epochs_adam} Adam + {args.epochs_lbfgs} L-BFGS epochs).",
        "",
        f"- Architecture: `PINN(width={PINN_WIDTH}, length={PINN_LENGTH})`",
        f"- Runs per variant: {args.num_runs}",
        f"- Base seed: {args.seed}",
        "",
        "| Variant | Final loss | Relative L2 u | Relative L2 v | Relative L2 speed | Relative L2 p | Vertical RMSE | Horizontal RMSE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for variant, results, metrics in summary:
        lines.append(
            f"| `{variant}` | {_format(results['final_total_loss_mean'])} | "
            f"{_format(metrics['l2_relative_u'])} | {_format(metrics['l2_relative_v'])} | "
            f"{_format(metrics['l2_relative_speed'])} | {_format(metrics['l2_relative_p'])} | "
            f"{_format(metrics['centerline_rmse_vertical'])} | "
            f"{_format(metrics['centerline_rmse_horizontal'])} |"
        )
    lines += [
        "",
        "The `corner_excluded` test removes only the endpoint samples from the four "
        "rectangle edges. The `boundary_refined` test retains the endpoint behavior "
        "but doubles the uniform samples on each physical edge.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved report: {report_path}")


if __name__ == "__main__":
    main()
