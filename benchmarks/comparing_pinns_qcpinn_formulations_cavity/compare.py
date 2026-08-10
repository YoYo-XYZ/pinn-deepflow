#!/usr/bin/env python3
"""Compare all four cavity benchmark setups and write a Markdown report."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from common_config import (
    BOUNDARY_POINTS,
    CAVITY_X,
    CAVITY_Y,
    FEM_REFERENCE_FILENAME,
    EPOCHS_ADAM,
    EPOCHS_LBFGS,
    INTERIOR_POINTS,
    LID_VELOCITY,
    MU,
    PINN_LENGTH,
    PINN_WIDTH,
    QC_ITERATIONS,
    QC_NQUBITS,
    QC_POST,
    QC_PRE,
    REYNOLDS,
    REPORT_FILE,
    RESAMPLE_EVERY,
    RESULTS_DIR,
    SEEDS,
    THRESHOLD_ADAM,
    THRESHOLD_LBFGS,
)


SETUPS = (
    ("PINN-UVP", "PINN", "UVP", "pinn_uvp_results.npz", "C0"),
    ("QCPINN-UVP", "QCPINN", "UVP", "qcpinn_uvp_results.npz", "C1"),
    ("PINN-PSIP", "PINN", "PSIP", "pinn_psip_results.npz", "C2"),
    ("QCPINN-PSIP", "QCPINN", "PSIP", "qcpinn_psip_results.npz", "C3"),
)


LEGACY_PLOT_FILES = (
    "compare_u_field.png",
    "compare_v_field.png",
    "compare_velocity_magnitude.png",
    "compare_p_field.png",
    "compare_psi_field.png",
    "compare_continuity_residual.png",
    "compare_x_momentum_residual.png",
    "compare_y_momentum_residual.png",
)


LEGACY_FEM_PLOT_FILES = (
    "cfd_reference_fields.png",
    "compare_pinn_uvp_cfd_errors.png",
    "compare_qcpinn_uvp_cfd_errors.png",
    "compare_pinn_psip_cfd_errors.png",
    "compare_qcpinn_psip_cfd_errors.png",
)


def _load(path: Path):
    if not path.is_file():
        print(f"[WARN] {path} not found.")
        return None
    return np.load(path)


def _arr(data, key):
    if data is None or key not in data.files:
        return np.array([])
    return data[key]


def _scalar(data, key, default=float("nan")):
    if data is None or key not in data.files or data[key].size == 0:
        return default
    return float(data[key].flat[0])


def _text(data, key, default="N/A"):
    if data is None or key not in data.files:
        return default
    return str(data[key].item())


def _safe_int(value, default="N/A"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return default


def _format_mean_std(mean, std):
    if np.isnan(mean):
        return "N/A"
    if std > 0:
        return f"{mean:.6e} +/- {std:.6e}"
    return f"{mean:.6e}"


def _pct_delta(reference, candidate):
    if reference == 0 or np.isnan(reference) or np.isnan(candidate):
        return float("nan")
    return (candidate - reference) / abs(reference) * 100.0


def _validate_results(results):
    required = (
        "num_runs",
        "epochs_adam",
        "epochs_lbfgs",
        "seeds",
        "n_params",
        "final_total_loss_mean",
        "final_pde_loss_mean",
    )
    metadata = None
    for label, data, _, _, _ in results:
        if data is None:
            raise ValueError(f"{label} results are missing.")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{label} results are missing metadata: {missing}")
        current = (
            _scalar(data, "num_runs"),
            _scalar(data, "epochs_adam"),
            _scalar(data, "epochs_lbfgs"),
            np.asarray(data["seeds"]).reshape(-1),
        )
        if metadata is None:
            metadata = current
        else:
            if current[0:3] != metadata[0:3]:
                raise ValueError("The four setups do not use matching run metadata.")
            if not np.array_equal(current[3], metadata[3]):
                raise ValueError("The four setups do not use matching seeds.")


def _shared_range(*arrays):
    values = []
    for array in arrays:
        flat = np.asarray(array).ravel()
        flat = flat[np.isfinite(flat)]
        if flat.size:
            values.append(flat)
    if not values:
        return (0.0, 1.0)
    values = np.concatenate(values)
    return float(np.min(values)), float(np.max(values))


def _symmetric_range(*arrays):
    lower, upper = _shared_range(*arrays)
    radius = max(abs(lower), abs(upper), 1.0e-14)
    return -radius, radius


def _field_values(data, field):
    if field == "velocity_magnitude":
        u = _arr(data, "u")
        v = _arr(data, "v")
        if not u.size or not v.size:
            return np.array([])
        return np.sqrt(u**2 + v**2)
    return _arr(data, field)


def _grid_field(data, field, values=None):
    """Return sorted grid coordinates and a 2D field for contour plots."""
    x = np.asarray(_arr(data, "x")).reshape(-1)
    y = np.asarray(_arr(data, "y")).reshape(-1)
    if values is None:
        values = _arr(data, field)
    values = np.asarray(values).reshape(-1)
    x_values = np.unique(x)
    y_values = np.unique(y)
    if x_values.size * y_values.size != values.size:
        raise ValueError(f"Expected a rectangular evaluation grid for '{field}'.")

    x_indices = np.searchsorted(x_values, x)
    y_indices = np.searchsorted(y_values, y)
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=values.dtype)
    grid[y_indices, x_indices] = values
    return x_values, y_values, grid


def _contour_plot(
    ax,
    x,
    y,
    values,
    title,
    cmap="viridis",
    vrange=None,
    colorbar=True,
    show_x_label=True,
    show_y_label=True,
):
    if vrange is None:
        plot = ax.contourf(x, y, values, levels=50, cmap=cmap)
    else:
        lower, upper = vrange
        if np.isclose(lower, upper):
            lower -= 0.5
            upper += 0.5
        plot = ax.contourf(
            x,
            y,
            values,
            levels=np.linspace(lower, upper, 51),
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            extend="both",
        )
    ax.set_title(title)
    ax.set_xlabel("x" if show_x_label else "")
    ax.set_ylabel("y" if show_y_label else "")
    ax.set_aspect("equal")
    ax.set_xlim(CAVITY_X)
    ax.set_ylim(CAVITY_Y)
    if colorbar:
        plt.colorbar(plot, ax=ax, shrink=0.8)
    return plot


def _plot_solution_fields(data_by_setup):
    fields = ("u", "v", "velocity_magnitude", "p")
    titles = ("u velocity", "v velocity", "Velocity magnitude", "Pressure")
    ranges = []
    for field in fields:
        values = [
            _field_values(data, field)
            for _, data, _, _, _ in data_by_setup
            if _field_values(data, field).size
        ]
        ranges.append(_shared_range(*values))

    fig, axes = plt.subplots(
        len(data_by_setup),
        len(fields),
        figsize=(20, 14),
        squeeze=False,
        constrained_layout=True,
    )
    mappables = [None] * len(fields)
    for row, (label, data, _, _, _) in enumerate(data_by_setup):
        for column, (field, title) in enumerate(zip(fields, titles)):
            ax = axes[row, column]
            values = _field_values(data, field)
            if not values.size:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{title} -- {label}")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            x, y, grid = _grid_field(data, field, values=values)
            mappables[column] = _contour_plot(
                ax,
                x,
                y,
                grid,
                f"{title} -- {label}",
                vrange=ranges[column],
                colorbar=False,
                show_x_label=row == len(data_by_setup) - 1,
                show_y_label=column == 0,
            )

    for column, mappable in enumerate(mappables):
        if mappable is None:
            continue
        fig.colorbar(mappable, ax=axes[:, column].tolist(), shrink=0.8)

    fig.suptitle("Solution fields across PINN/QCPINN and UVP/PSIP setups")
    fig.savefig(RESULTS_DIR / "compare_solution_fields.png", dpi=150)
    plt.close(fig)
    print("  -> compare_solution_fields.png")


def _plot_residual_fields(data_by_setup):
    fields = ("continuity_residual", "x_momentum_residual", "y_momentum_residual")
    titles = ("|continuity residual|", "x-momentum residual", "y-momentum residual")
    absolute = (True, False, False)
    ranges = []
    for field, use_absolute in zip(fields, absolute):
        values = [
            np.abs(_arr(data, field)) if use_absolute else _arr(data, field)
            for _, data, _, _, _ in data_by_setup
            if _arr(data, field).size
        ]
        ranges.append(
            _shared_range(*values)
            if use_absolute
            else _symmetric_range(*values)
        )

    fig, axes = plt.subplots(
        len(data_by_setup),
        len(fields),
        figsize=(15, 14),
        squeeze=False,
        constrained_layout=True,
    )
    mappables = [None] * len(fields)
    for row, (label, data, _, _, _) in enumerate(data_by_setup):
        for column, (field, title, use_absolute) in enumerate(zip(fields, titles, absolute)):
            values = _arr(data, field)
            ax = axes[row, column]
            if not values.size:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(f"{title} -- {label}")
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if use_absolute:
                values = np.abs(values)
            x, y, grid = _grid_field(data, field, values=values)
            mappables[column] = _contour_plot(
                ax,
                x,
                y,
                grid,
                f"{title} -- {label}",
                cmap="viridis" if use_absolute else "RdBu_r",
                vrange=ranges[column],
                colorbar=False,
                show_x_label=row == len(data_by_setup) - 1,
                show_y_label=column == 0,
            )

    for column, mappable in enumerate(mappables):
        if mappable is not None:
            fig.colorbar(mappable, ax=axes[:, column].tolist(), shrink=0.8)

    fig.suptitle("PDE residual fields across benchmark setups")
    fig.savefig(RESULTS_DIR / "compare_residual_fields.png", dpi=150)
    plt.close(fig)
    print("  -> compare_residual_fields.png")


def _plot_loss_curves(data_by_setup):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, key, title in zip(
        axes,
        ("total_loss_history", "bc_loss_history", "pde_loss_history"),
        ("Total loss", "BC loss", "PDE loss"),
    ):
        for label, data, _, _, color in data_by_setup:
            history = _arr(data, key)
            if history.size:
                ax.semilogy(
                    np.maximum(history, 1.0e-16), label=label, color=color
                )
        ax.set_title(title)
        ax.set_xlabel("L-BFGS epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / "compare_loss_curves.png", dpi=150)
    plt.close(fig)
    print("  -> compare_loss_curves.png")


def _reference_on_model_grid(cfd_data, model_data, field):
    model_values = _field_values(model_data, field)
    model_x, model_y, model_values = _grid_field(
        model_data, field, values=model_values
    )
    points = np.stack(
        np.meshgrid(model_y, model_x, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    interpolator = RegularGridInterpolator(
        (_arr(cfd_data, "y"), _arr(cfd_data, "x")),
        _field_values(cfd_data, field),
        bounds_error=False,
        fill_value=None,
    )
    reference_values = interpolator(points).reshape(model_values.shape)
    return model_x, model_y, model_values, reference_values


def _reference_metrics(cfd_data, model_data):
    metrics = {}
    model_x, model_y, _, _ = _reference_on_model_grid(cfd_data, model_data, "u")
    x_min, x_max = _arr(cfd_data, "x")[[0, -1]]
    y_min, y_max = _arr(cfd_data, "y")[[0, -1]]
    interior = (
        (model_x[None, :] >= x_min)
        & (model_x[None, :] <= x_max)
        & (model_y[:, None] >= y_min)
        & (model_y[:, None] <= y_max)
    )
    fields = {}
    for field in ("u", "v", "p"):
        _, _, model_values, reference_values = _reference_on_model_grid(
            cfd_data, model_data, field
        )
        fields[field] = (model_values, reference_values)
        difference = model_values[interior] - reference_values[interior]
        denominator = max(float(np.linalg.norm(reference_values[interior])), 1.0e-14)
        metrics[f"l2_relative_{field}"] = float(
            np.linalg.norm(difference) / denominator
        )

    model_speed = np.sqrt(fields["u"][0] ** 2 + fields["v"][0] ** 2)
    reference_speed = np.sqrt(fields["u"][1] ** 2 + fields["v"][1] ** 2)
    fields["velocity_magnitude"] = (model_speed, reference_speed)
    metrics["l2_relative_speed"] = float(
        np.linalg.norm((model_speed - reference_speed)[interior])
        / max(float(np.linalg.norm(reference_speed[interior])), 1.0e-14)
    )

    vertical_y = _arr(model_data, "vertical_y")
    vertical_u = _arr(model_data, "vertical_u")
    horizontal_x = _arr(model_data, "horizontal_x")
    horizontal_v = _arr(model_data, "horizontal_v")
    vertical_order = np.argsort(vertical_y)
    horizontal_order = np.argsort(horizontal_x)
    vertical_prediction = np.interp(
        _arr(cfd_data, "vertical_y"),
        vertical_y[vertical_order],
        vertical_u[vertical_order],
    )
    horizontal_prediction = np.interp(
        _arr(cfd_data, "horizontal_x"),
        horizontal_x[horizontal_order],
        horizontal_v[horizontal_order],
    )
    metrics["rmse_vertical_u"] = float(
        np.sqrt(np.mean((vertical_prediction - _arr(cfd_data, "vertical_u")) ** 2))
    )
    metrics["rmse_horizontal_v"] = float(
        np.sqrt(
            np.mean((horizontal_prediction - _arr(cfd_data, "horizontal_v")) ** 2)
        )
    )
    return metrics, fields, interior


def _plot_fem_reference_and_errors(data_by_setup, cfd_data):
    fields = ("velocity_magnitude", "v", "p")
    field_titles = ("Velocity magnitude", "v velocity", "Pressure")
    error_data = []
    for label, data, _, _, _ in data_by_setup:
        _, model_fields, interior = _reference_metrics(cfd_data, data)
        differences = {}
        for field in fields:
            model_values, reference_values = model_fields[field]
            differences[field] = np.where(
                interior, model_values - reference_values, np.nan
            )
        error_data.append((label, data, differences))

    error_ranges = {
        field: _symmetric_range(
            *(differences[field] for _, _, differences in error_data)
        )
        for field in fields
    }

    reference_ranges = []
    for field in fields:
        values = _field_values(cfd_data, field)
        if field == "p":
            reference_ranges.append(tuple(np.percentile(values, [2.0, 98.0])))
        else:
            reference_ranges.append(_shared_range(values))

    fig, axes = plt.subplots(
        len(data_by_setup) + 1,
        len(fields),
        figsize=(16, 20),
        squeeze=False,
        constrained_layout=True,
    )
    reference_mappables = []
    for column, (field, title) in enumerate(zip(fields, field_titles)):
        reference_mappables.append(
            _contour_plot(
                axes[0, column],
                _arr(cfd_data, "x"),
                _arr(cfd_data, "y"),
                _field_values(cfd_data, field),
                f"FEM reference -- {title}",
                vrange=reference_ranges[column],
                colorbar=False,
                show_x_label=False,
                show_y_label=column == 0,
            )
        )

    error_mappables = [None] * len(fields)
    for row, (label, data, differences) in enumerate(error_data, start=1):
        for column, (field, title) in enumerate(zip(fields, field_titles)):
            model_x, model_y, _, _ = _reference_on_model_grid(
                cfd_data, data, field
            )
            mappable = _contour_plot(
                axes[row, column],
                model_x,
                model_y,
                differences[field],
                f"{label} - FEM: {title}",
                cmap="RdBu_r",
                vrange=error_ranges[field],
                colorbar=False,
                show_x_label=row == len(data_by_setup),
                show_y_label=column == 0,
            )
            if row == 1:
                error_mappables[column] = mappable

    for column, mappable in enumerate(reference_mappables):
        fig.colorbar(mappable, ax=axes[0, column], shrink=0.8)
        fig.colorbar(
            error_mappables[column],
            ax=axes[1:, column].tolist(),
            shrink=0.8,
        )

    fig.suptitle("FEM reference fields and model-minus-FEM errors")
    fig.savefig(RESULTS_DIR / "compare_fem_reference_and_errors.png", dpi=150)
    plt.close(fig)
    print("  -> compare_fem_reference_and_errors.png")


def _plot_centerlines(data_by_setup, cfd_data):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), constrained_layout=True)
    for label, data, _, _, color in data_by_setup:
        vertical_order = np.argsort(_arr(data, "vertical_y"))
        horizontal_order = np.argsort(_arr(data, "horizontal_x"))
        axes[0].plot(
            _arr(data, "vertical_u")[vertical_order],
            _arr(data, "vertical_y")[vertical_order],
            label=label,
            color=color,
        )
        axes[1].plot(
            _arr(data, "horizontal_x")[horizontal_order],
            _arr(data, "horizontal_v")[horizontal_order],
            label=label,
            color=color,
        )
    if cfd_data is not None:
        axes[0].plot(
            _arr(cfd_data, "vertical_u"),
            _arr(cfd_data, "vertical_y"),
            label="FEM",
            color="black",
            linestyle="--",
        )
        axes[1].plot(
            _arr(cfd_data, "horizontal_x"),
            _arr(cfd_data, "horizontal_v"),
            label="FEM",
            color="black",
            linestyle="--",
        )
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("y")
    axes[0].set_title("Vertical centerline: u(y) at x=0.5")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v")
    axes[1].set_title("Horizontal centerline: v(x) at y=0.5")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.savefig(RESULTS_DIR / "compare_centerline_profiles.png", dpi=150)
    plt.close(fig)
    print("  -> compare_centerline_profiles.png")


def _write_report(data_by_setup, cfd_data, reference_metrics):
    first_data = data_by_setup[0][1]
    report_seeds = _arr(first_data, "seeds")
    interior_resolution = INTERIOR_POINTS[0]
    interior_count = int(np.prod(interior_resolution))
    training_text = (
        f"L-BFGS({EPOCHS_LBFGS} epochs) with no Adam warm-up"
        if EPOCHS_ADAM == 0
        else f"Adam({EPOCHS_ADAM} epochs) -> L-BFGS({EPOCHS_LBFGS} epochs)"
    )
    resampling_text = (
        "disabled (fixed collocation set)"
        if RESAMPLE_EVERY is None
        else f"uniform resample every {RESAMPLE_EVERY} L-BFGS epochs"
    )

    lines = [
        "# Benchmark Report: PINN/QCPINN × UVP/PSIP",
        "",
        "## 2D Lid-Driven Cavity Flow (Re=10)",
        "",
        f"- **Geometry**: unit square [{CAVITY_X[0]}, {CAVITY_X[1]}] x "
        f"[{CAVITY_Y[0]}, {CAVITY_Y[1]}]",
        f"- **PDE**: steady incompressible 2D Navier-Stokes; Re={REYNOLDS} "
        f"(mu={MU}, rho=1, U={LID_VELOCITY}, L=1)",
        "- **UVP boundary conditions**: no-slip on left/bottom/right, "
        "top lid `u=1, v=0`, and `p=0` at the lower-left corner.",
        "- **PSIP boundary conditions**: stationary walls `psi_x=0, psi_y=0`, "
        "top lid `psi_x=0, psi_y=1`, and `p=0` at the lower-left corner.",
        f"- **Sampling**: uniform -- {sum(BOUNDARY_POINTS)} boundary points, "
        f"{interior_count} interior points",
        f"- **Resampling**: {resampling_text}",
        f"- **Training**: {training_text}",
        "- **Loss**: raw `df.calc_loss_simple` (unweighted BC + PDE sum)",
        f"- **Seeds**: {report_seeds.tolist()}",
        "- **PSIP note**: velocity is derived as `u=psi_y`, `v=-psi_x`; "
        "continuity is therefore satisfied analytically.",
        "- **Capacity note**: the four cells preserve the existing benchmark "
        "architectures; QCPINN and PINN parameter counts are intentionally not equal.",
        "",
        "## Network Architectures",
        "",
        "| Setup | Architecture | Parameters | PDE residuals |",
        "|-------|--------------|-----------:|--------------:|",
    ]
    for label, data, _, _, _ in data_by_setup:
        lines.append(
            f"| **{label}** | `{_text(data, 'network_description')}` | "
            f"{_safe_int(_scalar(data, 'n_params'))} | "
            f"{_safe_int(_scalar(data, 'pde_residual_count'))} |"
        )

    median_indices = ", ".join(
        f"{label}={_safe_int(_scalar(data, 'median_run_idx'))}"
        for label, data, _, _, _ in data_by_setup
    )
    lines += [
        "",
        "## Training and Residual Summary",
        "",
        "Raw UVP and PSIP PDE totals are not directly comparable because UVP has "
        "three residual equations and PSIP has two. PDE loss per residual and "
        "the field/FEM errors are the preferred cross-formulation measures.",
        "",
        "| Setup | Total loss | PDE loss | PDE/residual | Max continuity | Max x-momentum | Max y-momentum | Total time (s) |",
        "|-------|-----------:|---------:|-------------:|---------------:|----------------:|----------------:|---------------:|",
    ]
    for label, data, _, _, _ in data_by_setup:
        lines.append(
            f"| {label} | {_format_mean_std(_scalar(data, 'final_total_loss_mean'), _scalar(data, 'final_total_loss_std'))} | "
            f"{_format_mean_std(_scalar(data, 'final_pde_loss_mean'), _scalar(data, 'final_pde_loss_std'))} | "
            f"{_format_mean_std(_scalar(data, 'pde_loss_per_equation_mean'), _scalar(data, 'pde_loss_per_equation_std'))} | "
            f"{_format_mean_std(_scalar(data, 'max_continuity_mean'), _scalar(data, 'max_continuity_std'))} | "
            f"{_format_mean_std(_scalar(data, 'max_x_momentum_mean'), _scalar(data, 'max_x_momentum_std'))} | "
            f"{_format_mean_std(_scalar(data, 'max_y_momentum_mean'), _scalar(data, 'max_y_momentum_std'))} | "
            f"{_format_mean_std(_scalar(data, 'total_time_s_mean'), _scalar(data, 'total_time_s_std'))} |"
        )

    if reference_metrics:
        lines += [
            "",
            "## Error Against Fresh FEM Reference",
            "",
            "| Setup | Relative L2 u | Relative L2 v | Relative L2 speed | Relative L2 p | u centerline RMSE | v centerline RMSE |",
            "|-------|---------------:|---------------:|------------------:|--------------:|------------------:|------------------:|",
        ]
        for label, _, _, _, _ in data_by_setup:
            metrics = reference_metrics[label]
            lines.append(
                f"| {label} | {metrics['l2_relative_u']:.6e} | "
                f"{metrics['l2_relative_v']:.6e} | "
                f"{metrics['l2_relative_speed']:.6e} | "
                f"{metrics['l2_relative_p']:.6e} | "
                f"{metrics['rmse_vertical_u']:.6e} | "
                f"{metrics['rmse_horizontal_v']:.6e} |"
            )

        lines += [
            "",
            "### Factorized Interpretation",
            "",
            "At fixed formulation, compare PINN-UVP with QCPINN-UVP and "
            "PINN-PSIP with QCPINN-PSIP. At fixed model family, compare UVP "
            "with PSIP. These comparisons use the same sampling, optimizer, "
            "seed, and FEM reference.",
        ]
        cfd_path = RESULTS_DIR / FEM_REFERENCE_FILENAME
        if cfd_path.is_file():
            with np.load(cfd_path) as cfd:
                lines += [
                    "",
                    "### FEM Reference Diagnostics",
                    "",
                    f"- **Grid**: {_safe_int(_scalar(cfd, 'nx'))} x {_safe_int(_scalar(cfd, 'ny'))} samples",
                    f"- **FEM mesh size**: {_scalar(cfd, 'mesh_size'):.6g}",
                    f"- **FEM elements**: {_safe_int(_scalar(cfd, 'mesh_elements'))}",
                    f"- **Iterations**: {_safe_int(_scalar(cfd, 'iterations'))}",
                    f"- **Final residual**: {_scalar(cfd, 'final_residual'):.6e}",
                    f"- **Converged**: {bool(_scalar(cfd, 'converged'))}",
                    f"- **Pressure gauge**: {_text(cfd, 'pressure_gauge')}",
                ]

    lines += [
        "",
        "## Visual Comparisons",
        "",
        "### Solution fields",
        "",
        "![Solution fields](compare_solution_fields.png)",
        "",
        "### PDE residual fields",
        "",
        "![PDE residual fields](compare_residual_fields.png)",
        "",
        "### Training losses",
        "",
        "![Training loss curves](compare_loss_curves.png)",
        "",
        "### Centerline profiles",
        "",
        "![Centerline velocity profiles](compare_centerline_profiles.png)",
    ]
    if cfd_data is not None:
        lines += [
            "",
            "### FEM reference and errors",
            "",
            "![FEM reference and model errors](compare_fem_reference_and_errors.png)",
        ]

    lines += [
        "",
        "## Generated Artifacts",
        "",
        "- `pinn_uvp_results.npz`, `qcpinn_uvp_results.npz`, `pinn_psip_results.npz`, `qcpinn_psip_results.npz` -- setup results",
        "- `compare_solution_fields.png` -- u, v, velocity magnitude, and pressure fields",
        "- `compare_residual_fields.png` -- continuity and momentum residual fields",
        "- `compare_loss_curves.png` -- loss histories for all four setups",
        "- `compare_centerline_profiles.png` -- centerline velocity profiles",
    ]
    if cfd_data is not None:
        lines.append(
            "- `compare_fem_reference_and_errors.png` -- FEM reference fields and setup errors"
        )
    lines += [
        "",
        "## Reproducibility",
        "",
        f"- Median run indices: {median_indices}",
        "- The benchmark uses the local DeepFlow source tree and PennyLane for QCPINN.",
        "",
        "---",
        "*Report generated by `compare.py`*",
    ]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  -> {REPORT_FILE}")


def _remove_legacy_plots(include_fem):
    filenames = list(LEGACY_PLOT_FILES)
    if include_fem:
        filenames.extend(LEGACY_FEM_PLOT_FILES)
    for filename in filenames:
        path = RESULTS_DIR / filename
        if path.is_file():
            path.unlink()
            print(f"  removed obsolete plot: {filename}")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_by_setup = []
    for label, model, formulation, filename, color in SETUPS:
        data_by_setup.append(
            (label, _load(RESULTS_DIR / filename), formulation, model, color)
        )
    _validate_results(data_by_setup)
    cfd_data = _load(RESULTS_DIR / FEM_REFERENCE_FILENAME)

    print("=" * 105)
    print("PINN/QCPINN x UVP/PSIP -- Re=10 Cavity Benchmark")
    print("=" * 105)
    print(
        f"{'Setup':<16} {'Params':>8} {'Total loss':>15} "
        f"{'PDE loss':>15} {'PDE/equation':>15}"
    )
    print("-" * 105)
    for label, data, _, _, _ in data_by_setup:
        print(
            f"{label:<16} {_safe_int(_scalar(data, 'n_params')):>8} "
            f"{_scalar(data, 'final_total_loss_mean'):>15.6e} "
            f"{_scalar(data, 'final_pde_loss_mean'):>15.6e} "
            f"{_scalar(data, 'pde_loss_per_equation_mean'):>15.6e}"
        )

    reference_metrics = {}
    if cfd_data is not None:
        print(
            f"FEM reference: {_safe_int(_scalar(cfd_data, 'nx'))} x "
            f"{_safe_int(_scalar(cfd_data, 'ny'))}, "
            f"converged={bool(_scalar(cfd_data, 'converged'))}"
        )
        for label, data, _, _, _ in data_by_setup:
            reference_metrics[label] = _reference_metrics(cfd_data, data)[0]
            metrics = reference_metrics[label]
            print(
                f"  {label}: L2 u={metrics['l2_relative_u']:.6e}, "
                f"v={metrics['l2_relative_v']:.6e}, "
                f"speed={metrics['l2_relative_speed']:.6e}, "
                f"p={metrics['l2_relative_p']:.6e}"
            )
    else:
        print("[WARN] FEM reference is unavailable; FEM metrics will be skipped.")

    print("\nGenerating comparison plots ...")
    _plot_solution_fields(data_by_setup)
    _plot_residual_fields(data_by_setup)
    _plot_loss_curves(data_by_setup)
    _plot_centerlines(data_by_setup, cfd_data)
    if cfd_data is not None:
        _plot_fem_reference_and_errors(data_by_setup, cfd_data)

    _remove_legacy_plots(include_fem=cfd_data is not None)

    print("\nWriting Markdown report ...")
    _write_report(data_by_setup, cfd_data, reference_metrics)
    print("Done.")


if __name__ == "__main__":
    main()
