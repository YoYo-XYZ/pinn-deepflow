#!/usr/bin/env python3
"""Compare PINN and QCPINN results for the Re=10 cavity benchmark."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    CAVITY_X,
    CAVITY_Y,
    CFD_GRID_CONVERGENCE_FILENAME,
    CFD_REFERENCE_FILENAME,
    EPOCHS_ADAM,
    EPOCHS_LBFGS,
    INTERIOR_POINTS,
    LID_VELOCITY,
    LR_ADAM,
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
    QCPINN_RESULTS_FILE,
    PINN_RESULTS_FILE,
)


def _load(path):
    if not path.is_file():
        print(f"[WARN] {path} not found.")
        return None
    return np.load(path)


def _scalar(data, key, default=float("nan")):
    if data is None or key not in data.files or data[key].size == 0:
        return default
    return float(data[key].flat[0])


def _arr(data, key):
    if data is None or key not in data.files:
        return np.array([])
    return data[key]


def _shared_range(*arrays):
    values = [np.asarray(array).ravel() for array in arrays if np.asarray(array).size]
    if not values:
        return None
    values = np.concatenate(values)
    return float(np.nanmin(values)), float(np.nanmax(values))


def _format_mean_std(mean, std):
    if std > 0:
        return f"{mean:.6e} +/- {std:.6e}"
    return f"{mean:.6e}"


def _safe_int(value, default="N/A"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return str(int(value))
    except (ValueError, TypeError):
        return default


def _pct_delta(a, b):
    if a == 0:
        return float("nan")
    return (b - a) / a * 100.0


def _required_int(data, key, label):
    if data is None or key not in data.files or data[key].size != 1:
        raise ValueError(
            f"{label} results are missing scalar metadata '{key}'; rerun the benchmark."
        )
    return int(data[key].flat[0])


def _required_seeds(data, label):
    if data is None or "seeds" not in data.files:
        raise ValueError(f"{label} results are missing seed metadata; rerun the benchmark.")
    return np.asarray(data["seeds"]).reshape(-1)


def _validate_matching_runs(pinn, qcpinn):
    pinn_runs = _required_int(pinn, "num_runs", "PINN")
    qcpinn_runs = _required_int(qcpinn, "num_runs", "QCPINN")
    if pinn_runs != qcpinn_runs:
        raise ValueError(f"num_runs differs (PINN={pinn_runs}, QCPINN={qcpinn_runs}).")

    pinn_seeds = _required_seeds(pinn, "PINN")
    qcpinn_seeds = _required_seeds(qcpinn, "QCPINN")
    if len(pinn_seeds) != pinn_runs or len(qcpinn_seeds) != qcpinn_runs:
        raise ValueError("seed metadata length does not match num_runs.")
    if not np.array_equal(pinn_seeds, qcpinn_seeds):
        raise ValueError(
            f"seed lists differ (PINN={pinn_seeds.tolist()}, "
            f"QCPINN={qcpinn_seeds.tolist()})."
        )

    for key in ("epochs_adam", "epochs_lbfgs"):
        pinn_epochs = _required_int(pinn, key, "PINN")
        qcpinn_epochs = _required_int(qcpinn, key, "QCPINN")
        if pinn_epochs != qcpinn_epochs:
            raise ValueError(f"{key} differs (PINN={pinn_epochs}, QCPINN={qcpinn_epochs}).")


def _grid_field(data, field, values=None):
    """Return sorted grid coordinates and a 2D field for color plotting."""
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


def _color_plot(ax, data, field, title, cmap="viridis", vrange=None, values=None):
    x, y, values = _grid_field(data, field, values=values)
    if vrange is None:
        color_plot = ax.contourf(x, y, values, levels=50, cmap=cmap)
    else:
        lower, upper = vrange
        if np.isclose(lower, upper):
            lower -= 0.5
            upper += 0.5
        levels = np.linspace(lower, upper, 51)
        color_plot = ax.contourf(
            x,
            y,
            values,
            levels=levels,
            cmap=cmap,
            vmin=lower,
            vmax=upper,
            extend="both",
        )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(CAVITY_X)
    ax.set_ylim(CAVITY_Y)
    plt.colorbar(color_plot, ax=ax, shrink=0.8)


def _contour_plot(ax, x, y, values, title, cmap="viridis", vrange=None):
    """Draw a filled contour plot from one-dimensional grid coordinates."""
    if vrange is None:
        color_plot = ax.contourf(x, y, values, levels=50, cmap=cmap)
    else:
        lower, upper = vrange
        if np.isclose(lower, upper):
            lower -= 0.5
            upper += 0.5
        color_plot = ax.contourf(
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(CAVITY_X)
    ax.set_ylim(CAVITY_Y)
    plt.colorbar(color_plot, ax=ax, shrink=0.8)


def _reference_on_model_grid(cfd_data, model_data, field):
    """Interpolate a CFD cell-centered field onto a model evaluation grid."""
    model_x, model_y, model_values = _grid_field(model_data, field)
    points = np.stack(
        np.meshgrid(model_y, model_x, indexing="ij"), axis=-1
    ).reshape(-1, 2)
    interpolator = RegularGridInterpolator(
        (_arr(cfd_data, "y"), _arr(cfd_data, "x")),
        _arr(cfd_data, field),
        bounds_error=False,
        fill_value=None,
    )
    reference_values = interpolator(points).reshape(model_values.shape)
    return model_x, model_y, model_values, reference_values


def _reference_metrics(cfd_data, model_data):
    """Compute field errors on the common interior of CFD and model grids."""
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
    metrics["l2_relative_speed"] = float(
        np.linalg.norm((model_speed - reference_speed)[interior])
        / max(float(np.linalg.norm(reference_speed[interior])), 1.0e-14)
    )
    return metrics, fields, interior


def _plot_cfd_reference(cfd_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = _arr(cfd_data, "x")
    y = _arr(cfd_data, "y")
    for ax, field, title in zip(
        axes,
        ("u", "v", "p"),
        ("u velocity -- CFD", "v velocity -- CFD", "Pressure -- CFD"),
    ):
        values = _arr(cfd_data, field)
        value_range = (
            np.percentile(values, [2.0, 98.0]) if field == "p" else None
        )
        _contour_plot(ax, x, y, values, title, vrange=value_range)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "cfd_reference_fields.png", dpi=150)
    plt.close(fig)
    print("  -> cfd_reference_fields.png")


def _plot_model_cfd_errors(label, model_data, cfd_data):
    _, fields, interior = _reference_metrics(cfd_data, model_data)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x, y, _, _ = _reference_on_model_grid(cfd_data, model_data, "u")
    for ax, field, title in zip(
        axes,
        ("u", "v", "p"),
        (f"{label} - CFD: u", f"{label} - CFD: v", f"{label} - CFD: p"),
    ):
        model_values, reference_values = fields[field]
        difference = np.where(interior, model_values - reference_values, np.nan)
        value_range = _shared_range(difference[np.isfinite(difference)])
        _contour_plot(ax, x, y, difference, title, cmap="RdBu_r", vrange=value_range)
    plt.tight_layout()
    filename = f"compare_{label.lower()}_cfd_errors.png"
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  -> {filename}")


def _plot_field(filename, field, title, cmap="viridis"):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if field == "continuity_residual":
        value_range = _shared_range(
            np.abs(_arr(pinn_data, field)), np.abs(_arr(qc_data, field))
        )
    else:
        value_range = _shared_range(_arr(pinn_data, field), _arr(qc_data, field))
    for ax, (label, data, _color) in zip(axes, models):
        if data is not None:
            values = _arr(data, field)
            if field == "continuity_residual":
                values = np.abs(values)
            _color_plot(
                ax,
                data,
                field,
                f"{title} -- {label}",
                cmap=cmap,
                vrange=value_range,
                values=values,
            )
        else:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  -> {filename}")


pinn_data = _load(PINN_RESULTS_FILE)
qc_data = _load(QCPINN_RESULTS_FILE)
cfd_data = _load(RESULTS_DIR / CFD_REFERENCE_FILENAME)

if pinn_data is None and qc_data is None and cfd_data is None:
    print("No PINN, QCPINN, or CFD reference results found. Run the benchmarks first.")
    raise SystemExit(1)

if pinn_data is not None and qc_data is not None:
    try:
        _validate_matching_runs(pinn_data, qc_data)
    except ValueError as exc:
        print(f"[ERROR] Cannot compare results: {exc}")
        raise SystemExit(1)

models = [("PINN", pinn_data, "C0"), ("QCPINN", qc_data, "C1")]

print("=" * 90)
print("QCPINN vs PINN vs CFD Benchmark -- 2D Lid-Driven Cavity Flow (Re=10)")
print("=" * 90)
header = f"{'Metric':<30} {'PINN':>24} {'QCPINN':>24} {'Delta':>10}"
print(header)
print("-" * len(header))

pinn_params = _scalar(pinn_data, "n_params")
qc_params = _scalar(qc_data, "n_params")
print(
    f"{'Parameters':<30} {_safe_int(pinn_params):>24} "
    f"{_safe_int(qc_params):>24}"
)

rows = [
    ("Final total loss", "final_total_loss"),
    ("Final BC loss", "final_bc_loss"),
    ("Final PDE loss", "final_pde_loss"),
    ("Max |continuity|", "max_continuity"),
    ("Max |x-momentum|", "max_x_momentum"),
    ("Max |y-momentum|", "max_y_momentum"),
    ("Mean |continuity|", "mean_abs_continuity"),
    ("Mean |x-momentum|", "mean_abs_x_momentum"),
    ("Mean |y-momentum|", "mean_abs_y_momentum"),
    ("Adam time (s)", "adam_time_s"),
    ("L-BFGS time (s)", "lbfgs_time_s"),
    ("Total time (s)", "total_time_s"),
]
for description, key in rows:
    pinn_mean = _scalar(pinn_data, f"{key}_mean")
    pinn_std = _scalar(pinn_data, f"{key}_std")
    qc_mean = _scalar(qc_data, f"{key}_mean")
    qc_std = _scalar(qc_data, f"{key}_std")
    delta = _pct_delta(pinn_mean, qc_mean)
    delta_text = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
    print(
        f"{description:<30} {_format_mean_std(pinn_mean, pinn_std):>24} "
        f"{_format_mean_std(qc_mean, qc_std):>24} {delta_text:>10}"
    )
print("-" * len(header))
print("Interpretation: negative Delta for losses/residuals = QCPINN lower/better.")
print("                positive Delta for time = QCPINN slower.")
print("=" * 90)

reference_metrics = {}
if cfd_data is not None:
    print(
        f"CFD reference: {_safe_int(_scalar(cfd_data, 'nx'))} x "
        f"{_safe_int(_scalar(cfd_data, 'ny'))} cells, "
        f"{_safe_int(_scalar(cfd_data, 'iterations'))} SIMPLE iterations, "
        f"converged={bool(_scalar(cfd_data, 'converged'))}"
    )
    for label, data in (("PINN", pinn_data), ("QCPINN", qc_data)):
        if data is not None:
            metrics, _, _ = _reference_metrics(cfd_data, data)
            reference_metrics[label] = metrics
            print(
                f"  {label} relative L2: u={metrics['l2_relative_u']:.6e}, "
                f"v={metrics['l2_relative_v']:.6e}, "
                f"speed={metrics['l2_relative_speed']:.6e}, "
                f"p={metrics['l2_relative_p']:.6e}"
            )

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print("\nGenerating comparison plots ...")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (key, title) in zip(
    axes,
    (
        ("total_loss_history", "Total loss"),
        ("bc_loss_history", "BC loss"),
        ("pde_loss_history", "PDE loss"),
    ),
):
    for label, data, color in models:
        history = _arr(data, key)
        if len(history):
            ax.semilogy(history, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_loss_curves.png", dpi=150)
plt.close(fig)
print("  -> compare_loss_curves.png")

_plot_field("compare_u_field.png", "u", "u velocity")
_plot_field("compare_v_field.png", "v", "v velocity")
_plot_field("compare_p_field.png", "p", "Pressure")
_plot_field(
    "compare_continuity_residual.png",
    "continuity_residual",
    "|continuity residual|",
)

if cfd_data is not None:
    _plot_cfd_reference(cfd_data)
    for label, data in (("PINN", pinn_data), ("QCPINN", qc_data)):
        if data is not None:
            _plot_model_cfd_errors(label, data, cfd_data)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for label, data, color in models:
    if data is None:
        continue
    vertical_y = _arr(data, "vertical_y")
    vertical_u = _arr(data, "vertical_u")
    horizontal_x = _arr(data, "horizontal_x")
    horizontal_v = _arr(data, "horizontal_v")
    if len(vertical_y):
        order = np.argsort(vertical_y)
        axes[0].plot(vertical_u[order], vertical_y[order], label=label, color=color)
    if len(horizontal_x):
        order = np.argsort(horizontal_x)
        axes[1].plot(horizontal_x[order], horizontal_v[order], label=label, color=color)
if cfd_data is not None:
    axes[0].plot(
        _arr(cfd_data, "vertical_u"),
        _arr(cfd_data, "vertical_y"),
        label="CFD",
        color="black",
        linestyle="--",
    )
    axes[1].plot(
        _arr(cfd_data, "horizontal_x"),
        _arr(cfd_data, "horizontal_v"),
        label="CFD",
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
    ax.legend()
    ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_centerline_profiles.png", dpi=150)
plt.close(fig)
print("  -> compare_centerline_profiles.png")

print("\nWriting Markdown report ...")
pinn_seeds = _arr(pinn_data, "seeds")
qc_seeds = _arr(qc_data, "seeds")
report_seeds = pinn_seeds if len(pinn_seeds) else qc_seeds
pinn_med_idx = int(_scalar(pinn_data, "median_run_idx", 0))
qc_med_idx = int(_scalar(qc_data, "median_run_idx", 0))
interior_resolution = (
    INTERIOR_POINTS[0]
    if len(INTERIOR_POINTS) == 1 and isinstance(INTERIOR_POINTS[0], (list, tuple))
    else INTERIOR_POINTS
)
interior_count = (
    int(np.prod(interior_resolution))
    if len(interior_resolution) > 1
    else int(interior_resolution[0])
)
sampling_text = (
    f"- **Sampling**: uniform -- {sum(BOUNDARY_POINTS)} boundary points, "
    f"{interior_count} interior points"
)
resampling_text = (
    "- **Resampling**: disabled (the collocation set is fixed)"
    if RESAMPLE_EVERY is None
    else f"- **Resampling**: uniform resample every {RESAMPLE_EVERY} L-BFGS epochs"
)
training_text = (
    f"- **Training**: L-BFGS({EPOCHS_LBFGS} epochs, "
    f"threshold={THRESHOLD_LBFGS}); no Adam warm-up"
    if EPOCHS_ADAM == 0
    else f"- **Training**: Adam(lr={LR_ADAM}, {EPOCHS_ADAM} epochs, "
         f"threshold={THRESHOLD_ADAM}) -> L-BFGS({EPOCHS_LBFGS} epochs, "
         f"threshold={THRESHOLD_LBFGS})"
)

lines = [
    "# Benchmark Report: QCPINN vs PINN vs CFD",
    "",
    "## 2D Lid-Driven Cavity Flow (Re=10)",
    "",
    f"- **Geometry**: unit square [{CAVITY_X[0]}, {CAVITY_X[1]}] x "
    f"[{CAVITY_Y[0]}, {CAVITY_Y[1]}]",
    f"- **PDE**: 2D steady incompressible Navier-Stokes; Re = {REYNOLDS} "
    f"(mu={MU}, rho=1, U={LID_VELOCITY}, L=1)",
    "- **Boundary conditions**:",
    "  - Left, bottom, and right walls: no-slip (u=v=0)",
    f"  - Top lid: u={LID_VELOCITY}, v=0",
    "  - Pressure reference: p=0 at the lower-left corner",
    sampling_text,
    resampling_text,
    training_text,
    "- **Loss**: `df.calc_loss_simple` (unweighted BC + PDE sum)",
    f"- **Seeds**: {report_seeds.tolist() if len(report_seeds) else SEEDS}",
    "- **Formulation note**: this is a DeepFlow-compatible direct `(u, v, p)` "
    "adaptation. The referenced paper learns `(psi, p)` and derives velocity "
    "from the stream function.",
    f"- **Runs per model**: PINN = {_safe_int(_scalar(pinn_data, 'num_runs'), '?')}, "
    f"QCPINN = {_safe_int(_scalar(qc_data, 'num_runs'), '?')} "
    "(median-loss run used for representative fields)",
    "",
    "## Network Architectures",
    "",
    "| Model | Architecture | Parameters |",
    "|-------|--------------|------------|",
    f"| **PINN** | `PINN(width={PINN_WIDTH}, length={PINN_LENGTH})` -- "
    f"{PINN_LENGTH}x{PINN_WIDTH}-neuron hidden layers, Tanh | {_safe_int(pinn_params)} |",
    f"| **QCPINN** | `QCPINN(pre={QC_PRE}, post={QC_POST}, "
    f"nqubits={QC_NQUBITS}, q_layer_iterations={QC_ITERATIONS})` | "
    f"{_safe_int(qc_params)} |",
    "",
    "## Summary Table",
    "",
    "| Metric | PINN | QCPINN | Delta |",
    "|--------|------|--------|-------|",
    f"| Parameters | {_safe_int(pinn_params)} | {_safe_int(qc_params)} | -- |",
]

for description, key in rows:
    pinn_mean = _scalar(pinn_data, f"{key}_mean")
    pinn_std = _scalar(pinn_data, f"{key}_std")
    qc_mean = _scalar(qc_data, f"{key}_mean")
    qc_std = _scalar(qc_data, f"{key}_std")
    delta = _pct_delta(pinn_mean, qc_mean)
    delta_text = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
    lines.append(
        f"| {description} | {_format_mean_std(pinn_mean, pinn_std)} | "
        f"{_format_mean_std(qc_mean, qc_std)} | {delta_text} |"
    )

if cfd_data is not None:
    lines += [
        "",
        "## Finite-Volume CFD Reference",
        "",
        "The independent reference uses a staggered-grid finite-volume SIMPLE "
        "solver with central-difference convection and diffusion.",
        f"- **Grid**: {_safe_int(_scalar(cfd_data, 'nx'))} x "
        f"{_safe_int(_scalar(cfd_data, 'ny'))} cells",
        f"- **Iterations**: {_safe_int(_scalar(cfd_data, 'iterations'))}",
        f"- **Final normalized residual**: "
        f"{_scalar(cfd_data, 'final_residual'):.6e}",
        f"- **Converged**: {bool(_scalar(cfd_data, 'converged'))}",
        "",
        "### Model Error Against CFD",
        "",
        "| Model | Relative L2 u | Relative L2 v | Relative L2 speed | Relative L2 p |",
        "|-------|---------------:|---------------:|------------------:|--------------:|",
    ]
    for label in ("PINN", "QCPINN"):
        if label in reference_metrics:
            metrics = reference_metrics[label]
            lines.append(
                f"| {label} | {metrics['l2_relative_u']:.6e} | "
                f"{metrics['l2_relative_v']:.6e} | "
                f"{metrics['l2_relative_speed']:.6e} | "
                f"{metrics['l2_relative_p']:.6e} |"
            )
    convergence_path = RESULTS_DIR / CFD_GRID_CONVERGENCE_FILENAME
    if convergence_path.is_file():
        with np.load(convergence_path) as convergence:
            lines += [
                "",
                "### CFD Grid Convergence",
                "",
                "The 101x101 and 201x201 solutions are compared by interpolating "
                "the refined fields onto the coarse grid.",
                f"- **Relative L2 u**: {float(convergence['l2_relative_u']):.6e}",
                f"- **Relative L2 v**: {float(convergence['l2_relative_v']):.6e}",
                f"- **Relative L2 p**: {float(convergence['l2_relative_p']):.6e}",
                f"- **u centerline RMSE**: {float(convergence['rmse_vertical_u']):.6e}",
                f"- **v centerline RMSE**: {float(convergence['rmse_horizontal_v']):.6e}",
            ]

lines += [
    "",
    "## Generated Figures",
    "",
    "- `compare_loss_curves.png` -- total / BC / PDE loss curves",
    "- `compare_u_field.png` -- u velocity field",
    "- `compare_v_field.png` -- v velocity field",
    "- `compare_p_field.png` -- pressure field",
    "- `compare_continuity_residual.png` -- continuity residual field",
    "- `compare_centerline_profiles.png` -- cavity centerline velocity profiles",
]
if cfd_data is not None:
    lines += [
        "- `cfd_reference_fields.png` -- finite-volume CFD u, v, and p fields",
        "- `compare_pinn_cfd_errors.png` -- PINN-minus-CFD field errors",
        "- `compare_qcpinn_cfd_errors.png` -- QCPINN-minus-CFD field errors, when available",
    ]
    if (RESULTS_DIR / CFD_GRID_CONVERGENCE_FILENAME).is_file():
        lines.append("- `cfd_grid_convergence.npz` -- 101x101 versus 201x201 differences")
lines += [
    "",
    "## Reproducibility",
    "",
    f"- Median run index: PINN = {pinn_med_idx}, QCPINN = {qc_med_idx}",
    f"- Per-run final total losses: PINN = {_arr(pinn_data, 'final_total_loss_runs')}, "
    f"QCPINN = {_arr(qc_data, 'final_total_loss_runs')}",
    "",
    "---",
    "*Report generated by `compare.py`*",
]

REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"  -> {REPORT_FILE}")
print("\nDone.")
