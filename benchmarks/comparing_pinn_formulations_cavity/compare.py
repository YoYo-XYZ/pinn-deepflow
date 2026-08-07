#!/usr/bin/env python3
"""Compare direct and stream-function PINN results for the Re=10 cavity."""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    CAVITY_X,
    CAVITY_Y,
    CFD_REFERENCE_FILE,
    INTERIOR_POINTS,
    LID_VELOCITY,
    MU,
    REPORT_FILE,
    RESULTS_DIR,
    REYNOLDS,
    SEEDS,
    UVP_RESULTS_FILE,
    PSIP_RESULTS_FILE,
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


def _text(data, key, default="N/A"):
    if data is None or key not in data.files or data[key].size == 0:
        return default
    return str(data[key].flat[0])


def _arr(data, key):
    if data is None or key not in data.files:
        return np.array([])
    return data[key]


def _safe_int(value, default="N/A"):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return default


def _format_mean_std(mean, std):
    if std > 0:
        return f"{mean:.6e} +/- {std:.6e}"
    return f"{mean:.6e}"


def _pct_delta(a, b):
    if a == 0 or np.isnan(a) or np.isnan(b):
        return float("nan")
    return (b - a) / a * 100.0


def _shared_range(*arrays):
    values = [np.asarray(array).ravel() for array in arrays if np.asarray(array).size]
    if not values:
        return None
    values = np.concatenate(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    return float(np.min(values)), float(np.max(values))


def _validate_matching_runs(uvp, psip):
    for key in ("num_runs", "epochs_adam", "epochs_lbfgs"):
        if _scalar(uvp, key) != _scalar(psip, key):
            raise ValueError(
                f"{key} differs (UVP={_scalar(uvp, key)}, "
                f"PSIP={_scalar(psip, key)})."
            )
    uvp_seeds = _arr(uvp, "seeds")
    psip_seeds = _arr(psip, "seeds")
    if not np.array_equal(uvp_seeds, psip_seeds):
        raise ValueError(
            f"seeds differ (UVP={uvp_seeds.tolist()}, PSIP={psip_seeds.tolist()})."
        )


def _grid_field(data, field, values=None):
    x = np.asarray(_arr(data, "x")).reshape(-1)
    y = np.asarray(_arr(data, "y")).reshape(-1)
    values = np.asarray(_arr(data, field) if values is None else values).reshape(-1)
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(CAVITY_X)
    ax.set_ylim(CAVITY_Y)
    plt.colorbar(plot, ax=ax, shrink=0.8)


def _plot_field(filename, field, title, uvp, psip, cmap="viridis"):
    models = [("UVP", uvp), ("PSIP", psip)]
    plotted_values = []
    for _, data in models:
        if data is not None and field in data.files:
            values = np.abs(_arr(data, field)) if field == "continuity_residual" else _arr(data, field)
            plotted_values.append(values)
    value_range = _shared_range(*plotted_values)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (label, data) in zip(axes, models):
        if data is None or field not in data.files:
            ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{title} -- {label}")
            continue
        values = np.abs(_arr(data, field)) if field == "continuity_residual" else _arr(data, field)
        _color_plot(ax, data, field, f"{title} -- {label}", cmap, value_range, values)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  -> {filename}")


def _plot_single_field(filename, field, title, data, cmap="viridis"):
    if data is None or field not in data.files:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    _color_plot(ax, data, field, title, cmap=cmap)
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)
    print(f"  -> {filename}")


def _reference_on_model_grid(cfd_data, model_data, field):
    model_x, model_y, model_values = _grid_field(model_data, field)
    points = np.stack(np.meshgrid(model_y, model_x, indexing="ij"), axis=-1).reshape(-1, 2)
    interpolator = RegularGridInterpolator(
        (_arr(cfd_data, "y"), _arr(cfd_data, "x")),
        _arr(cfd_data, field),
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
    metrics["l2_relative_speed"] = float(
        np.linalg.norm((model_speed - reference_speed)[interior])
        / max(float(np.linalg.norm(reference_speed[interior])), 1.0e-14)
    )
    return metrics, fields, interior


def _contour_plot(ax, x, y, values, title, cmap="viridis", vrange=None):
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
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(CAVITY_X)
    ax.set_ylim(CAVITY_Y)
    plt.colorbar(plot, ax=ax, shrink=0.8)


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
        value_range = np.percentile(values, [2.0, 98.0]) if field == "p" else None
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


def _plot_loss_curves(uvp, psip):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, title in zip(
        axes,
        ("total_loss_history", "bc_loss_history", "pde_loss_history"),
        ("Total loss", "BC loss", "Raw PDE loss"),
    ):
        for label, data, color in (("UVP", uvp, "C0"), ("PSIP", psip, "C1")):
            history = _arr(data, key)
            if len(history):
                ax.semilogy(history, label=label, color=color)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend()
    plt.tight_layout()
    fig.savefig(RESULTS_DIR / "compare_loss_curves.png", dpi=150)
    plt.close(fig)
    print("  -> compare_loss_curves.png")


def _plot_centerlines(uvp, psip, cfd_data):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for label, data, color in (("UVP", uvp, "C0"), ("PSIP", psip, "C1")):
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


uvp_data = _load(UVP_RESULTS_FILE)
psip_data = _load(PSIP_RESULTS_FILE)
cfd_data = _load(CFD_REFERENCE_FILE)

if uvp_data is None or psip_data is None:
    print("Both UVP and PSIP result files are required. Run the model benchmarks first.")
    raise SystemExit(1)

try:
    _validate_matching_runs(uvp_data, psip_data)
except ValueError as exc:
    print(f"[ERROR] Cannot compare results: {exc}")
    raise SystemExit(1)

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 95)
print("Direct (u,v,p) vs Stream-Function (psi,p) -- Re=10 Cavity Benchmark")
print("=" * 95)
header = f"{'Metric':<30} {'UVP':>24} {'PSIP':>24} {'Delta':>10}"
print(header)
print("-" * len(header))
print(
    f"{'Parameters':<30} {_safe_int(_scalar(uvp_data, 'n_params')):>24} "
    f"{_safe_int(_scalar(psip_data, 'n_params')):>24}"
)

rows = [
    ("Final total loss (raw)", "final_total_loss"),
    ("Final BC loss", "final_bc_loss"),
    ("Final PDE loss (raw)", "final_pde_loss"),
    ("PDE loss per residual", "pde_loss_per_equation"),
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
    uvp_mean = _scalar(uvp_data, f"{key}_mean")
    uvp_std = _scalar(uvp_data, f"{key}_std")
    psip_mean = _scalar(psip_data, f"{key}_mean")
    psip_std = _scalar(psip_data, f"{key}_std")
    delta = _pct_delta(uvp_mean, psip_mean)
    delta_text = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
    print(
        f"{description:<30} {_format_mean_std(uvp_mean, uvp_std):>24} "
        f"{_format_mean_std(psip_mean, psip_std):>24} {delta_text:>10}"
    )
print("-" * len(header))
print("Raw PDE totals are formulation-specific because UVP has 3 residuals and PSIP has 2.")
print("Field errors and equation-wise residuals are the primary cross-formulation metrics.")

reference_metrics = {}
if cfd_data is not None:
    print(
        f"CFD reference: {_safe_int(_scalar(cfd_data, 'nx'))} x "
        f"{_safe_int(_scalar(cfd_data, 'ny'))} cells, "
        f"converged={bool(_scalar(cfd_data, 'converged'))}"
    )
    for label, data in (("UVP", uvp_data), ("PSIP", psip_data)):
        metrics, _, _ = _reference_metrics(cfd_data, data)
        reference_metrics[label] = metrics
        print(
            f"  {label} relative L2: u={metrics['l2_relative_u']:.6e}, "
            f"v={metrics['l2_relative_v']:.6e}, "
            f"speed={metrics['l2_relative_speed']:.6e}, "
            f"p={metrics['l2_relative_p']:.6e}"
        )
else:
    print("[WARN] Shared cavity CFD reference is unavailable; skipping CFD metrics.")

print("\nGenerating comparison plots ...")
_plot_loss_curves(uvp_data, psip_data)
_plot_field("compare_u_field.png", "u", "u velocity", uvp_data, psip_data)
_plot_field("compare_v_field.png", "v", "v velocity", uvp_data, psip_data)
_plot_field("compare_p_field.png", "p", "Pressure", uvp_data, psip_data)
_plot_single_field("compare_psi_field.png", "psi", "Stream function -- PSIP", psip_data)
_plot_field(
    "compare_continuity_residual.png",
    "continuity_residual",
    "|continuity residual|",
    uvp_data,
    psip_data,
)
_plot_field(
    "compare_x_momentum_residual.png",
    "x_momentum_residual",
    "x-momentum residual",
    uvp_data,
    psip_data,
    cmap="RdBu_r",
)
_plot_field(
    "compare_y_momentum_residual.png",
    "y_momentum_residual",
    "y-momentum residual",
    uvp_data,
    psip_data,
    cmap="RdBu_r",
)
_plot_centerlines(uvp_data, psip_data, cfd_data)

if cfd_data is not None:
    _plot_cfd_reference(cfd_data)
    _plot_model_cfd_errors("UVP", uvp_data, cfd_data)
    _plot_model_cfd_errors("PSIP", psip_data, cfd_data)

print("\nWriting Markdown report ...")
interior_resolution = (
    INTERIOR_POINTS[0]
    if len(INTERIOR_POINTS) == 1 and isinstance(INTERIOR_POINTS[0], (list, tuple))
    else INTERIOR_POINTS
)
interior_count = int(np.prod(interior_resolution))
report_seeds = _arr(uvp_data, "seeds")
report_epochs_adam = _safe_int(_scalar(uvp_data, "epochs_adam"))
report_epochs_lbfgs = _safe_int(_scalar(uvp_data, "epochs_lbfgs"))

lines = [
    "# Benchmark Report: Direct `(u,v,p)` vs Stream-Function `(psi,p)`",
    "",
    "## 2D Lid-Driven Cavity Flow (Re=10)",
    "",
    f"- **Geometry**: unit square [{CAVITY_X[0]}, {CAVITY_X[1]}] x "
    f"[{CAVITY_Y[0]}, {CAVITY_Y[1]}]",
    f"- **PDE**: steady incompressible 2D Navier-Stokes; Re = {REYNOLDS} "
    f"(mu={MU}, rho=1, U={LID_VELOCITY}, L=1)",
    "- **UVP boundary conditions**: no-slip on left/bottom/right, top lid "
    "`u=1, v=0`, and `p=0` at the lower-left corner.",
    "- **PSIP boundary conditions**: left/bottom/right `psi_x=0, psi_y=0`, "
    "top lid `psi_x=0, psi_y=1`, and `p=0` at the lower-left corner.",
    f"- **Sampling**: uniform -- {sum(BOUNDARY_POINTS)} boundary points, "
    f"{interior_count} interior points",
    "- **Resampling**: disabled (fixed collocation set)",
    f"- **Training**: Adam({report_epochs_adam} epochs) -> "
    f"L-BFGS({report_epochs_lbfgs} epochs)",
    "- **Loss**: raw `df.calc_loss_simple` (unweighted BC + PDE sum)",
    f"- **Seeds**: {report_seeds.tolist() if len(report_seeds) else SEEDS}",
    "- **Interpretation**: UVP optimizes continuity plus two momentum residuals; "
    "PSIP derives `u=psi_y` and `v=-psi_x`, so continuity is identically satisfied "
    "and only the two momentum residuals enter its PDE loss.",
    f"- **Runs per model**: {_safe_int(_scalar(uvp_data, 'num_runs'))} "
    "(median-loss run used for representative fields)",
    "",
    "## Network Architectures",
    "",
    "| Model | Architecture | Parameters |",
    "|-------|--------------|------------|",
    f"| **UVP** | `{_text(uvp_data, 'network_description')}` | "
    f"{_safe_int(_scalar(uvp_data, 'n_params'))} |",
    f"| **PSIP** | `{_text(psip_data, 'network_description')}` | "
    f"{_safe_int(_scalar(psip_data, 'n_params'))} |",
    "",
    "## Summary Table",
    "",
    "| Metric | UVP | PSIP | Delta |",
    "|--------|-----|------|-------|",
    f"| Parameters | {_safe_int(_scalar(uvp_data, 'n_params'))} | "
    f"{_safe_int(_scalar(psip_data, 'n_params'))} | -- |",
]

for description, key in rows:
    uvp_mean = _scalar(uvp_data, f"{key}_mean")
    uvp_std = _scalar(uvp_data, f"{key}_std")
    psip_mean = _scalar(psip_data, f"{key}_mean")
    psip_std = _scalar(psip_data, f"{key}_std")
    delta = _pct_delta(uvp_mean, psip_mean)
    delta_text = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
    lines.append(
        f"| {description} | {_format_mean_std(uvp_mean, uvp_std)} | "
        f"{_format_mean_std(psip_mean, psip_std)} | {delta_text} |"
    )

if reference_metrics:
    lines += [
        "",
        "## Model Error Against Shared CFD Reference",
        "",
        "| Model | Relative L2 u | Relative L2 v | Relative L2 speed | Relative L2 p |",
        "|-------|---------------:|---------------:|------------------:|--------------:|",
    ]
    for label in ("UVP", "PSIP"):
        metrics = reference_metrics[label]
        lines.append(
            f"| {label} | {metrics['l2_relative_u']:.6e} | "
            f"{metrics['l2_relative_v']:.6e} | "
            f"{metrics['l2_relative_speed']:.6e} | "
            f"{metrics['l2_relative_p']:.6e} |"
        )

lines += [
    "",
    "## Generated Figures",
    "",
    "- `compare_loss_curves.png` -- total, BC, and raw PDE loss curves",
    "- `compare_u_field.png`, `compare_v_field.png`, `compare_p_field.png` -- common fields",
    "- `compare_psi_field.png` -- stream-function field",
    "- `compare_continuity_residual.png` -- continuity residual, including the PSIP identity",
    "- `compare_x_momentum_residual.png`, `compare_y_momentum_residual.png` -- momentum residuals",
    "- `compare_centerline_profiles.png` -- cavity centerline velocity profiles",
]
if cfd_data is not None:
    lines += [
        "- `cfd_reference_fields.png` -- shared finite-volume CFD fields",
        "- `compare_uvp_cfd_errors.png`, `compare_psip_cfd_errors.png` -- model-minus-CFD fields",
    ]
lines += [
    "",
    "## Reproducibility",
    "",
    f"- Median run index: UVP = {_safe_int(_scalar(uvp_data, 'median_run_idx'))}, "
    f"PSIP = {_safe_int(_scalar(psip_data, 'median_run_idx'))}",
    f"- Per-run final total losses: UVP = {_arr(uvp_data, 'final_total_loss_runs')}, "
    f"PSIP = {_arr(psip_data, 'final_total_loss_runs')}",
    f"- Shared CFD file: `{CFD_REFERENCE_FILE}`",
    "",
    "---",
    "*Report generated by `compare.py`*",
]

REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"  -> {REPORT_FILE}")
print("\nDone.")
