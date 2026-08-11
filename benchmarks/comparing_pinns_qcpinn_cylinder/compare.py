"""Compare the four cylinder benchmark cells and write figures/report."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from common_config import (
    BOUNDARY_POINTS, CHANNEL_X, CHANNEL_Y, CYLINDER_CX, CYLINDER_CY,
    CYLINDER_R, EPOCHS_ADAM, EPOCHS_LBFGS, FEM_REFERENCE_FILENAME,
    INTERIOR_POINTS, MU, REYNOLDS, REPORT_FILE, RESULTS_DIR, U_INF,
)

SETUPS = (
    ("PINN-UVP", "pinn_uvp_results.npz", "C0"),
    ("QCPINN-UVP", "qcpinn_uvp_results.npz", "C1"),
    ("PINN-PSIP", "pinn_psip_results.npz", "C2"),
    ("QCPINN-PSIP", "qcpinn_psip_results.npz", "C3"),
)


def _load(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark result: {path}")
    return np.load(path)


def _arr(data, key):
    return np.asarray(data[key]) if key in data.files else np.array([])


def _scalar(data, key, default=float("nan")):
    values = _arr(data, key)
    return default if values.size == 0 else float(values.flat[0])


def _text(data, key, default="N/A"):
    values = _arr(data, key)
    return default if values.size == 0 else str(values.item())


def _safe_int(value):
    return "N/A" if np.isnan(value) else str(int(value))


def _format(value, std):
    return f"{value:.6e} +/- {std:.6e}" if std > 0 else f"{value:.6e}"


def _shared_range(*arrays, symmetric=False):
    values = []
    for array in arrays:
        flat = np.asarray(array).reshape(-1)
        flat = flat[np.isfinite(flat)]
        if flat.size:
            values.append(flat)
    if not values:
        return (-1.0, 1.0)
    values = np.concatenate(values)
    if symmetric:
        radius = max(float(np.max(np.abs(values))), 1.0e-14)
        return -radius, radius
    lower, upper = float(np.min(values)), float(np.max(values))
    return (lower - 0.5, upper + 0.5) if np.isclose(lower, upper) else (lower, upper)


def _field_values(data, field):
    if field == "velocity_magnitude":
        return np.sqrt(_arr(data, "u") ** 2 + _arr(data, "v") ** 2)
    return _arr(data, field)


def _grid_field(data, field, values=None):
    x = _arr(data, "x").reshape(-1)
    y = _arr(data, "y").reshape(-1)
    values = _field_values(data, field) if values is None else np.asarray(values).reshape(-1)
    x_values, y_values = np.unique(x), np.unique(y)
    grid = np.full((y_values.size, x_values.size), np.nan, dtype=float)
    xi = np.searchsorted(x_values, x)
    yi = np.searchsorted(y_values, y)
    grid[yi, xi] = values
    return x_values, y_values, np.ma.masked_invalid(grid)


def _contour(ax, x, y, values, title, cmap, vrange, labels=True):
    plot = ax.contourf(x, y, values, levels=np.linspace(*vrange, 51), cmap=cmap)
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_xlim(CHANNEL_X)
    ax.set_ylim(CHANNEL_Y)
    if labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    return plot


def _validate(results):
    metadata = None
    for label, data, _ in results:
        required = ("num_runs", "epochs_adam", "epochs_lbfgs", "seeds", "n_params")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f"{label} results are missing {missing}")
        current = (
            _scalar(data, "num_runs"), _scalar(data, "epochs_adam"),
            _scalar(data, "epochs_lbfgs"), _arr(data, "seeds").reshape(-1),
        )
        if metadata is None:
            metadata = current
        elif current[:3] != metadata[:3] or not np.array_equal(current[3], metadata[3]):
            raise ValueError("All four setups must use matching run metadata and seeds.")


def _plot_solution_fields(results):
    fields = ("u", "v", "velocity_magnitude", "p")
    titles = ("u velocity", "v velocity", "Velocity magnitude", "Pressure")
    ranges = [_shared_range(*[_field_values(d, f) for _, d, _ in results]) for f in fields]
    fig, axes = plt.subplots(4, 4, figsize=(18, 16), squeeze=False, constrained_layout=True)
    for row, (label, data, _) in enumerate(results):
        for col, (field, title) in enumerate(zip(fields, titles)):
            x, y, grid = _grid_field(data, field)
            mappable = _contour(axes[row, col], x, y, grid, f"{title} -- {label}", "viridis", ranges[col])
            if row == 0:
                fig.colorbar(mappable, ax=axes[:, col].tolist(), shrink=0.8)
    fig.savefig(RESULTS_DIR / "compare_solution_fields.png", dpi=150)
    plt.close(fig)


def _plot_residuals(results):
    fields = ("continuity_residual", "x_momentum_residual", "y_momentum_residual")
    titles = ("|continuity|", "x-momentum", "y-momentum")
    ranges = [_shared_range(*[
        np.abs(_arr(d, f)) if i == 0 else _arr(d, f)
        for _, d, _ in results
    ], symmetric=i != 0) for i, f in enumerate(fields)]
    fig, axes = plt.subplots(4, 3, figsize=(15, 16), squeeze=False, constrained_layout=True)
    for row, (label, data, _) in enumerate(results):
        for col, (field, title) in enumerate(zip(fields, titles)):
            values = np.abs(_arr(data, field)) if col == 0 else _arr(data, field)
            x, y, grid = _grid_field(data, field, values)
            mappable = _contour(axes[row, col], x, y, grid, f"{title} -- {label}", "magma" if col == 0 else "RdBu_r", ranges[col])
            if row == 0:
                fig.colorbar(mappable, ax=axes[:, col].tolist(), shrink=0.8)
    fig.savefig(RESULTS_DIR / "compare_residual_fields.png", dpi=150)
    plt.close(fig)


def _plot_losses(results):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    for ax, key, title in zip(axes, ("total_loss_history", "bc_loss_history", "pde_loss_history"), ("Total loss", "BC loss", "PDE loss")):
        for label, data, color in results:
            history = _arr(data, key)
            if history.size:
                ax.semilogy(np.maximum(history, 1.0e-16), label=label, color=color)
        ax.set_title(title)
        ax.set_xlabel("L-BFGS epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, which="both", ls="--", alpha=0.5)
        ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / "compare_loss_curves.png", dpi=150)
    plt.close(fig)


def _plot_profiles(results, reference):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for label, data, color in results:
        order = np.argsort(_arr(data, "outlet_y"))
        axes[0].plot(_arr(data, "outlet_u")[order], _arr(data, "outlet_y")[order], label=label, color=color)
        order = np.argsort(_arr(data, "wake_x"))
        axes[1].plot(_arr(data, "wake_x")[order], _arr(data, "wake_u")[order], label=label, color=color)
    if reference is not None:
        axes[0].plot(_arr(reference, "outlet_u"), _arr(reference, "outlet_y"), "k--", label="FEM")
        axes[1].plot(_arr(reference, "wake_x"), _arr(reference, "wake_u"), "k--", label="FEM")
    axes[0].set(xlabel="u", ylabel="y", title="Outlet profile: u(y) at x=1.1")
    axes[1].set(xlabel="x", ylabel="u", title="Wake centerline: u(x) at y=0.2")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(RESULTS_DIR / "compare_flow_profiles.png", dpi=150)
    plt.close(fig)


def _reference_on_model_grid(reference, model, field):
    cx, cy, cgrid = _grid_field(reference, field)
    mx, my, mgrid = _grid_field(model, field)
    interpolator = RegularGridInterpolator(
        (cy, cx), cgrid.filled(np.nan), bounds_error=False, fill_value=np.nan
    )
    yy, xx = np.meshgrid(my, mx, indexing="ij")
    ref_grid = interpolator(np.stack((yy, xx), axis=-1)).reshape(mgrid.shape)
    valid = (~np.ma.getmaskarray(mgrid)) & np.isfinite(ref_grid)
    return mx, my, mgrid.filled(np.nan), ref_grid, valid


def _reference_metrics(reference, model):
    metrics = {}
    for field in ("u", "v", "p"):
        _, _, predicted, expected, valid = _reference_on_model_grid(reference, model, field)
        difference = predicted[valid] - expected[valid]
        metrics[f"l2_relative_{field}"] = float(
            np.linalg.norm(difference) / max(np.linalg.norm(expected[valid]), 1.0e-14)
        )
    _, _, u, ur, valid_u = _reference_on_model_grid(reference, model, "u")
    _, _, v, vr, valid_v = _reference_on_model_grid(reference, model, "v")
    valid = valid_u & valid_v
    speed, ref_speed = np.sqrt(u**2 + v**2), np.sqrt(ur**2 + vr**2)
    metrics["l2_relative_speed"] = float(
        np.linalg.norm((speed - ref_speed)[valid]) / max(np.linalg.norm(ref_speed[valid]), 1.0e-14)
    )
    for prefix, xkey, ykey, rxkey, rfield in (
        ("outlet", "outlet_y", "outlet_u", "outlet_y", "outlet_u"),
        ("wake", "wake_x", "wake_u", "wake_x", "wake_u"),
    ):
        order = np.argsort(_arr(model, xkey))
        prediction = np.interp(_arr(reference, rxkey), _arr(model, xkey)[order], _arr(model, ykey)[order])
        metrics[f"rmse_{prefix}_u"] = float(np.sqrt(np.mean((prediction - _arr(reference, rfield)) ** 2)))
    return metrics


def _plot_fem_errors(results, reference):
    fields = ("velocity_magnitude", "v", "p")
    titles = ("Velocity magnitude", "v velocity", "Pressure")
    ref_ranges = [_shared_range(_field_values(reference, field)) for field in fields]
    errors = []
    for label, data, _ in results:
        current = {}
        for field in fields:
            _, _, predicted, expected, valid = _reference_on_model_grid(reference, data, field)
            current[field] = np.ma.masked_where(~valid, predicted - expected)
        errors.append((label, data, current))
    error_ranges = [_shared_range(*[e[2][f] for e in errors], symmetric=True) for f in fields]
    fig, axes = plt.subplots(5, 3, figsize=(16, 20), squeeze=False, constrained_layout=True)
    for col, (field, title) in enumerate(zip(fields, titles)):
        x, y, grid = _grid_field(reference, field)
        mappable = _contour(axes[0, col], x, y, grid, f"FEM -- {title}", "viridis", ref_ranges[col])
        fig.colorbar(mappable, ax=axes[0, col], shrink=0.8)
    for row, (label, data, current) in enumerate(errors, start=1):
        for col, (field, title) in enumerate(zip(fields, titles)):
            x, y, _ = _grid_field(data, field)
            mappable = _contour(axes[row, col], x, y, current[field], f"{label} - FEM -- {title}", "RdBu_r", error_ranges[col])
            if row == 1:
                fig.colorbar(mappable, ax=axes[1:, col].tolist(), shrink=0.8)
    fig.savefig(RESULTS_DIR / "compare_fem_reference_and_errors.png", dpi=150)
    plt.close(fig)


def _write_report(results, reference, metrics):
    first = results[0][1]
    runs = _arr(first, "seeds").tolist()
    interior_count = int(np.prod(INTERIOR_POINTS[0]))
    training = f"L-BFGS({EPOCHS_LBFGS}) with no Adam warm-up" if EPOCHS_ADAM == 0 else f"Adam({EPOCHS_ADAM}) -> L-BFGS({EPOCHS_LBFGS})"
    lines = [
        "# Benchmark Report: PINN/QCPINN x UVP/PSIP",
        "",
        f"## 2D Steady Cylinder Flow (Re={REYNOLDS:g})",
        "",
        f"- **Geometry**: channel [{CHANNEL_X[0]}, {CHANNEL_X[1]}] x [{CHANNEL_Y[0]}, {CHANNEL_Y[1]}], cylinder center ({CYLINDER_CX}, {CYLINDER_CY}), radius {CYLINDER_R}.",
        f"- **PDE**: steady incompressible Navier-Stokes; U={U_INF}, rho=1, mu={MU}, L=1.",
        "- **UVP BCs**: parabolic inlet, no-slip channel walls/cylinder, and p=0 at the outlet.",
        "- **PSIP BCs**: psi_y equals the inlet profile, psi_x=0 at the inlet, zero velocity derivatives on solid boundaries, and p=0 at the outlet.",
        f"- **Sampling**: uniform -- {sum(BOUNDARY_POINTS)} boundary points, {interior_count} interior points.",
        f"- **Training**: {training}; seeds={runs}.",
        "- **PSIP note**: u=psi_y and v=-psi_x, so continuity is satisfied analytically.",
        "",
        "## Network Architectures",
        "",
        "| Setup | Architecture | Parameters | PDE residuals |",
        "|---|---|---:|---:|",
    ]
    for label, data, _ in results:
        lines.append(f"| {label} | `{_text(data, 'network_description')}` | {_safe_int(_scalar(data, 'n_params'))} | {_safe_int(_scalar(data, 'pde_residual_count'))} |")
    lines += ["", "## Training and Residual Summary", "", "| Setup | Total loss | PDE loss | PDE/residual | Max continuity | Total time (s) |", "|---|---:|---:|---:|---:|---:|"]
    for label, data, _ in results:
        lines.append(f"| {label} | {_format(_scalar(data, 'final_total_loss_mean'), _scalar(data, 'final_total_loss_std'))} | {_format(_scalar(data, 'final_pde_loss_mean'), _scalar(data, 'final_pde_loss_std'))} | {_format(_scalar(data, 'pde_loss_per_equation_mean'), _scalar(data, 'pde_loss_per_equation_std'))} | {_format(_scalar(data, 'max_continuity_mean'), _scalar(data, 'max_continuity_std'))} | {_format(_scalar(data, 'total_time_s_mean'), _scalar(data, 'total_time_s_std'))} |")
    if metrics:
        lines += ["", "## Error Against Fresh FEM Reference", "", "| Setup | Relative L2 u | Relative L2 v | Relative L2 speed | Relative L2 p | Outlet RMSE | Wake RMSE |", "|---|---:|---:|---:|---:|---:|---:|"]
        for label, _, _ in results:
            m = metrics[label]
            lines.append(f"| {label} | {m['l2_relative_u']:.6e} | {m['l2_relative_v']:.6e} | {m['l2_relative_speed']:.6e} | {m['l2_relative_p']:.6e} | {m['rmse_outlet_u']:.6e} | {m['rmse_wake_u']:.6e} |")
        lines += ["", "### FEM Reference Diagnostics", "", f"- Grid: {_safe_int(_scalar(reference, 'nx'))} x {_safe_int(_scalar(reference, 'ny'))}", f"- Mesh size: {_scalar(reference, 'mesh_size'):.6g}", f"- Elements: {_safe_int(_scalar(reference, 'mesh_elements'))}", f"- Iterations: {_safe_int(_scalar(reference, 'iterations'))}", f"- Final residual: {_scalar(reference, 'final_residual'):.6e}", f"- Pressure gauge: {_text(reference, 'pressure_gauge')}"]
    lines += ["", "## Generated Figures", "", "- `compare_solution_fields.png`", "- `compare_residual_fields.png`", "- `compare_loss_curves.png`", "- `compare_flow_profiles.png`", "- `compare_fem_reference_and_errors.png`", "", "---", "*Report generated by `compare.py`*"]
    REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = [(label, _load(RESULTS_DIR / filename), color) for label, filename, color in SETUPS]
    _validate(results)
    reference_path = RESULTS_DIR / FEM_REFERENCE_FILENAME
    reference = _load(reference_path) if reference_path.is_file() else None
    print("=" * 100)
    print(f"PINN/QCPINN x UVP/PSIP -- Re={REYNOLDS:g} Cylinder Benchmark")
    print("=" * 100)
    for label, data, _ in results:
        print(f"{label:<14} params={_safe_int(_scalar(data, 'n_params')):>5} total={_scalar(data, 'final_total_loss_mean'):.6e} pde={_scalar(data, 'final_pde_loss_mean'):.6e}")
    metrics = {}
    if reference is not None:
        print(f"FEM reference converged={bool(_scalar(reference, 'converged'))}")
        for label, data, _ in results:
            metrics[label] = _reference_metrics(reference, data)
            print(f"  {label}: L2 speed={metrics[label]['l2_relative_speed']:.6e}")
    _plot_solution_fields(results)
    _plot_residuals(results)
    _plot_losses(results)
    _plot_profiles(results, reference)
    if reference is not None:
        _plot_fem_errors(results, reference)
    _write_report(results, reference, metrics)
    print(f"Artifacts written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
