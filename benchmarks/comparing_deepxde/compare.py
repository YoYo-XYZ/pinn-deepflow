#!/usr/bin/env python3
"""
Load DeepFlow and DeepXDE results, generate comparison table, plots, and report.

All figures are saved to ``results/`` as PNG images.
A Markdown summary is written to ``results/benchmark_report.md``.
"""

import os
import sys
import textwrap

# ---------------------------------------------------------------------------
# Ensure imports resolve regardless of the caller's working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np

# trapezoidal integration: prefer np.trapezoid (NumPy 2.0+), fallback np.trapz
if hasattr(np, "trapezoid"):
    _trapz = np.trapezoid  # type: ignore[attr-defined]
else:
    _trapz = np.trapz  # type: ignore[attr-defined]

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from common_config import Lx, Ly, Re, WIDTH, DEPTH, LR, EPOCHS, BOUNDARY_POINTS, INTERIOR_POINTS

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def _load(npz_name: str):
    """Load an NPZ results file; returns None if missing."""
    path = os.path.join(_RESULTS_DIR, npz_name)
    if not os.path.isfile(path):
        print(f"[WARN] {path} not found.")
        return None
    return np.load(path)


# ===========================================================================
# 1. Load data
# ===========================================================================
df_data = _load("deepflow_results.npz")
dx_data = _load("deepxde_results.npz")

if df_data is None and dx_data is None:
    print("Neither DeepFlow nor DeepXDE results found. Run the benchmarks first.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Helper: safely extract a scalar or array
# ---------------------------------------------------------------------------
def _get(d, key, default=None):
    if d is None:
        return default
    arr = d.get(key)
    if arr is None:
        return default
    return arr


def _scalar(d, key, default=float("nan")):
    v = _get(d, key)
    if v is None or v.size == 0:
        return default
    return float(v.flat[0])


def _arr(d, key):
    return _get(d, key, np.array([]))


# ===========================================================================
# 2. Compute metrics
# ===========================================================================
metrics = {}

for label, data in [("DeepFlow", df_data), ("DeepXDE", dx_data)]:
    entry = {}
    if data is None:
        metrics[label] = entry
        continue

    entry["train_time_s"]       = _scalar(data, "train_time_s")
    entry["final_total_loss"]   = _scalar(data, "final_total_loss")
    entry["best_loss_train"]    = _scalar(data, "best_loss_train")
    entry["best_loss_test"]     = _scalar(data, "best_loss_test")

    u = _arr(data, "u")
    v = _arr(data, "v")
    entry["mean_u"] = float(np.mean(np.abs(u))) if len(u) else float("nan")
    entry["mean_v"] = float(np.mean(np.abs(v))) if len(v) else float("nan")

    for res_key in ["continuity_residual", "x_momentum_residual", "y_momentum_residual"]:
        r = _arr(data, res_key)
        entry[f"max_{res_key}"] = float(np.max(np.abs(r))) if len(r) else float("nan")

    # Mass flux at inlet (x ~ 0) and outlet (x ~ Lx)
    x_coord = _arr(data, "x")
    y_coord = _arr(data, "y")
    u_arr   = _arr(data, "u")
    if len(x_coord) and len(u_arr):
        # Find indices near x=0 and x=Lx
        eps = 0.01
        inlet_mask  = np.abs(x_coord - 0.0) < eps
        outlet_mask = np.abs(x_coord - 5.0) < eps

        if np.any(inlet_mask):
            # Mass flux ~ ∫ u dy ≈ sum(u * Δy), but since points are not
            # uniformly spaced over y, sort by y and use trapezoidal approx
            y_in = y_coord[inlet_mask]
            u_in = u_arr[inlet_mask]
            sort_idx = np.argsort(y_in)
            entry["mass_flux_inlet"] = float(_trapz(u_in[sort_idx], y_in[sort_idx]))
        else:
            entry["mass_flux_inlet"] = float("nan")

        if np.any(outlet_mask):
            y_out = y_coord[outlet_mask]
            u_out = u_arr[outlet_mask]
            sort_idx = np.argsort(y_out)
            entry["mass_flux_outlet"] = float(_trapz(u_out[sort_idx], y_out[sort_idx]))
        else:
            entry["mass_flux_outlet"] = float("nan")

        if not np.isnan(entry.get("mass_flux_inlet", float("nan"))) and \
           not np.isnan(entry.get("mass_flux_outlet", float("nan"))) and \
           abs(entry["mass_flux_inlet"]) > 1e-15:
            entry["mass_flux_rel_error"] = abs(
                entry["mass_flux_inlet"] - entry["mass_flux_outlet"]
            ) / abs(entry["mass_flux_inlet"])
        else:
            entry["mass_flux_rel_error"] = float("nan")
    else:
        entry["mass_flux_inlet"]      = float("nan")
        entry["mass_flux_outlet"]     = float("nan")
        entry["mass_flux_rel_error"]  = float("nan")

    metrics[label] = entry

# ===========================================================================
# 3. Console summary table
# ===========================================================================
print("=" * 100)
print("Benchmark Comparison: DeepFlow vs DeepXDE (2D Steady Channel Flow)")
print("=" * 100)

header = f"{'Metric':<40} {'DeepFlow':>20} {'DeepXDE':>20} {'Unit':>15}"
sep = "-" * len(header)
print(sep)
print(header)
print(sep)

rows = [
    ("Train time",        "train_time_s",        "{:.2f}"),
    ("Final total loss",  "final_total_loss",    "{:.6e}"),
    ("Best train loss",   "best_loss_train",     "{:.6e}"),
    ("Best test loss",    "best_loss_test",      "{:.6e}"),
    ("Mean |u|",          "mean_u",             "{:.6f}"),
    ("Mean |v|",          "mean_v",             "{:.6f}"),
    ("Max continuity|",   "max_continuity_residual",   "{:.6e}"),
    ("Max x-momentum|",   "max_x_momentum_residual",   "{:.6e}"),
    ("Max y-momentum|",   "max_y_momentum_residual",   "{:.6e}"),
    ("Mass flux inlet",   "mass_flux_inlet",      "{:.6f}"),
    ("Mass flux outlet",  "mass_flux_outlet",     "{:.6f}"),
    ("Mass flux rel.err", "mass_flux_rel_error",  "{:.6e}"),
]

for label, key, fmt in rows:
    df_val = metrics["DeepFlow"].get(key, float("nan"))
    dx_val = metrics["DeepXDE"].get(key, float("nan"))

    df_str = fmt.format(df_val) if not (isinstance(df_val, float) and np.isnan(df_val)) else "N/A"
    dx_str = fmt.format(dx_val) if not (isinstance(dx_val, float) and np.isnan(dx_val)) else "N/A"

    print(f"{label:<40} {df_str:>20} {dx_str:>20}")

print(sep)
print()

# ===========================================================================
# 4. Plots
# ===========================================================================
print("Generating comparison plots ...")

os.makedirs(_RESULTS_DIR, exist_ok=True)


def _field_data(data, key):
    """Return (x, y, field) arrays."""
    x = _arr(data, "x")
    y = _arr(data, "y")
    f = _arr(data, key)
    return x, y, f


def _scatter_plot(ax, x, y, field, title, cmap="viridis", vrange=None):
    """Scatter plot on given axis."""
    sc = ax.scatter(x, y, c=field, s=1, cmap=cmap, marker="s")
    if vrange is not None:
        sc.set_clim(vrange)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(sc, ax=ax, shrink=0.8)


def _shared_range(*arrays):
    """Return a shared (vmin, vmax) covering all supplied arrays, ignoring NaNs."""
    flat = [np.asarray(a).ravel() for a in arrays if a is not None and len(np.asarray(a))]
    if not flat:
        return None
    vals = np.concatenate(flat)
    if len(vals) == 0:
        return None
    return float(np.nanmin(vals)), float(np.nanmax(vals))

def _line_plot(ax, x, y, label, color):
    ax.plot(x, y, label=label, color=color)
    ax.set_xlabel("x" if "u(x)" in label or "u(y)" not in label else "y")
    ax.set_ylabel("u")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _profile_u_vs_y_at_x(data, x_target):
    """Extract u(y) profile near a given x coordinate."""
    x_arr = _arr(data, "x")
    y_arr = _arr(data, "y")
    u_arr = _arr(data, "u")
    if not len(x_arr):
        return np.array([]), np.array([])
    mask = np.abs(x_arr - x_target) < 0.02
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([]), np.array([])
    sort_i = np.argsort(y_arr[idx])
    return y_arr[idx][sort_i], u_arr[idx][sort_i]


def _profile_u_vs_x_at_y(data, y_target):
    """Extract u(x) profile near a given y coordinate."""
    x_arr = _arr(data, "x")
    y_arr = _arr(data, "y")
    u_arr = _arr(data, "u")
    if not len(x_arr):
        return np.array([]), np.array([])
    mask = np.abs(y_arr - y_target) < 0.02
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.array([]), np.array([])
    sort_i = np.argsort(x_arr[idx])
    return x_arr[idx][sort_i], u_arr[idx][sort_i]


# -- (a) u velocity field ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
u_range = _shared_range(
    _field_data(df_data, "u")[2] if df_data is not None else None,
    _field_data(dx_data, "u")[2] if dx_data is not None else None,
)
for ax, label, data in zip(axes, ["DeepFlow", "DeepXDE"], [df_data, dx_data]):
    if data is not None:
        x, y, u = _field_data(data, "u")
        _scatter_plot(ax, x, y, u, f"u – {label}", cmap="jet", vrange=u_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_u_field.png"), dpi=150)
plt.close(fig)

# -- (b) v velocity field ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
v_range = _shared_range(
    _field_data(df_data, "v")[2] if df_data is not None else None,
    _field_data(dx_data, "v")[2] if dx_data is not None else None,
)
for ax, label, data in zip(axes, ["DeepFlow", "DeepXDE"], [df_data, dx_data]):
    if data is not None:
        x, y, v = _field_data(data, "v")
        _scatter_plot(ax, x, y, v, f"v – {label}", cmap="jet", vrange=v_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_v_field.png"), dpi=150)
plt.close(fig)

# -- (c) Pressure field ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
p_range = _shared_range(
    _field_data(df_data, "p")[2] if df_data is not None else None,
    _field_data(dx_data, "p")[2] if dx_data is not None else None,
)
for ax, label, data in zip(axes, ["DeepFlow", "DeepXDE"], [df_data, dx_data]):
    if data is not None:
        x, y, p = _field_data(data, "p")
        _scatter_plot(ax, x, y, p, f"p – {label}", cmap="jet", vrange=p_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_p_field.png"), dpi=150)
plt.close(fig)

# -- (d) Velocity magnitude ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
if df_data is not None:
    x_df, y_df, u_df = _field_data(df_data, "u")
    _, _, v_df = _field_data(df_data, "v")
    mag_df = np.sqrt(u_df**2 + v_df**2)
else:
    mag_df = None
if dx_data is not None:
    x_dx, y_dx, u_dx = _field_data(dx_data, "u")
    _, _, v_dx = _field_data(dx_data, "v")
    mag_dx = np.sqrt(u_dx**2 + v_dx**2)
else:
    mag_dx = None
mag_range = _shared_range(mag_df, mag_dx)
for ax, label, data in zip(axes, ["DeepFlow", "DeepXDE"], [df_data, dx_data]):
    if data is not None:
        x, y, u = _field_data(data, "u")
        _, _, v = _field_data(data, "v")
        mag = np.sqrt(u**2 + v**2)
        _scatter_plot(ax, x, y, mag, f"|U| – {label}", cmap="jet", vrange=mag_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_velocity_magnitude.png"), dpi=150)
plt.close(fig)

# -- (e) Continuity residual -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
cont_range = _shared_range(
    np.abs(_field_data(df_data, "continuity_residual")[2]) if df_data is not None else None,
    np.abs(_field_data(dx_data, "continuity_residual")[2]) if dx_data is not None else None,
)
for ax, label, data in zip(axes, ["DeepFlow", "DeepXDE"], [df_data, dx_data]):
    if data is not None:
        x, y, r = _field_data(data, "continuity_residual")
        _scatter_plot(ax, x, y, np.abs(r), f"|Continuity| – {label}", cmap="hot", vrange=cont_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_continuity_residual.png"), dpi=150)
plt.close(fig)

# -- (f) X-momentum residual -----------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
xmom_range = _shared_range(
    np.abs(_field_data(df_data, "x_momentum_residual")[2]) if df_data is not None else None,
    np.abs(_field_data(dx_data, "x_momentum_residual")[2]) if dx_data is not None else None,
)
for ax, label, data in zip(axes, ["DeepFlow", "DeepXDE"], [df_data, dx_data]):
    if data is not None:
        x, y, r = _field_data(data, "x_momentum_residual")
        _scatter_plot(ax, x, y, np.abs(r), f"|x-Momentum| – {label}", cmap="hot", vrange=xmom_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_x_momentum_residual.png"), dpi=150)
plt.close(fig)

# -- (g) Loss curves -------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
if df_data is not None:
    tl = _arr(df_data, "total_loss")
    if len(tl):
        ax.semilogy(tl, label="DeepFlow total", color="C0")
if dx_data is not None:
    lt = _arr(dx_data, "loss_train")
    lt_steps = _arr(dx_data, "loss_steps")
    if len(lt):
        x = lt_steps if len(lt_steps) == len(lt) else np.linspace(0, EPOCHS, len(lt))
        ax.semilogy(x, lt, label="DeepXDE train", color="C1", linestyle="--")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
ax.set_title("Training Loss Curves")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_loss_curves.png"), dpi=150)
plt.close(fig)

# -- (h) Profile u(y) at x = 2.5 -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for label, data, color in [("DeepFlow", df_data, "C0"), ("DeepXDE", dx_data, "C1")]:
    if data is not None:
        y_p, u_p = _profile_u_vs_y_at_x(data, 2.5)
        if len(y_p):
            ax.plot(u_p, y_p, label=label, color=color)
ax.set_xlabel("u")
ax.set_ylabel("y")
ax.set_title("u(y) at x = 2.5")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_profile_u_y_at_x2.5.png"), dpi=150)
plt.close(fig)

# -- (i) Profile u(x) at y = 0.5 -------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for label, data, color in [("DeepFlow", df_data, "C0"), ("DeepXDE", dx_data, "C1")]:
    if data is not None:
        x_p, u_p = _profile_u_vs_x_at_y(data, 0.5)
        if len(x_p):
            ax.plot(x_p, u_p, label=label, color=color)
ax.set_xlabel("x")
ax.set_ylabel("u")
ax.set_title("u(x) at y = 0.5")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(_RESULTS_DIR, "compare_profile_u_x_at_y0.5.png"), dpi=150)
plt.close(fig)

print("Plots saved to results/")

# ===========================================================================
# 5. Markdown report
# ===========================================================================
report_lines = [
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

row_templates = {
    "train_time_s":        ("Train time", "{:.2f}", "s"),
    "final_total_loss":    ("Final total loss", "{:.6e}", "–"),
    "best_loss_train":     ("Best train loss", "{:.6e}", "–"),
    "best_loss_test":      ("Best test loss", "{:.6e}", "–"),
    "mean_u":              ("Mean |u|", "{:.6f}", "–"),
    "mean_v":              ("Mean |v|", "{:.6f}", "–"),
    "max_continuity_residual":   ("Max |continuity residual|", "{:.6e}", "–"),
    "max_x_momentum_residual":   ("Max |x-momentum residual|", "{:.6e}", "–"),
    "max_y_momentum_residual":   ("Max |y-momentum residual|", "{:.6e}", "–"),
    "mass_flux_inlet":     ("Mass flux inlet", "{:.6f}", "–"),
    "mass_flux_outlet":    ("Mass flux outlet", "{:.6f}", "–"),
    "mass_flux_rel_error": ("Mass flux rel. error", "{:.6e}", "–"),
}

def _fmt_val(val, fmt_spec):
    if isinstance(val, float) and np.isnan(val):
        return "N/A"
    return fmt_spec.format(val)

for key, (desc, fmt, unit) in row_templates.items():
    df_v = metrics["DeepFlow"].get(key, float("nan"))
    dx_v = metrics["DeepXDE"].get(key, float("nan"))
    df_s = _fmt_val(df_v, fmt)
    dx_s = _fmt_val(dx_v, fmt)
    report_lines.append(f"| {desc} | {df_s} | {dx_s} | {unit} |")

report_lines.extend([
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
])

report_path = os.path.join(_RESULTS_DIR, "benchmark_report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines) + "\n")

print(f"Report saved to {report_path}")
print("=" * 100)
