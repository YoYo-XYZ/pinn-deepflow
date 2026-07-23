#!/usr/bin/env python3
"""
Compare QCPINN vs classical PINN results for the 2D steady cylinder flow benchmark.

Loads ``results/pinn_results.npz`` and ``results/qcpinn_results.npz``,
prints a console summary table, generates side-by-side comparison plots,
and writes a Markdown report to ``results/benchmark_report.md``.

Usage:
    python compare.py
"""

import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from common_config import (
    CHANNEL_X,
    CHANNEL_Y,
    CYLINDER_CX,
    CYLINDER_CY,
    CYLINDER_R,
    REYNOLDS,
    BOUNDARY_POINTS,
    INTERIOR_POINTS,
    RESAMPLE_EVERY,
    LR_ADAM,
    EPOCHS_ADAM,
    EPOCHS_LBFGS,
    THRESHOLD_LBFGS,
    PINN_WIDTH,
    PINN_LENGTH,
    QC_PRE,
    QC_POST,
    QC_NQUBITS,
    QC_ITERATIONS,
    SEEDS,
    RESULTS_DIR,
    PINN_RESULTS_FILE,
    QCPINN_RESULTS_FILE,
    REPORT_FILE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path):
    """Load an NPZ results file; returns None if missing."""
    if not path.is_file():
        print(f"[WARN] {path} not found.")
        return None
    return np.load(path)


def _scalar(d, key, default=float("nan")):
    """Safely extract a scalar value from an NPZ array."""
    if d is None or key not in d.files:
        return default
    v = d[key]
    if v is None or v.size == 0:
        return default
    return float(v.flat[0])


def _arr(d, key, default=None):
    if d is None or key not in d.files:
        return np.array([]) if default is None else default
    return d[key]


def _shared_range(*arrays):
    """Return a shared (vmin, vmax) covering all supplied arrays, ignoring NaNs."""
    flat = [np.asarray(a).ravel() for a in arrays
            if a is not None and len(np.asarray(a))]
    if not flat:
        return None
    vals = np.concatenate(flat)
    if len(vals) == 0:
        return None
    return float(np.nanmin(vals)), float(np.nanmax(vals))


def _format_mean_std(mean, std):
    if std is not None and std > 0:
        return f"{mean:.6e} ± {std:.6e}"
    return f"{mean:.6e}"


def _safe_int(val, default="N/A"):
    """Convert to int, or return default string if NaN/missing."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return default
    try:
        return str(int(val))
    except (ValueError, TypeError):
        return default


def _pct_delta(a, b):
    if a == 0:
        return float("nan")
    return (b - a) / a * 100.0


def _scatter(ax, x, y, field, title, cmap="jet", vrange=None):
    sc = ax.scatter(x, y, c=field, s=2, cmap=cmap, marker="s")
    if vrange is not None:
        sc.set_clim(vrange)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(CHANNEL_X)
    ax.set_ylim(CHANNEL_Y)
    plt.colorbar(sc, ax=ax, shrink=0.8)


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------

pinn_data = _load(PINN_RESULTS_FILE)
qc_data = _load(QCPINN_RESULTS_FILE)

if pinn_data is None and qc_data is None:
    print("Neither PINN nor QCPINN results found. Run the benchmarks first.")
    sys.exit(1)

models = [
    ("PINN", pinn_data, "C0"),
    ("QCPINN", qc_data, "C1"),
]
loaded = [d for _, d, _ in models if d is not None]

# ---------------------------------------------------------------------------
# 2. Console summary table
# ---------------------------------------------------------------------------

print("=" * 90)
print("QCPINN vs PINN Benchmark — 2D Steady Cylinder Flow (Re=50)")
print("=" * 90)
header = f"{'Metric':<30} {'PINN':>24} {'QCPINN':>24} {'Delta':>10}"
print(header)
print("-" * len(header))

# Parameter count is a single value, not a mean±std
pinn_params = _scalar(pinn_data, "n_params")
qc_params = _scalar(qc_data, "n_params")
pinn_params_str = str(int(pinn_params)) if not np.isnan(pinn_params) else "N/A"
qc_params_str = str(int(qc_params)) if not np.isnan(qc_params) else "N/A"
print(f"{'Parameters':<30} {pinn_params_str:>24} {qc_params_str:>24}")

# Scalar metrics with mean±std
rows = [
    ("Final total loss",  "final_total_loss",  "final_total_loss"),
    ("Final BC loss",     "final_bc_loss",     "final_bc_loss"),
    ("Final PDE loss",    "final_pde_loss",    "final_pde_loss"),
    ("Max |continuity|",  "max_continuity",    "max_continuity"),
    ("Max |x-momentum|",  "max_x_momentum",    "max_x_momentum"),
    ("Max |y-momentum|",  "max_y_momentum",    "max_y_momentum"),
    ("Mean |continuity|", "mean_abs_continuity", "mean_abs_continuity"),
    ("Mean |x-momentum|", "mean_abs_x_momentum", "mean_abs_x_momentum"),
    ("Mean |y-momentum|", "mean_abs_y_momentum", "mean_abs_y_momentum"),
    ("Adam time (s)",     "adam_time_s",       "adam_time_s"),
    ("L-BFGS time (s)",   "lbfgs_time_s",      "lbfgs_time_s"),
    ("Total time (s)",    "total_time_s",      "total_time_s"),
]
for desc, k_p, k_q in rows:
    pm, ps = (_scalar(pinn_data, f"{k_p}_mean"), _scalar(pinn_data, f"{k_p}_std"))
    qm, qs = (_scalar(qc_data, f"{k_q}_mean"),   _scalar(qc_data, f"{k_q}_std"))
    p_str = _format_mean_std(pm, ps)
    q_str = _format_mean_std(qm, qs)
    delta = _pct_delta(pm, qm)
    delta_str = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
    print(f"{desc:<30} {p_str:>24} {q_str:>24} {delta_str:>10}")

print("-" * len(header))
print("Interpretation: negative Delta for losses/residuals = QCPINN lower/better.")
print("                positive Delta for time = QCPINN slower.")
print("=" * 90)

# ---------------------------------------------------------------------------
# 3. Plots
# ---------------------------------------------------------------------------

print("\nGenerating comparison plots ...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# (a) Loss curves
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, (key, title) in zip(axes, [
    ("total_loss_history", "Total loss"),
    ("bc_loss_history", "BC loss"),
    ("pde_loss_history", "PDE loss"),
]):
    for label, data, color in models:
        hist = _arr(data, key)
        if len(hist):
            ax.semilogy(hist, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.5)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_loss_curves.png", dpi=150)
plt.close(fig)
print("  -> compare_loss_curves.png")

# (b) u velocity field
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
u_range = _shared_range(_arr(pinn_data, "u"), _arr(qc_data, "u"))
for ax, (label, data, color) in zip(axes, models):
    if data is not None:
        _scatter(ax, _arr(data, "x"), _arr(data, "y"), _arr(data, "u"),
                 f"u — {label}", cmap="jet", vrange=u_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_u_field.png", dpi=150)
plt.close(fig)
print("  -> compare_u_field.png")

# (c) v velocity field
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
v_range = _shared_range(_arr(pinn_data, "v"), _arr(qc_data, "v"))
for ax, (label, data, color) in zip(axes, models):
    if data is not None:
        _scatter(ax, _arr(data, "x"), _arr(data, "y"), _arr(data, "v"),
                 f"v — {label}", cmap="jet", vrange=v_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_v_field.png", dpi=150)
plt.close(fig)
print("  -> compare_v_field.png")

# (d) Pressure field
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
p_range = _shared_range(_arr(pinn_data, "p"), _arr(qc_data, "p"))
for ax, (label, data, color) in zip(axes, models):
    if data is not None:
        _scatter(ax, _arr(data, "x"), _arr(data, "y"), _arr(data, "p"),
                 f"p — {label}", cmap="jet", vrange=p_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_p_field.png", dpi=150)
plt.close(fig)
print("  -> compare_p_field.png")

# (e) Continuity residual
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
cont_range = _shared_range(
    np.abs(_arr(pinn_data, "continuity_residual")),
    np.abs(_arr(qc_data, "continuity_residual")),
)
for ax, (label, data, color) in zip(axes, models):
    if data is not None:
        _scatter(ax, _arr(data, "x"), _arr(data, "y"),
                 np.abs(_arr(data, "continuity_residual")),
                 f"|continuity| — {label}", cmap="hot", vrange=cont_range)
    else:
        ax.text(0.5, 0.5, "N/A", ha="center", va="center", transform=ax.transAxes)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_continuity_residual.png", dpi=150)
plt.close(fig)
print("  -> compare_continuity_residual.png")

# (f) Outlet velocity profile
fig, ax = plt.subplots(figsize=(7, 5))
for label, data, color in models:
    if data is not None:
        oy = _arr(data, "outlet_y")
        ou = _arr(data, "outlet_u")
        if len(oy):
            sort_i = np.argsort(oy)
            ax.plot(oy[sort_i], ou[sort_i], label=label, color=color)
ax.set_xlabel("y")
ax.set_ylabel("u")
ax.set_title("Outlet velocity profile (u vs y at x = 1.1)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(RESULTS_DIR / "compare_outlet_velocity.png", dpi=150)
plt.close(fig)
print("  -> compare_outlet_velocity.png")

# ---------------------------------------------------------------------------
# 4. Markdown report
# ---------------------------------------------------------------------------

print("\nWriting Markdown report ...")

# Resolve which run indices were used as "median"
pinn_med_idx = int(_scalar(pinn_data, "median_run_idx", 0))
qc_med_idx = int(_scalar(qc_data, "median_run_idx", 0))

lines = [
    "# Benchmark Report: QCPINN vs PINN",
    "",
    "## 2D Steady Cylinder Flow (Re=50)",
    "",
    f"- **Geometry**: channel [{CHANNEL_X[0]}, {CHANNEL_X[1]}] × [{CHANNEL_Y[0]}, {CHANNEL_Y[1]}]"
    f" with a circular cylinder centered at ({CYLINDER_CX}, {CYLINDER_CY}),"
    f" radius {CYLINDER_R}.",
    f"- **PDE**: 2D steady incompressible Navier-Stokes; Re = {REYNOLDS}",
    "- **Boundary conditions**:",
    "  - Inlet (left, x=0): parabolic u(y) = 4·U·y·(H−y)/H², v=0 (U=1, H="
    f"{CHANNEL_Y[1]})",
    "  - Bottom / top walls (y=0, y="
    f"{CHANNEL_Y[1]}): no-slip (u=v=0)",
    "  - Outlet (right, x="
    f"{CHANNEL_X[1]}): pressure release (p=0)",
    "  - Cylinder surface (upper + lower): no-slip (u=v=0)",
    f"- **Sampling**: LHS initial — {sum(BOUNDARY_POINTS)} boundary points,"
    f" {sum(INTERIOR_POINTS)} interior points",
    f"- **Resampling**: \"randomr\" — full LHS resample every {RESAMPLE_EVERY} L-BFGS epochs",
    f"- **Training**: Adam(lr={LR_ADAM}, {EPOCHS_ADAM} epochs, threshold={THRESHOLD_LBFGS})"
    f" → L-BFGS({EPOCHS_LBFGS} epochs, threshold={THRESHOLD_LBFGS})",
    "- **Loss**: `df.calc_loss_simple` (unweighted BC + PDE sum)",
    f"- **Seeds**: {SEEDS}",
    f"- **Runs per model**: PINN = {_safe_int(_scalar(pinn_data, 'num_runs', 0), '?')},"
    f" QCPINN = {_safe_int(_scalar(qc_data, 'num_runs', 0), '?')}"
    f"  (median-loss run used for representative field plots)",
    "",
    "## Network Architectures",
    "",
    "| Model | Architecture | Parameters |",
    "|-------|--------------|------------|",
    f"| **PINN**   | `PINN(width={PINN_WIDTH}, length={PINN_LENGTH})` — {PINN_LENGTH}"
    f"×{PINN_WIDTH}-neuron hidden layers, Tanh | {_safe_int(pinn_params)} |",
    f"| **QCPINN** | `QCPINN(pre={QC_PRE}, post={QC_POST}, nqubits={QC_NQUBITS},"
    f" q_layer_iterations={QC_ITERATIONS})` — classical pre-layers →"
    f" PennyLane quantum circuit (AngleEmbedding + cascade ansatz + PauliZ)"
    f" → classical post-layers | {_safe_int(qc_params)} |",
    "",
    f"_Parameter matching: PINN has {_safe_int(pinn_params)} trainable parameters,"
    f" QCPINN has {_safe_int(qc_params)}._",
    "",
    "## Summary Table",
    "",
    "| Metric | PINN | QCPINN | Δ |",
    "|--------|------|--------|---|",
    f"| Parameters | {_safe_int(pinn_params)} | {_safe_int(qc_params)} | — |",
]

for desc, k_p, k_q in rows:
    pm, ps = (_scalar(pinn_data, f"{k_p}_mean"), _scalar(pinn_data, f"{k_p}_std"))
    qm, qs = (_scalar(qc_data, f"{k_q}_mean"),   _scalar(qc_data, f"{k_q}_std"))
    p_str = _format_mean_std(pm, ps)
    q_str = _format_mean_std(qm, qs)
    delta = _pct_delta(pm, qm)
    delta_str = f"{delta:+.1f}%" if not np.isnan(delta) else "N/A"
    lines.append(f"| {desc} | {p_str} | {q_str} | {delta_str} |")

lines += [
    "",
    "## Generated Figures",
    "",
    "All figures saved to `results/` (median-loss run per model):",
    "",
    "- `compare_loss_curves.png` — total / BC / PDE loss curves (semilogy)",
    "- `compare_u_field.png` — u velocity field, side-by-side",
    "- `compare_v_field.png` — v velocity field, side-by-side",
    "- `compare_p_field.png` — pressure field, side-by-side",
    "- `compare_continuity_residual.png` — |continuity residual| field, side-by-side",
    "- `compare_outlet_velocity.png` — outlet u(y) profile at x=1.1, overlaid",
    "",
    "## Reproducibility",
    "",
    f"- Median run index used for representative fields: PINN = {pinn_med_idx},"
    f" QCPINN = {qc_med_idx}",
    f"- Per-run final total losses: PINN = {_arr(pinn_data, 'final_total_loss_runs')},"
    f" QCPINN = {_arr(qc_data, 'final_total_loss_runs') if qc_data is not None else np.array([])}",
    "",
    "---",
    "*Report generated by `compare.py`*",
]

REPORT_FILE.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"  -> {REPORT_FILE}")
print("\nDone.")
