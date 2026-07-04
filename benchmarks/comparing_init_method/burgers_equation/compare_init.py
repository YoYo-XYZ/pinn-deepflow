#!/usr/bin/env python3
"""
Compare DeepFlow Burgers-equation performance with Glorot-normal (current default)
vs the old Kaiming-uniform initialization.

The script temporarily monkey-patches _init_weights to Kaiming, runs a short
benchmark for both initializers, and prints the metrics side by side.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch import sin, pi

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", "..", "src"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import deepflow as df
from common_config import X_RANGE, Y_RANGE, NU, WIDTH, DEPTH, LR, EPOCHS, BOUNDARY_POINTS, INTERIOR_POINTS, EVAL_GRID, SEED

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_domain():
    area = df.geometry.rectangle(list(X_RANGE), list(Y_RANGE))
    line_ic = df.geometry.line_horizontal(y=Y_RANGE[0], range_x=list(X_RANGE))
    line_bc1 = df.geometry.line_vertical(x=X_RANGE[0], range_y=list(Y_RANGE))
    line_bc2 = df.geometry.line_vertical(x=X_RANGE[1], range_y=list(Y_RANGE))
    domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)

    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})

    domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)
    return domain


def train_one(init_name, init_fn, seed):
    print(f"\n--- Training with {init_name} initialization ---")
    df.manual_seed(seed)
    # Monkey-patch the initialization for Kaiming run
    original_init = df.NN._init_weights
    df.NN._init_weights = init_fn

    try:
        domain = build_domain()
        model0 = df.PINN(
            width=WIDTH, length=DEPTH,
            input_vars=["x", "y"], output_vars=["u"],
        )
        t0 = time.perf_counter()
        model1, model1_best = model0.train_adam(
            calc_loss=df.calc_loss_simple(domain),
            learning_rate=LR, epochs=EPOCHS,
        )
        train_time = time.perf_counter() - t0

        # Evaluate on a uniform grid
        prediction = domain.area_list[0].evaluate(model1_best)
        prediction.sampling_area(EVAL_GRID)
        data = prediction.data_dict

        return {
            "init": init_name,
            "time": train_time,
            "final_total": float(data["total_loss"][-1]),
            "final_bc": float(data["bc_loss"][-1]),
            "final_pde": float(data["pde_loss"][-1]),
            "max_pde_residual": float(np.max(np.abs(data["pde_residual"]))),
            "x": np.asarray(data["x"]),
            "y": np.asarray(data["y"]),
            "u": np.asarray(data["u"]),
        }
    finally:
        df.NN._init_weights = original_init


def _scatter_plot(ax, x, y, field, title, cmap="jet", vrange=None):
    """Scatter plot on given axis (same style as compare.py)."""
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


def kaiming_init(self):
    """Restore the old PyTorch default (Kaiming/He uniform) initialization."""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
            if m.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
                nn.init.uniform_(m.bias, -bound, bound)


def glorot_init(self):
    """Current default: Glorot (Xavier) normal."""
    for m in self.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = [
        train_one("Glorot-normal", glorot_init, SEED),
        train_one("Kaiming-uniform", kaiming_init, SEED),
    ]

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"{'Metric':<20} {'Glorot-normal':>18} {'Kaiming-uniform':>18} {'Delta':>18}")
    print("-" * 80)
    for key, label in [
        ("final_total", "Final total loss"),
        ("final_bc", "Final BC loss"),
        ("final_pde", "Final PDE loss"),
        ("max_pde_residual", "Max |PDE residual|"),
        ("time", "Train time (s)"),
    ]:
        g = results[0][key]
        k = results[1][key]
        delta = ((g - k) / k * 100) if k != 0 and isinstance(g, float) else 0
        print(f"{label:<20} {g:>18.6e} {k:>18.6e} {delta:>17.1f}%")

    print("=" * 80)

    # ---------------------------------------------------------------------------
    # u-field visualization (same style as compare.py)
    # ---------------------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    u_range = _shared_range(results[0]["u"], results[1]["u"])
    for ax, label, r in zip(axes, ["Glorot-normal", "Kaiming-uniform"], results):
        _scatter_plot(ax, r["x"], r["y"], r["u"], f"u – {label}", cmap="jet", vrange=u_range)
    plt.tight_layout()
    out_path = os.path.join(_SCRIPT_DIR, "results", "init_u_field.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"u-field figure saved to: {out_path}")
