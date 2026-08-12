"""Plot the decisive low-Re experiment comparison."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri

HERE = Path(__file__).resolve().parent
reference = np.load(HERE / "fem_mu_0.2.npz")
baseline = np.load(HERE / "fields_baseline_mu02.npz")
conditioned = np.load(HERE / "fields_conditioned_mu02.npz")

datasets = [reference, baseline, conditioned]
labels = ["FEM reference", "DeepFlow baseline", "Conditioned PINN"]

fig, axes = plt.subplots(3, 2, figsize=(10, 7), constrained_layout=True)
for row, (data, label) in enumerate(zip(datasets, labels)):
    x, y = data["x"], data["y"]
    tri = mtri.Triangulation(x, y)
    centers_x = x[tri.triangles].mean(axis=1)
    centers_y = y[tri.triangles].mean(axis=1)
    tri.set_mask((centers_x - 0.2) ** 2 + (centers_y - 0.2) ** 2 < 0.05**2)
    speed = np.hypot(data["u"], data["v"])
    speed_plot = axes[row, 0].tricontourf(tri, speed, levels=np.linspace(0, 1.35, 28), cmap="viridis", extend="both")
    pressure_plot = axes[row, 1].tricontourf(tri, data["p"], levels=np.linspace(0, 32, 33), cmap="plasma", extend="both")
    axes[row, 0].set_ylabel(f"{label}\ny")
    for col in range(2):
        axes[row, col].set_aspect("equal")
        axes[row, col].set_xlim(0, 1.1)
        axes[row, col].set_ylim(0, 0.41)
        axes[row, col].set_xlabel("x")

axes[0, 0].set_title("Velocity magnitude")
axes[0, 1].set_title("Pressure")
fig.colorbar(speed_plot, ax=axes[:, 0], shrink=0.9, label="|u|")
fig.colorbar(pressure_plot, ax=axes[:, 1], shrink=0.9, label="p")
fig.suptitle("Cylinder flow at mu=0.2: collapse is removed by conditioning", fontsize=14)
fig.savefig(HERE / "low_re_comparison.png", dpi=180)
