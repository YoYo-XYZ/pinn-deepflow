"""Plot saved RFFPINN sampling-comparison results without retraining."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


RESULTS = Path(__file__).resolve().parent / "results"
SCHEMES = ("lhs", "uniform", "r3")
LABELS = {"lhs": "Fixed LHS", "uniform": "Uniform grid", "r3": "R3"}


def grid(data, key):
    x, y = np.unique(data["x"]), np.unique(data["y"])
    z = np.empty((len(y), len(x)))
    z[np.searchsorted(y, data["y"]), np.searchsorted(x, data["x"])] = data[key]
    return x, y, z


def main():
    data = np.load(RESULTS / "fields.npz")
    history = np.load(RESULTS / "training_history.npz")
    x, y, reference = grid(data, "u_fem")
    predictions = {scheme: grid(data, f"u_{scheme}")[2] for scheme in SCHEMES}

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for axis, scheme in zip(axes[0], SCHEMES):
        image = axis.pcolormesh(x, y, predictions[scheme], shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_title(LABELS[scheme])
        axis.set(xlabel="x", ylabel="time")
    figure.colorbar(image, ax=axes[0], label="u")
    maximum = max(np.abs(predictions[s] - reference).max() for s in SCHEMES)
    for axis, scheme in zip(axes[1], SCHEMES):
        error = np.abs(predictions[scheme] - reference)
        image = axis.pcolormesh(x, y, np.maximum(error, 1e-4), shading="auto", cmap="magma", norm=LogNorm(1e-4, maximum))
        axis.set_title(f"{LABELS[scheme]} absolute error")
        axis.set(xlabel="x", ylabel="time")
    figure.colorbar(image, ax=axes[1], label="absolute error")
    figure.suptitle("RFFPINN sampling schemes: fields and FEM errors")
    figure.savefig(RESULTS / "field_error_comparison.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for scheme in SCHEMES:
        axes[0].semilogy(history[f"{scheme}_total_loss"], label=LABELS[scheme])
        numerator = np.linalg.norm(predictions[scheme] - reference, axis=1)
        axes[1].semilogy(y, numerator / np.linalg.norm(reference, axis=1), label=LABELS[scheme])
    axes[0].set(title="Adam learning curves", xlabel="epoch", ylabel="mean-squared loss")
    axes[1].set(title="FEM relative L2 error by time", xlabel="time", ylabel="relative L2 error")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Training convergence and temporal error propagation")
    figure.savefig(RESULTS / "learning_and_temporal_error.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, target in zip(axes.flat, (0.25, 0.5, 0.75, 1.0)):
        index = int(np.argmin(np.abs(y - target)))
        axis.plot(x, reference[index], color="black", linewidth=2, label="FEM")
        for scheme in SCHEMES:
            axis.plot(x, predictions[scheme][index], label=LABELS[scheme])
        axis.set(title=f"time = {y[index]:.2f}", xlabel="x", ylabel="u")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Burgers solution profiles by sampling scheme")
    figure.savefig(RESULTS / "time_slices.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
