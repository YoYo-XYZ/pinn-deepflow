"""Create the essential visual comparisons for the Burgers benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIELDS_PATH = RESULTS_DIR / "fields.npz"
HISTORY_PATH = RESULTS_DIR / "training_history.npz"
FIELD_FIGURE = RESULTS_DIR / "field_error_comparison.png"
SLICE_FIGURE = RESULTS_DIR / "time_slices.png"
LEARNING_FIGURE = RESULTS_DIR / "learning_curves.png"


def _grid(x, y, values):
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    result = np.empty((len(y_unique), len(x_unique)))
    result[np.searchsorted(y_unique, y), np.searchsorted(x_unique, x)] = values
    return x_unique, y_unique, result


def _relative_error_by_time(reference, prediction):
    numerator = np.linalg.norm(prediction - reference, axis=1)
    denominator = np.linalg.norm(reference, axis=1)
    return numerator / denominator


def plot_field_comparison(data):
    x, y, reference = _grid(data["x"], data["y"], data["u_fem"])
    _, _, pinn = _grid(data["x"], data["y"], data["u_pinn"])
    _, _, rffpinn = _grid(data["x"], data["y"], data["u_rffpinn"])
    pinn_error = np.abs(pinn - reference)
    rff_error = np.abs(rffpinn - reference)

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    field_images = []
    for axis, field, title in zip(
        axes[0],
        (reference, pinn, rffpinn),
        ("FEM reference", "PINN", "RFFPINN"),
    ):
        field_images.append(
            axis.pcolormesh(x, y, field, shading="auto", cmap="coolwarm", vmin=-1, vmax=1)
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("time")
    figure.colorbar(field_images[0], ax=axes[0], label="u", shrink=0.9)

    maximum_error = max(float(pinn_error.max()), float(rff_error.max()))
    error_norm = LogNorm(vmin=1e-4, vmax=maximum_error)
    error_images = []
    for axis, error, title in zip(
        axes[1, :2],
        (pinn_error, rff_error),
        ("PINN absolute error", "RFFPINN absolute error"),
    ):
        error_images.append(
            axis.pcolormesh(x, y, np.maximum(error, 1e-4), shading="auto", cmap="magma", norm=error_norm)
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("time")
    figure.colorbar(error_images[0], ax=axes[1, :2], label="absolute error", shrink=0.9)

    axes[1, 2].semilogy(y, _relative_error_by_time(reference, pinn), label="PINN")
    axes[1, 2].semilogy(y, _relative_error_by_time(reference, rffpinn), label="RFFPINN")
    axes[1, 2].set_title("Relative L2 error by time")
    axes[1, 2].set_xlabel("time")
    axes[1, 2].set_ylabel("relative L2 error")
    axes[1, 2].grid(alpha=0.25)
    axes[1, 2].legend()

    figure.suptitle("Burgers solution and FEM error comparison")
    figure.savefig(FIELD_FIGURE, dpi=160)
    plt.close(figure)


def plot_time_slices(data):
    x, y, reference = _grid(data["x"], data["y"], data["u_fem"])
    _, _, pinn = _grid(data["x"], data["y"], data["u_pinn"])
    _, _, rffpinn = _grid(data["x"], data["y"], data["u_rffpinn"])

    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True, constrained_layout=True)
    for axis, target_time in zip(axes.flat, (0.25, 0.5, 0.75, 1.0)):
        index = int(np.argmin(np.abs(y - target_time)))
        axis.plot(x, reference[index], color="black", linewidth=2, label="FEM")
        axis.plot(x, pinn[index], linewidth=1.6, label="PINN")
        axis.plot(x, rffpinn[index], linewidth=1.6, label="RFFPINN")
        axis.set_title(f"time = {y[index]:.2f}")
        axis.set_xlabel("x")
        axis.set_ylabel("u")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    figure.suptitle("Burgers solution profiles over time")
    figure.savefig(SLICE_FIGURE, dpi=160)
    plt.close(figure)


def plot_learning_curves(history):
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True, constrained_layout=True)

    pinn_epochs = np.arange(1, len(history["pinn_total_loss"]) + 1)
    rff_epochs = np.arange(1, len(history["rffpinn_total_loss"]) + 1)
    axes[0].semilogy(pinn_epochs, history["pinn_total_loss"], label="PINN")
    axes[0].semilogy(rff_epochs, history["rffpinn_total_loss"], label="RFFPINN")
    axes[0].set_title("Total loss comparison")
    axes[0].legend()

    for axis, prefix, epochs, title in (
        (axes[1], "pinn", pinn_epochs, "PINN loss components"),
        (axes[2], "rffpinn", rff_epochs, "RFFPINN loss components"),
    ):
        axis.semilogy(epochs, history[f"{prefix}_total_loss"], label="total")
        axis.semilogy(epochs, history[f"{prefix}_bc_loss"], label="boundary")
        axis.semilogy(epochs, history[f"{prefix}_pde_loss"], label="PDE")
        axis.set_title(title)
        axis.legend()

    for axis in axes:
        for epoch in range(100, 500, 100):
            axis.axvline(epoch, color="0.6", linestyle="--", linewidth=0.8)
        axis.axvline(500.5, color="0.4", linestyle="-.", linewidth=1)
        axis.text(500.5, axis.get_ylim()[1], " L-BFGS", va="top", ha="left")
        axis.set_xlabel("recorded epoch")
        axis.set_ylabel("mean-squared loss")
        axis.grid(alpha=0.25)

    figure.suptitle("Burgers learning curves (R3 during Adam only)")
    figure.savefig(LEARNING_FIGURE, dpi=160)
    plt.close(figure)


def main():
    data = np.load(FIELDS_PATH)
    history = np.load(HISTORY_PATH)
    plot_field_comparison(data)
    plot_time_slices(data)
    plot_learning_curves(history)
    print(FIELD_FIGURE)
    print(SLICE_FIGURE)
    print(LEARNING_FIGURE)


if __name__ == "__main__":
    main()
