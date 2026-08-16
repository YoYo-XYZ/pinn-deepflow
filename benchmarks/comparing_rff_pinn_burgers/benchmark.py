"""Short PINN versus RFFPINN benchmark for the 1D Burgers equation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import pi, sin


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import deepflow as df  # noqa: E402


SEED = 69
NU = 0.01 / pi
WIDTH = 16
DEPTH = 4
ADAM_EPOCHS = 500
LBFGS_EPOCHS = 100
LEARNING_RATE = 0.004
R3_INTERVAL = 100
BOUNDARY_POINTS = [512, 256, 256]
INTERIOR_POINTS = [1024]
EMBED_DIM = 256
ALPHA = 5.0
EVALUATION_RESOLUTION = [161, 81]
FEM_MESH_SIZE = 0.02
FEM_TIME_STEP = 0.01

RESULTS_DIR = SCRIPT_DIR / "results"
METRICS_PATH = RESULTS_DIR / "metrics.json"
FIELDS_PATH = RESULTS_DIR / "fields.npz"
HISTORY_PATH = RESULTS_DIR / "training_history.npz"
PINN_CHECKPOINT_PATH = RESULTS_DIR / "pinn_checkpoint.pt"
RFFPINN_CHECKPOINT_PATH = RESULTS_DIR / "rffpinn_checkpoint.pt"


def _build_domain(*, sample_training: bool, for_reference: bool = False):
    area = df.geometry.rectangle([-1, 1], [0, 1])
    initial_condition = df.geometry.line_horizontal(y=0, range_x=[-1, 1])
    left_boundary = df.geometry.line_vertical(x=-1, range_y=[0, 1])
    right_boundary = df.geometry.line_vertical(x=1, range_y=[0, 1])
    domain = df.domain(
        area.area_list,
        initial_condition,
        left_boundary,
        right_boundary,
    )
    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})

    if for_reference:
        domain.area_list[0].define_time(
            (0.0, 1.0),
            sampling_scheme="uniform",
            expo_scaling=False,
        )
    if sample_training:
        domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)
    return domain


def _solve_reference() -> tuple[dict, dict]:
    domain = _build_domain(sample_training=False, for_reference=True)
    start = time.perf_counter()
    reference = domain.solve_fem(
        mesh_size=FEM_MESH_SIZE,
        time_step=FEM_TIME_STEP,
        tolerance=1e-8,
        max_iterations=50,
    )
    elapsed = time.perf_counter() - start
    area = domain.area_list[0]
    x_axis = np.linspace(area.ranges[0][0], area.ranges[0][1], EVALUATION_RESOLUTION[0])
    y_axis = np.linspace(area.ranges[1][0], area.ranges[1][1], EVALUATION_RESOLUTION[1])
    x, y = np.meshgrid(x_axis, y_axis, indexing="ij")
    values = reference.reference_solution.evaluate(x, y, fields=("u",))
    data = {
        "x": x.reshape(-1),
        "y": y.reshape(-1),
        "u_ref": np.asarray(values["u"]).reshape(-1),
    }
    metadata = reference.metadata
    summary = {
        "solve_time_s": elapsed,
        "backend": metadata["backend"],
        "mesh_elements": metadata["mesh"]["elements"],
        "time_steps": len(metadata["time_values"]) - 1,
        "converged": bool(metadata["converged"]),
        "max_solver_residual": float(max(metadata["solver_residuals"])),
    }
    return data, summary


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _predict(model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    parameter = next(model.parameters())
    inputs = {
        "x": torch.as_tensor(x, dtype=parameter.dtype, device=parameter.device),
        "y": torch.as_tensor(y, dtype=parameter.dtype, device=parameter.device),
    }
    model.eval()
    with torch.no_grad():
        return model(inputs)["u"].detach().cpu().numpy()


def _accuracy(prediction: np.ndarray, reference: np.ndarray, y: np.ndarray) -> dict:
    error = prediction - reference
    final_mask = np.isclose(y, 1.0)
    final_error = error[final_mask]
    final_reference = reference[final_mask]
    return {
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
        "final_time_relative_l2": float(
            np.linalg.norm(final_error) / np.linalg.norm(final_reference)
        ),
    }


def _save_checkpoint(model, model_class, model_kwargs, history, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_class": model_class.__name__,
            "model_kwargs": model_kwargs,
            "state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "loss_history": {
                key: values.tolist()
                for key, values in history.items()
            },
            "seed": SEED,
            "dtype": str(df.get_dtype()),
        },
        path,
    )


def load_checkpoint(path: Path):
    """Reconstruct a benchmark model on CPU without rerunning training."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    model_class = getattr(df, checkpoint["model_class"])
    model = model_class(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["state_dict"])
    model.loss_history = checkpoint["loss_history"]
    return model


def _train(model_class, reference_data: dict) -> tuple[dict, np.ndarray, dict]:
    df.manual_seed(SEED)
    domain = _build_domain(sample_training=True)
    df.manual_seed(SEED)

    model_kwargs = {
        "input_vars": ["x", "y"],
        "output_vars": ["u"],
        "width": WIDTH,
        "length": DEPTH,
    }
    if model_class is df.RFFPINN:
        model_kwargs.update(embed_dim=EMBED_DIM, alpha=ALPHA)
    model = model_class(**model_kwargs)
    trainable_parameters = sum(p.numel() for p in model.parameters())

    loss_function = df.calc_loss_simple(domain)

    def resample_r3(epoch, _model):
        if epoch % R3_INTERVAL == 0 and epoch < ADAM_EPOCHS:
            domain.sampling_R3(BOUNDARY_POINTS, INTERIOR_POINTS)

    _synchronize()
    start = time.perf_counter()
    _, best_adam = model.train_adam(
        learning_rate=LEARNING_RATE,
        epochs=ADAM_EPOCHS,
        calc_loss=loss_function,
        print_every=ADAM_EPOCHS,
        do_between_epochs=resample_r3,
    )
    _, best_model = best_adam.train_lbfgs(
        epochs=LBFGS_EPOCHS,
        calc_loss=loss_function,
        print_every=LBFGS_EPOCHS,
    )
    _synchronize()
    training_time = time.perf_counter() - start

    final_loss = float(loss_function(best_model)["total_loss"].detach().cpu())
    prediction = _predict(
        best_model,
        reference_data["x"],
        reference_data["y"],
    )
    metrics = {
        "train_time_s": training_time,
        "trainable_parameters": trainable_parameters,
        "final_training_loss": final_loss,
        **_accuracy(
            prediction,
            reference_data["u_ref"],
            reference_data["y"],
        ),
    }
    history = {
        key: np.asarray(values, dtype=np.float64)
        for key, values in best_model.loss_history.items()
    }
    checkpoint_path = (
        RFFPINN_CHECKPOINT_PATH
        if model_class is df.RFFPINN
        else PINN_CHECKPOINT_PATH
    )
    _save_checkpoint(
        best_model,
        model_class,
        model_kwargs,
        history,
        checkpoint_path,
    )
    metrics["checkpoint"] = checkpoint_path.name
    return metrics, prediction, history


def _commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    print("Solving FEM reference...")
    reference_data, fem_summary = _solve_reference()
    if not fem_summary["converged"]:
        raise RuntimeError("FEM reference did not converge")

    results = {}
    predictions = {}
    histories = {}
    for model_class in (df.PINN, df.RFFPINN):
        print(f"\nTraining {model_class.__name__}...")
        metrics, prediction, history = _train(model_class, reference_data)
        results[model_class.__name__] = metrics
        predictions[model_class.__name__] = prediction
        histories[model_class.__name__] = history

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": _commit_hash(),
        "device": str(df.device),
        "dtype": str(df.get_dtype()),
        "config": {
            "seed": SEED,
            "nu": float(NU),
            "width": WIDTH,
            "depth": DEPTH,
            "adam_epochs": ADAM_EPOCHS,
            "lbfgs_epochs": LBFGS_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "sampling_scheme": "R3",
            "r3_interval": R3_INTERVAL,
            "boundary_points": BOUNDARY_POINTS,
            "interior_points": INTERIOR_POINTS,
            "embed_dim": EMBED_DIM,
            "alpha": ALPHA,
            "evaluation_resolution": EVALUATION_RESOLUTION,
            "fem_mesh_size": FEM_MESH_SIZE,
            "fem_time_step": FEM_TIME_STEP,
        },
        "fem": fem_summary,
        "models": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        FIELDS_PATH,
        x=reference_data["x"],
        y=reference_data["y"],
        u_fem=reference_data["u_ref"],
        u_pinn=predictions["PINN"],
        u_rffpinn=predictions["RFFPINN"],
    )
    np.savez_compressed(
        HISTORY_PATH,
        pinn_total_loss=histories["PINN"]["total_loss"],
        pinn_bc_loss=histories["PINN"]["bc_loss"],
        pinn_pde_loss=histories["PINN"]["pde_loss"],
        rffpinn_total_loss=histories["RFFPINN"]["total_loss"],
        rffpinn_bc_loss=histories["RFFPINN"]["bc_loss"],
        rffpinn_pde_loss=histories["RFFPINN"]["pde_loss"],
    )
    from plot_results import main as plot_results

    plot_results()

    print("\nBenchmark results")
    for name, metrics in results.items():
        print(
            f"{name:8s} relative L2={metrics['relative_l2']:.6f}, "
            f"final-time relative L2={metrics['final_time_relative_l2']:.6f}, "
            f"time={metrics['train_time_s']:.2f}s"
        )
    print(f"Metrics: {METRICS_PATH}")
    print(f"Fields:  {FIELDS_PATH}")
    print(f"History: {HISTORY_PATH}")
    print(f"PINN checkpoint:    {PINN_CHECKPOINT_PATH}")
    print(f"RFFPINN checkpoint: {RFFPINN_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
