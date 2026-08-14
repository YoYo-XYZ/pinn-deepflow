"""Compare LHS, uniform, and R3 sampling for one Burgers RFFPINN."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import pi, sin


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import deepflow as df  # noqa: E402


SEED = 69
NU = 0.01 / pi
MODEL_KWARGS = dict(
    input_vars=["x", "y"], output_vars=["u"], width=16, length=4,
    embed_dim=256, alpha=5.0,
)
ADAM_EPOCHS = 1000
LEARNING_RATE = 0.004
R3_INTERVAL = 100
BOUNDARY_POINTS = [512, 256, 256]
INTERIOR_POINTS = [1024]
UNIFORM_INTERIOR_RESOLUTION = [[32, 32]]
EVALUATION_RESOLUTION = [161, 81]
RESULTS_DIR = SCRIPT_DIR / "results"


def build_domain():
    area = df.geometry.rectangle([-1, 1], [0, 1])
    initial = df.geometry.line_horizontal(y=0, range_x=[-1, 1])
    left = df.geometry.line_vertical(x=-1, range_y=[0, 1])
    right = df.geometry.line_vertical(x=1, range_y=[0, 1])
    domain = df.domain(area.area_list, initial, left, right)
    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=NU))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})
    return domain


def solve_reference():
    domain = build_domain()
    domain.area_list[0].define_time((0.0, 1.0), sampling_scheme="uniform", expo_scaling=False)
    result = domain.solve_fem(
        mesh_size=0.02, time_step=0.01, tolerance=1e-8, max_iterations=50,
    )
    area = domain.area_list[0]
    x_axis = np.linspace(area.ranges[0][0], area.ranges[0][1], EVALUATION_RESOLUTION[0])
    y_axis = np.linspace(area.ranges[1][0], area.ranges[1][1], EVALUATION_RESOLUTION[1])
    x, y = np.meshgrid(x_axis, y_axis, indexing="ij")
    values = result.evaluate(x, y, fields=("u",))
    data = {
        "x": x.reshape(-1),
        "y": y.reshape(-1),
        "u_ref": np.asarray(values["u"]).reshape(-1),
    }
    return data, result.metadata


def predict(model, data):
    parameter = next(model.parameters())
    inputs = {
        key: torch.as_tensor(data[key], dtype=parameter.dtype, device=parameter.device)
        for key in ("x", "y")
    }
    model.eval()
    with torch.no_grad():
        return model(inputs)["u"].detach().cpu().numpy()


def metrics(prediction, reference, time_values):
    error = prediction - reference
    final = np.isclose(time_values, 1.0)
    return {
        "relative_l2": float(np.linalg.norm(error) / np.linalg.norm(reference)),
        "final_time_relative_l2": float(
            np.linalg.norm(error[final]) / np.linalg.norm(reference[final])
        ),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def train(scheme, reference_data):
    df.manual_seed(SEED)
    domain = build_domain()
    if scheme == "uniform":
        domain.sampling_uniform(BOUNDARY_POINTS, UNIFORM_INTERIOR_RESOLUTION)
    else:
        domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)

    df.manual_seed(SEED)
    model = df.RFFPINN(**MODEL_KWARGS)
    loss_function = df.calc_loss_simple(domain)

    def resample(epoch, _model):
        if scheme == "r3" and epoch % R3_INTERVAL == 0 and epoch < ADAM_EPOCHS:
            domain.sampling_R3(BOUNDARY_POINTS, INTERIOR_POINTS)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    trained_model, _ = model.train_adam(
        learning_rate=LEARNING_RATE,
        epochs=ADAM_EPOCHS,
        calc_loss=loss_function,
        print_every=ADAM_EPOCHS,
        do_between_epochs=resample,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    prediction = predict(trained_model, reference_data)
    result_metrics = metrics(prediction, reference_data["u_ref"], reference_data["y"])
    result_metrics.update(
        train_time_s=elapsed,
        final_training_loss=float(loss_function(trained_model)["total_loss"].detach().cpu()),
    )
    history = {key: np.asarray(value) for key, value in trained_model.loss_history.items()}
    torch.save(
        {
            "model_class": "RFFPINN",
            "model_kwargs": MODEL_KWARGS,
            "state_dict": {key: value.detach().cpu() for key, value in trained_model.state_dict().items()},
            "loss_history": {key: value.tolist() for key, value in history.items()},
            "sampling_scheme": scheme,
            "seed": SEED,
        },
        RESULTS_DIR / f"rffpinn_{scheme}_checkpoint.pt",
    )
    return result_metrics, prediction, history


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print("Solving FEM reference...")
    reference, fem_metadata = solve_reference()
    if not fem_metadata["converged"]:
        raise RuntimeError("FEM reference did not converge")

    all_metrics, predictions, histories = {}, {}, {}
    for scheme in ("lhs", "uniform", "r3"):
        print(f"\nTraining RFFPINN with {scheme} sampling...")
        all_metrics[scheme], predictions[scheme], histories[scheme] = train(scheme, reference)

    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(df.device),
        "dtype": str(df.get_dtype()),
        "config": {
            "seed": SEED, "adam_epochs": ADAM_EPOCHS, "lbfgs_epochs": 0,
            "learning_rate": LEARNING_RATE, "alpha": MODEL_KWARGS["alpha"],
            "embed_dim": MODEL_KWARGS["embed_dim"], "boundary_points": BOUNDARY_POINTS,
            "interior_points": 1024, "uniform_interior_resolution": [32, 32],
            "r3_interval": R3_INTERVAL,
        },
        "models": all_metrics,
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    np.savez_compressed(
        RESULTS_DIR / "fields.npz", x=reference["x"], y=reference["y"],
        u_fem=reference["u_ref"], **{f"u_{key}": value for key, value in predictions.items()},
    )
    np.savez_compressed(
        RESULTS_DIR / "training_history.npz",
        **{f"{scheme}_{key}": value for scheme, history in histories.items() for key, value in history.items()},
    )
    from plot_results import main as plot_results
    plot_results()

    print("\nSampling comparison")
    for scheme, values in all_metrics.items():
        print(f"{scheme:7s} L2={values['relative_l2']:.6f}, final={values['final_time_relative_l2']:.6f}, time={values['train_time_s']:.2f}s")


if __name__ == "__main__":
    main()
