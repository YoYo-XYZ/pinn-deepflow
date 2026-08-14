"""Shared training and evaluation workflow for the cylinder benchmark."""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import deepflow as df  # noqa: E402

from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    CHANNEL_X,
    CHANNEL_Y,
    CYLINDER_CX,
    CYLINDER_CY,
    CYLINDER_R,
    DEFAULT_NUM_RUNS,
    EPOCHS_ADAM,
    EPOCHS_LBFGS,
    EVAL_GRID,
    INTERIOR_POINTS,
    L_CHAR,
    LR_ADAM,
    MU,
    PROFILE_POINTS,
    RESULTS_DIR,
    RHO,
    SEEDS,
    U_INF,
)

MODEL_FIELDS = ("x", "y", "u", "v", "p")
RESIDUAL_FIELDS = ("continuity", "x_momentum", "y_momentum")
PDE_RESIDUAL_COUNTS = {"uvp": 3, "psip": 2}
HISTORY_FIELDS = ("total_loss", "bc_loss", "pde_loss")
SCALAR_METRICS = (
    "final_total_loss", "final_bc_loss", "final_pde_loss",
    "pde_loss_per_equation", "max_continuity", "max_x_momentum",
    "max_y_momentum", "mean_abs_continuity", "mean_abs_x_momentum",
    "mean_abs_y_momentum", "adam_time_s", "lbfgs_time_s", "total_time_s",
)
REPRESENTATIVE_FIELDS = (
    *MODEL_FIELDS,
    "continuity_residual", "x_momentum_residual", "y_momentum_residual",
    "total_loss_history", "bc_loss_history", "pde_loss_history",
    "outlet_y", "outlet_u", "outlet_v", "wake_x", "wake_u", "wake_v",
)


def make_pde(formulation: str):
    if formulation == "uvp":
        return df.pde.NavierStokes(U=U_INF, L=L_CHAR, mu=MU, rho=RHO)
    if formulation == "psip":
        return df.pde.StreamFunctionNavierStokes(
            U=U_INF, L=L_CHAR, mu=MU, rho=RHO
        )
    raise ValueError(f"Unknown formulation: {formulation}")


def build_domain(
    formulation: str,
    boundary_points: Optional[List[int]] = None,
    interior_points: Optional[List[object]] = None,
    sample: bool = True,
):
    """Build the cylinder geometry, physics, boundary conditions, and samples."""
    boundary_points = BOUNDARY_POINTS if boundary_points is None else boundary_points
    interior_points = INTERIOR_POINTS if interior_points is None else interior_points

    circle = df.geometry.circle(CYLINDER_CX, CYLINDER_CY, CYLINDER_R)
    rectangle = df.geometry.rectangle(list(CHANNEL_X), list(CHANNEL_Y))
    domain = df.domain(rectangle - circle, circle.bound_list)
    domain.area_list[0].define_pde(make_pde(formulation))

    height = CHANNEL_Y[1]
    inlet_u = [
        "y",
        lambda y: 4 * U_INF * y * (height - y) / height**2,
    ]
    if formulation == "uvp":
        domain.bound_list[0].define_bc({"u": inlet_u, "v": 0})
        for boundary in (1, 3, 4, 5):
            domain.bound_list[boundary].define_bc({"u": 0, "v": 0})
    elif formulation == "psip":
        domain.bound_list[0].define_bc({"psi_x": 0, "psi_y": inlet_u})
        for boundary in (1, 3, 4, 5):
            domain.bound_list[boundary].define_bc({"psi_x": 0, "psi_y": 0})
    else:
        raise ValueError(f"Unknown formulation: {formulation}")
    domain.bound_list[2].define_bc({"p": 0})

    if sample:
        domain.sampling_uniform(list(boundary_points), interior_points)
    return domain


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _evaluate_line(model, geometry, formulation: str) -> Dict[str, np.ndarray]:
    geometry.define_pde(make_pde(formulation))
    geometry.sampling_line(PROFILE_POINTS)
    return geometry.evaluate(model).data_dict


def evaluate_profiles(model, formulation: str):
    outlet = df.geometry.line_vertical(CHANNEL_X[1], list(CHANNEL_Y))
    wake = df.geometry.line_horizontal(
        CYLINDER_CY, [CYLINDER_CX + CYLINDER_R, CHANNEL_X[1]]
    )
    return (
        _evaluate_line(model, outlet, formulation),
        _evaluate_line(model, wake, formulation),
    )


def train_one(
    seed: int,
    model_factory: Callable[[], object],
    label: str,
    model_name: str,
    formulation: str,
    network_description: str,
    epochs_adam: int,
    epochs_lbfgs: int,
    boundary_points: Optional[List[int]] = None,
    interior_points: Optional[List[object]] = None,
) -> Dict[str, object]:
    print(f"\n--- {label} run (seed {seed}) ---")
    df.manual_seed(seed)
    domain = build_domain(
        formulation,
        boundary_points=boundary_points,
        interior_points=interior_points,
    )
    calc_loss = df.calc_loss_simple(domain)
    model = model_factory().to(df.device)
    n_params = count_params(model)
    print(f"  Model: {network_description} -- {n_params} trainable params")

    start = time.perf_counter()
    _, adam_best = model.train_adam(
        calc_loss=calc_loss,
        learning_rate=LR_ADAM,
        epochs=epochs_adam,
        print_every=max(1, epochs_adam // 10),
    )
    adam_time = time.perf_counter() - start

    start = time.perf_counter()
    _, best_model = adam_best.train_lbfgs(
        calc_loss=calc_loss,
        epochs=epochs_lbfgs,
        print_every=max(1, epochs_lbfgs // 10),
    )
    lbfgs_time = time.perf_counter() - start

    final_losses = calc_loss(best_model)
    area_eval = domain.area_list[0].evaluate(best_model)
    area_eval.sampling_area(EVAL_GRID)
    data = area_eval.data_dict
    outlet_data, wake_data = evaluate_profiles(best_model, formulation)
    residual_count = PDE_RESIDUAL_COUNTS[formulation]

    metrics: Dict[str, object] = {
        "label": label,
        "model": model_name,
        "formulation": formulation,
        "network_description": network_description,
        "seed": seed,
        "n_params": n_params,
        "pde_residual_count": residual_count,
        "adam_time_s": float(adam_time),
        "lbfgs_time_s": float(lbfgs_time),
        "total_time_s": float(adam_time + lbfgs_time),
        "final_total_loss": float(final_losses["total_loss"].detach().cpu().item()),
        "final_bc_loss": float(final_losses["bc_loss"].detach().cpu().item()),
        "final_pde_loss": float(final_losses["pde_loss"].detach().cpu().item()),
        "pde_loss_per_equation": float(
            final_losses["pde_loss"].detach().cpu().item() / residual_count
        ),
    }
    for name in RESIDUAL_FIELDS:
        residual = np.asarray(data[f"{name}_residual"])
        metrics[f"max_{name}"] = float(np.max(np.abs(residual)))
        metrics[f"mean_abs_{name}"] = float(np.mean(np.abs(residual)))

    metrics.update({name: np.asarray(data[name]) for name in MODEL_FIELDS})
    if "psi" in data:
        metrics["psi"] = np.asarray(data["psi"])
    metrics.update({
        f"{name}_residual": np.asarray(data[f"{name}_residual"])
        for name in RESIDUAL_FIELDS
    })
    metrics.update({
        f"{name}_history": np.asarray(best_model.loss_history[name], dtype=np.float64)
        for name in HISTORY_FIELDS
    })
    metrics.update({
        "outlet_y": np.asarray(outlet_data["y"]),
        "outlet_u": np.asarray(outlet_data["u"]),
        "outlet_v": np.asarray(outlet_data["v"]),
        "wake_x": np.asarray(wake_data["x"]),
        "wake_u": np.asarray(wake_data["u"]),
        "wake_v": np.asarray(wake_data["v"]),
    })
    return metrics


def _mean_std(values, num_runs: int):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if num_runs > 1 else 0.0


def aggregate(
    per_run: List[Dict[str, object]],
    label: str,
    model_name: str,
    formulation: str,
    network_description: str,
    num_runs: int,
    epochs_adam: int,
    epochs_lbfgs: int,
) -> Dict[str, object]:
    losses = np.asarray([run["final_total_loss"] for run in per_run])
    median_idx = int(np.argsort(losses)[len(losses) // 2])
    results: Dict[str, object] = {
        "label": np.asarray(label),
        "model": np.asarray(model_name),
        "formulation": np.asarray(formulation),
        "network_description": np.asarray(network_description),
        "n_params": per_run[0]["n_params"],
        "pde_residual_count": per_run[0]["pde_residual_count"],
        "num_runs": num_runs,
        "epochs_adam": epochs_adam,
        "epochs_lbfgs": epochs_lbfgs,
        "median_run_idx": median_idx,
        "seeds": np.asarray([run["seed"] for run in per_run], dtype=np.int64),
        "final_total_loss_runs": losses,
        "total_time_s_runs": np.asarray([run["total_time_s"] for run in per_run]),
    }
    for key in SCALAR_METRICS:
        mean, std = _mean_std([run[key] for run in per_run], num_runs)
        results[f"{key}_mean"] = mean
        results[f"{key}_std"] = std
    results.update({key: per_run[median_idx][key] for key in REPRESENTATIVE_FIELDS})
    if "psi" in per_run[median_idx]:
        results["psi"] = per_run[median_idx]["psi"]
    return results


def run_benchmark(
    label: str,
    model_name: str,
    formulation: str,
    description: str,
    network_description: str,
    model_factory: Callable[[], object],
    results_file: Path,
):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--num_runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--epochs_adam", type=int, default=EPOCHS_ADAM)
    parser.add_argument("--epochs_lbfgs", type=int, default=EPOCHS_LBFGS)
    args = parser.parse_args()
    if args.num_runs < 1 or args.epochs_adam < 0 or args.epochs_lbfgs < 0:
        parser.error("num_runs must be positive and epochs must be non-negative")

    print("=" * 88)
    print(f"{label} -- 2D Steady Cylinder Flow (Re={int(U_INF * L_CHAR * RHO / MU)})")
    print("=" * 88)
    print(f"Device:           {df.device}")
    print(f"Runs:             {args.num_runs}")
    print(f"Adam epochs:      {args.epochs_adam}")
    print(f"L-BFGS epochs:    {args.epochs_lbfgs}")
    print(f"Network:          {network_description}")

    per_run = []
    for index in range(args.num_runs):
        seed = SEEDS[index] if index < len(SEEDS) else SEEDS[-1] + index - len(SEEDS) + 1
        per_run.append(train_one(
            seed, model_factory, label, model_name, formulation,
            network_description, args.epochs_adam, args.epochs_lbfgs,
        ))
    results = aggregate(
        per_run, label, model_name, formulation, network_description,
        args.num_runs, args.epochs_adam, args.epochs_lbfgs,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(results_file, **results)
    print(f"\nResults saved to: {results_file}")
    print(f"  final total loss: {results['final_total_loss_mean']:.4e}")
    print(f"  final PDE loss:   {results['final_pde_loss_mean']:.4e}")
