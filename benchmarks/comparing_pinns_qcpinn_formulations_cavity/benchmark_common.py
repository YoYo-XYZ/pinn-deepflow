"""Shared training and evaluation workflow for the four cavity setups."""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parents[2] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import deepflow as df  # noqa: E402

from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    CENTERLINE_POINTS,
    CAVITY_X,
    CAVITY_Y,
    DEFAULT_NUM_RUNS,
    EPOCHS_ADAM,
    EPOCHS_LBFGS,
    EVAL_GRID,
    INTERIOR_POINTS,
    L_CHAR,
    LR_ADAM,
    MU,
    RESULTS_DIR,
    RHO,
    SEEDS,
    U_INF,
)

MODEL_FIELDS = ("x", "y", "u", "v", "p")
RESIDUAL_FIELDS = ("continuity", "x_momentum", "y_momentum")
HISTORY_FIELDS = ("total_loss", "bc_loss", "pde_loss")
SCALAR_METRICS = (
    "final_total_loss",
    "final_bc_loss",
    "final_pde_loss",
    "pde_loss_per_equation",
    "max_continuity",
    "max_x_momentum",
    "max_y_momentum",
    "mean_abs_continuity",
    "mean_abs_x_momentum",
    "mean_abs_y_momentum",
    "adam_time_s",
    "lbfgs_time_s",
    "total_time_s",
)
REPRESENTATIVE_FIELDS = (
    *MODEL_FIELDS,
    "continuity_residual",
    "x_momentum_residual",
    "y_momentum_residual",
    "total_loss_history",
    "bc_loss_history",
    "pde_loss_history",
    "vertical_y",
    "vertical_u",
    "vertical_v",
    "horizontal_x",
    "horizontal_u",
    "horizontal_v",
)


def _make_pde(formulation: str):
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
):
    """Build and sample the shared unit-square cavity domain."""
    boundary_points = BOUNDARY_POINTS if boundary_points is None else boundary_points
    interior_points = INTERIOR_POINTS if interior_points is None else interior_points

    rectangle = df.geometry.rectangle(list(CAVITY_X), list(CAVITY_Y))
    pressure_point = df.geometry.point(CAVITY_X[0], CAVITY_Y[0])
    domain = df.domain(rectangle, pressure_point)
    domain.area_list[0].define_pde(_make_pde(formulation))

    if formulation == "uvp":
        for boundary in (0, 1, 2):
            domain.bound_list[boundary].define_bc({"u": 0, "v": 0})
        domain.bound_list[3].define_bc({"u": U_INF, "v": 0})
    elif formulation == "psip":
        for boundary in (0, 1, 2):
            domain.bound_list[boundary].define_bc({"psi_x": 0, "psi_y": 0})
        domain.bound_list[3].define_bc({"psi_x": 0, "psi_y": U_INF})
    else:
        raise ValueError(f"Unknown formulation: {formulation}")

    domain.bound_list[4].define_bc({"p": 0})
    domain.sampling_uniform(list(boundary_points), interior_points)
    return domain


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _evaluate_line(model, geometry, formulation: str) -> Dict[str, np.ndarray]:
    """Evaluate a centerline through DeepFlow's public evaluator API."""
    geometry.define_pde(_make_pde(formulation))
    geometry.sampling_line(CENTERLINE_POINTS)
    return geometry.evaluate(model).data_dict


def _evaluate_centerlines(model, formulation: str):
    vertical = df.geometry.line_vertical(0.5, list(CAVITY_Y))
    vertical_data = _evaluate_line(model, vertical, formulation)

    horizontal = df.geometry.line_horizontal(0.5, list(CAVITY_X))
    horizontal_data = _evaluate_line(model, horizontal, formulation)
    return vertical_data, horizontal_data


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
    """Train and evaluate one model/formulation run."""
    print(f"\n--- {label} run (seed {seed}) ---")
    df.manual_seed(seed)

    domain = build_domain(
        formulation,
        boundary_points=boundary_points,
        interior_points=interior_points,
    )
    calc_loss = df.calc_loss_simple(domain)
    model = model_factory()
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
    vertical_data, horizontal_data = _evaluate_centerlines(best_model, formulation)
    residual_count = len(domain.area_list[0].PDE.residual_fields)

    metrics = {
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
        f"{name}_history": np.asarray(
            best_model.loss_history[name], dtype=np.float64
        )
        for name in HISTORY_FIELDS
    })
    metrics.update({
        "vertical_y": np.asarray(vertical_data["y"]),
        "vertical_u": np.asarray(vertical_data["u"]),
        "vertical_v": np.asarray(vertical_data["v"]),
        "horizontal_x": np.asarray(horizontal_data["x"]),
        "horizontal_u": np.asarray(horizontal_data["u"]),
        "horizontal_v": np.asarray(horizontal_data["v"]),
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
    """Aggregate scalar metrics and retain the median-loss run's fields."""
    final_losses = np.asarray([run["final_total_loss"] for run in per_run])
    median_idx = int(np.argsort(final_losses)[len(final_losses) // 2])
    results = {
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
        "final_total_loss_runs": final_losses,
        "total_time_s_runs": np.asarray([run["total_time_s"] for run in per_run]),
    }
    for key in SCALAR_METRICS:
        mean, std = _mean_std([run[key] for run in per_run], num_runs)
        results[f"{key}_mean"] = mean
        results[f"{key}_std"] = std

    median_run = per_run[median_idx]
    results.update({key: median_run[key] for key in REPRESENTATIVE_FIELDS})
    if "psi" in median_run:
        results["psi"] = median_run["psi"]
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
    """Parse options, run requested seeds, and save aggregate results."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--num_runs", type=int, default=DEFAULT_NUM_RUNS,
        help=f"Number of independent runs (default: {DEFAULT_NUM_RUNS}).",
    )
    parser.add_argument(
        "--epochs_adam", type=int, default=EPOCHS_ADAM,
        help=f"Adam epochs (default: {EPOCHS_ADAM}).",
    )
    parser.add_argument(
        "--epochs_lbfgs", type=int, default=EPOCHS_LBFGS,
        help=f"L-BFGS epochs (default: {EPOCHS_LBFGS}).",
    )
    args = parser.parse_args()
    if args.num_runs < 1:
        parser.error("--num_runs must be at least 1")
    if args.epochs_adam < 0 or args.epochs_lbfgs < 0:
        parser.error("training epochs must be non-negative")

    print("=" * 88)
    print(f"{label} -- 2D Lid-Driven Cavity Flow (Re=10)")
    print("=" * 88)
    print(f"Device:           {df.device}")
    print(f"Runs:             {args.num_runs}")
    print(f"Adam epochs:      {args.epochs_adam}")
    print(f"L-BFGS epochs:    {args.epochs_lbfgs}")
    print(f"Base seed:        {SEEDS[0]}")
    print(f"Network:          {network_description}")
    print(f"Formulation:      {formulation}")

    per_run = []
    for index in range(args.num_runs):
        seed = (
            SEEDS[index]
            if index < len(SEEDS)
            else SEEDS[-1] + index - len(SEEDS) + 1
        )
        per_run.append(
            train_one(
                seed,
                model_factory,
                label,
                model_name,
                formulation,
                network_description,
                args.epochs_adam,
                args.epochs_lbfgs,
            )
        )

    results = aggregate(
        per_run,
        label,
        model_name,
        formulation,
        network_description,
        args.num_runs,
        args.epochs_adam,
        args.epochs_lbfgs,
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(results_file, **results)
    print(f"\nResults saved to: {results_file}")
    print("\n" + "-" * 62)
    print(f"{label} ({results['n_params']} params, {args.num_runs} runs)")
    print(
        f"  final total loss : {results['final_total_loss_mean']:.4e} +/- "
        f"{results['final_total_loss_std']:.4e}"
    )
    print(
        f"  final PDE loss   : {results['final_pde_loss_mean']:.4e} +/- "
        f"{results['final_pde_loss_std']:.4e}"
    )
    print(
        f"  max |continuity| : {results['max_continuity_mean']:.4e} +/- "
        f"{results['max_continuity_std']:.4e}"
    )
    print("-" * 62)
