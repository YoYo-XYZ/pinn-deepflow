"""Shared training workflow for the PINN cylinder-flow benchmarks."""

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parents[2] / "src"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import deepflow as df  # noqa: E402

from common_config import (  # noqa: E402
    BOUNDARY_POINTS,
    CHANNEL_X,
    CHANNEL_Y,
    CYLINDER_CX,
    CYLINDER_CY,
    CYLINDER_R,
    EPOCHS_ADAM,
    EPOCHS_LBFGS,
    EVAL_GRID,
    INTERIOR_POINTS,
    L_CHAR,
    LR_ADAM,
    MU,
    OUTLET_LINE_POINTS,
    RESAMPLE_EVERY,
    RESULTS_DIR,
    RHO,
    SEEDS,
    THRESHOLD_ADAM,
    THRESHOLD_LBFGS,
    U_INF,
)

MODEL_FIELDS = ("x", "y", "u", "v", "p")
RESIDUAL_FIELDS = ("continuity", "x_momentum", "y_momentum")
HISTORY_FIELDS = ("total_loss", "bc_loss", "pde_loss")
SCALAR_METRICS = (
    "final_total_loss",
    "final_bc_loss",
    "final_pde_loss",
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
    "outlet_y",
    "outlet_u",
)


def build_domain(
    boundary_points: List[int] = BOUNDARY_POINTS,
    interior_points: List[int] = INTERIOR_POINTS,
):
    """Build and sample the shared cylinder-flow domain."""
    circle = df.geometry.circle(CYLINDER_CX, CYLINDER_CY, CYLINDER_R)
    rectangle = df.geometry.rectangle(list(CHANNEL_X), list(CHANNEL_Y))
    domain = df.domain(rectangle - circle, circle.bound_list)

    domain.area_list[0].define_pde(
        df.pde.NavierStokes(U=U_INF, L=L_CHAR, mu=MU, rho=RHO)
    )

    channel_height = CHANNEL_Y[1]
    domain.bound_list[0].define_bc({
        "u": ["y", lambda y: 4 * U_INF * y * (channel_height - y) / channel_height ** 2],
        "v": 0,
    })
    for boundary in (1, 3, 4, 5):
        domain.bound_list[boundary].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.sampling_lhs(boundary_points, interior_points)
    return domain


def resample_callback(domain):
    """Return the L-BFGS callback used for periodic LHS resampling."""
    def resample(epoch, _model):
        if epoch > 0 and epoch % RESAMPLE_EVERY == 0:
            domain.sampling_lhs(BOUNDARY_POINTS, INTERIOR_POINTS)

    return resample


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def train_one(
    seed: int,
    model_factory: Callable[[], object],
    label: str,
    network_description: str,
    epochs_adam: int,
    epochs_lbfgs: int,
) -> Dict[str, object]:
    """Train and evaluate one model run."""
    print(f"\n--- {label} run (seed {seed}) ---")
    df.manual_seed(seed)

    domain = build_domain()
    calc_loss = df.calc_loss_simple(domain)
    model = model_factory()
    n_params = count_params(model)
    print(f"  Model: {network_description} — {n_params} trainable params")

    start = time.perf_counter()
    _, adam_best = model.train_adam(
        calc_loss=calc_loss,
        learning_rate=LR_ADAM,
        epochs=epochs_adam,
        threshold_loss=THRESHOLD_ADAM,
        do_between_epochs=None,
        print_every=max(1, epochs_adam // 10),
    )
    adam_time = time.perf_counter() - start

    start = time.perf_counter()
    _, best_model = adam_best.train_lbfgs(
        calc_loss=calc_loss,
        epochs=epochs_lbfgs,
        threshold_loss=THRESHOLD_LBFGS,
        do_between_epochs=resample_callback(domain),
        print_every=max(1, epochs_lbfgs // 10),
    )
    lbfgs_time = time.perf_counter() - start

    final_losses = calc_loss(best_model)
    area_eval = domain.area_list[0].evaluate(best_model)
    area_eval.sampling_area(EVAL_GRID)
    data = area_eval.data_dict
    outlet_eval = domain.bound_list[2].evaluate(best_model)
    outlet_eval.sampling_line(OUTLET_LINE_POINTS)
    outlet_data = outlet_eval.data_dict

    metrics = {
        "seed": seed,
        "n_params": n_params,
        "adam_time_s": float(adam_time),
        "lbfgs_time_s": float(lbfgs_time),
        "total_time_s": float(adam_time + lbfgs_time),
        "final_total_loss": float(final_losses["total_loss"].detach().cpu().item()),
        "final_bc_loss": float(final_losses["bc_loss"].detach().cpu().item()),
        "final_pde_loss": float(final_losses["pde_loss"].detach().cpu().item()),
    }
    for name in RESIDUAL_FIELDS:
        residual = np.asarray(data[f"{name}_residual"])
        metrics[f"max_{name}"] = float(np.max(np.abs(residual)))
        metrics[f"mean_abs_{name}"] = float(np.mean(np.abs(residual)))

    metrics.update({name: np.asarray(data[name]) for name in MODEL_FIELDS})
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
    metrics["outlet_y"] = np.asarray(outlet_data["y"])
    metrics["outlet_u"] = np.asarray(outlet_data["u"])
    return metrics


def _mean_std(values, num_runs: int):
    values = np.asarray(values, dtype=np.float64)
    return float(values.mean()), float(values.std(ddof=1)) if num_runs > 1 else 0.0


def aggregate(
    per_run: List[Dict[str, object]],
    label: str,
    num_runs: int,
    epochs_adam: int,
    epochs_lbfgs: int,
) -> Dict[str, object]:
    """Aggregate scalar metrics and retain the median-loss run's fields."""
    final_losses = np.asarray([run["final_total_loss"] for run in per_run])
    median_idx = int(np.argsort(final_losses)[len(final_losses) // 2])
    results = {
        "label": label,
        "n_params": per_run[0]["n_params"],
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
    return results


def run_benchmark(
    label: str,
    description: str,
    network_description: str,
    model_factory: Callable[[], object],
    results_file: Path,
):
    """Parse benchmark options, run the requested seeds, and save results."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--num_runs", type=int, default=len(SEEDS),
        help=f"Number of independent runs (default: {len(SEEDS)}).",
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

    print("=" * 80)
    print(f"{label} Benchmark — 2D Steady Cylinder Flow (Re=50)")
    print("=" * 80)
    print(f"Device:           {df.device}")
    print(f"Runs:             {args.num_runs}")
    print(f"Adam epochs:      {args.epochs_adam}")
    print(f"L-BFGS epochs:    {args.epochs_lbfgs}")
    print(f"Base seed:        {SEEDS[0]}")
    print(f"Network:          {network_description}")

    per_run = []
    for index in range(args.num_runs):
        seed = SEEDS[index] if index < len(SEEDS) else SEEDS[-1] + index - len(SEEDS) + 1
        per_run.append(
            train_one(
                seed,
                model_factory,
                label,
                network_description,
                args.epochs_adam,
                args.epochs_lbfgs,
            )
        )

    results = aggregate(
        per_run, label, args.num_runs, args.epochs_adam, args.epochs_lbfgs
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(results_file, **results)
    print(f"\nResults saved to: {results_file}")
    print("\n" + "-" * 60)
    print(f"{label}  ({results['n_params']} params, {args.num_runs} runs)")
    print(f"  final total loss : {results['final_total_loss_mean']:.4e} ± {results['final_total_loss_std']:.4e}")
    print(f"  final PDE loss   : {results['final_pde_loss_mean']:.4e} ± {results['final_pde_loss_std']:.4e}")
    print(f"  max |continuity| : {results['max_continuity_mean']:.4e} ± {results['max_continuity_std']:.4e}")
    print("-" * 60)
