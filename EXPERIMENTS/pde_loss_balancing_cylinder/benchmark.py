#!/usr/bin/env python3
"""Paired benchmark of default and per-equation-balanced cylinder-flow PINNs."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_SRC))

import deepflow as df  # noqa: E402
from loss_balancing import (  # noqa: E402
    FixedWeightedPDELoss,
    GradientBalancedPDELoss,
    raw_pde_term_losses,
)


EQUATIONS = ("continuity", "x_momentum", "y_momentum")
PHYSICS = {"U": 1.0, "mu": 0.2, "rho": 1.0, "L": 1.0}
FIXED_WEIGHTS = (1.0, 0.05, 0.05)


def build_problem(seed: int, boundary_points: int, interior_points: int):
    """Build the requested Re=5 cylinder problem and its initial model."""
    df.manual_seed(seed, deterministic=True)
    circle = df.geometry.circle(0.2, 0.2, 0.05)
    rectangle = df.geometry.rectangle([0.0, 1.1], [0.0, 0.41])
    area = rectangle - circle
    domain = df.domain(area, circle.bound_list)

    domain.bound_list[0].define_bc(
        {"u": ["y", lambda y: 4.0 * (0.41 - y) * y / 0.41**2], "v": 0}
    )
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.bound_list[3].define_bc({"u": 0, "v": 0})
    domain.bound_list[4].define_bc({"u": 0, "v": 0})
    domain.bound_list[5].define_bc({"u": 0, "v": 0})
    domain.area_list[0].define_pde(df.NavierStokes(**PHYSICS))
    domain.sampling_lhs([boundary_points] * 6, [interior_points])

    model = df.PINN(
        width=32,
        length=5,
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
    )
    return domain, model


def fem_reference(args) -> dict[str, object]:
    """Load or generate the shared NGSolve reference on a masked uniform grid."""
    cache = args.output_dir / "fem_reference.npz"
    expected = {
        "U": PHYSICS["U"],
        "mu": PHYSICS["mu"],
        "rho": PHYSICS["rho"],
        "L": PHYSICS["L"],
        "mesh_size": args.fem_mesh_size,
        "nx": args.fem_grid[0],
        "ny": args.fem_grid[1],
    }
    if cache.exists():
        loaded = dict(np.load(cache, allow_pickle=False))
        if all(np.asarray(loaded[key]).item() == value for key, value in expected.items()):
            print(f"Using cached FEM reference: {cache}")
            return loaded

    print("Generating NGSolve FEM reference...")
    domain, _ = build_problem(args.seeds[0] + 20_000, 20, 50)
    started = time.perf_counter()
    reference = domain.solve_fem(
        mesh_size=args.fem_mesh_size,
        boundary_resolution=args.fem_boundary_resolution,
        tolerance=1e-8,
        max_iterations=200,
        area_sampling_res=list(args.fem_grid),
        bound_sampling_res=args.fem_boundary_resolution,
    )
    metadata = reference.metadata
    if not metadata.get("converged", False):
        raise RuntimeError("FEM reference did not converge")
    data = reference.area_evaluators[0].data_dict
    residuals = np.asarray(metadata.get("solver_residuals", []), dtype=float)
    mesh = metadata.get("mesh", {})
    payload = {
        **{key: np.asarray(value) for key, value in expected.items()},
        "x": np.asarray(data["x"]),
        "y": np.asarray(data["y"]),
        "u": np.asarray(data["u_ref"]),
        "v": np.asarray(data["v_ref"]),
        "p": np.asarray(data["p_ref"]),
        "converged": np.asarray(True),
        "iterations": np.asarray(metadata.get("iterations", -1)),
        "final_residual": np.asarray(residuals[-1] if residuals.size else np.nan),
        "runtime_s": np.asarray(time.perf_counter() - started),
        "mesh_elements": np.asarray(mesh.get("elements", -1)),
        "mesh_vertices": np.asarray(mesh.get("vertices", -1)),
        "pressure_gauge": np.asarray(metadata.get("pressure_gauge", "unknown")),
    }
    np.savez(cache, **payload)
    print(f"Wrote FEM reference: {cache}")
    return payload


def synchronize() -> None:
    if torch.cuda.is_available() and str(df.device).startswith("cuda"):
        torch.cuda.synchronize()


def _error_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(prediction) & np.isfinite(reference)
    difference = prediction[valid] - reference[valid]
    reference_valid = reference[valid]
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "relative_l2": float(
            np.linalg.norm(difference) / max(np.linalg.norm(reference_valid), 1e-12)
        ),
    }


def evaluate(model, seed: int, evaluation_points: int, fem: dict[str, object]) -> dict[str, object]:
    """Evaluate residuals freshly and solution error at the FEM coordinates."""
    domain, _ = build_problem(seed + 10_000, 50, evaluation_points)
    terms = raw_pde_term_losses(domain, model).cpu().numpy()
    area = domain.area_list[0]
    area.calc_residual_field(model)
    residuals = area.residual_field_raw.detach().cpu().numpy()
    metrics = {
        "pde_term_mse": dict(zip(EQUATIONS, terms.tolist())),
        "pde_loss": float(terms.sum()),
        "mean_abs_residual": dict(zip(EQUATIONS, np.mean(np.abs(residuals), axis=1).tolist())),
        "max_abs_residual": dict(zip(EQUATIONS, np.max(np.abs(residuals), axis=1).tolist())),
    }

    fem_domain, _ = build_problem(seed + 30_000, 10, 20)
    fem_area = fem_domain.area_list[0]
    fem_area.X = torch.as_tensor(np.asarray(fem["x"]), dtype=df.dtype)
    fem_area.Y = torch.as_tensor(np.asarray(fem["y"]), dtype=df.dtype)
    fem_area.process_coordinates()
    prediction = {
        key: value.detach().cpu().numpy()
        for key, value in fem_area.process_model(model).items()
    }
    field_errors = {
        key: _error_metrics(prediction[key], np.asarray(fem[key]))
        for key in ("u", "v", "p")
    }
    speed_prediction = np.hypot(prediction["u"], prediction["v"])
    speed_reference = np.hypot(np.asarray(fem["u"]), np.asarray(fem["v"]))
    field_errors["velocity_magnitude"] = _error_metrics(speed_prediction, speed_reference)
    metrics["fem_error"] = field_errors
    return metrics


def _train_adam(model, loss_fn, epochs: int, learning_rate: float, balancer=None, interval=200):
    """Train while tying balancing updates to completed optimizer epochs."""
    model = copy.deepcopy(model).to(df.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history = {"total_loss": [], "bc_loss": [], "pde_loss": []}

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = loss_fn(model)
        losses["total_loss"].backward()
        optimizer.step()

        for key in history:
            history[key].append(float(losses[key].detach().cpu()))

        # The first new weights are computed after epoch 200 and used starting
        # at epoch 201. This remains exact even if other code evaluates losses.
        if balancer is not None and epoch % interval == 0:
            balancer.update(model, epoch)

        if epoch == 1 or epoch % 200 == 0 or epoch == epochs:
            weights = "" if balancer is None or balancer.weights is None else (
                f", weights={balancer.weights.cpu().tolist()}"
            )
            print(
                f"Epoch {epoch}: BC={history['bc_loss'][-1]:.5g}, "
                f"raw PDE={history['pde_loss'][-1]:.5g}{weights}"
            )

    model.loss_history = history
    return model


def train_one(args, method: str, seed: int, fem) -> tuple[object, dict[str, object], object | None]:
    domain, model = build_problem(seed, args.boundary_points, args.interior_points)
    balancer = None
    if method == "default":
        loss_fn = df.calc_loss_simple(domain)
    elif method == "fixed":
        loss_fn = FixedWeightedPDELoss(domain, FIXED_WEIGHTS)
    elif method == "balanced":
        balancer = GradientBalancedPDELoss(
            domain,
            scope=args.scope,
            alpha=args.alpha,
        )

        loss_fn = balancer
    else:
        raise ValueError(f"Unknown benchmark method: {method}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    synchronize()
    started = time.perf_counter()
    trained = _train_adam(
        model,
        loss_fn,
        args.epochs,
        args.learning_rate,
        balancer=balancer,
        interval=args.balance_every,
    )
    synchronize()
    elapsed = time.perf_counter() - started

    history = trained.loss_history
    metrics = evaluate(trained, seed, args.evaluation_points, fem)
    metrics.update(
        {
            "method": method,
            "seed": seed,
            "train_time_s": elapsed,
            "peak_cuda_memory_mb": (
                torch.cuda.max_memory_allocated() / 1024**2 if torch.cuda.is_available() else None
            ),
            "final_training_bc_loss": history["bc_loss"][-1],
            "final_training_pde_loss": history["pde_loss"][-1],
            "final_training_raw_total": history["bc_loss"][-1] + history["pde_loss"][-1],
            "loss_history": {key: list(values) for key, values in history.items()},
        }
    )
    if balancer is not None:
        metrics["balancing_diagnostics"] = balancer.diagnostics
        metrics["final_weights"] = balancer.weights.cpu().tolist()
    return trained, metrics, balancer


def method_config(args, method: str, seed: int) -> dict[str, object]:
    """Return the settings that determine one method's cached result."""
    config = {
        "method": method,
        "seed": seed,
        "epochs": args.epochs,
        "boundary_points": args.boundary_points,
        "interior_points": args.interior_points,
        "evaluation_points": args.evaluation_points,
        "learning_rate": args.learning_rate,
        "physics": PHYSICS,
    }
    if method == "fixed":
        config["fixed_weights"] = list(FIXED_WEIGHTS)
    if method == "balanced":
        config.update(
            scope=args.scope,
            balance_every=args.balance_every,
            alpha=args.alpha,
        )
    return config


def plot_results(results: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for result in results:
        history = result["loss_history"]
        label = f"{result['method']} (seed {result['seed']})"
        axes[0].semilogy(history["bc_loss"], label=f"{label}: BC", alpha=0.8)
        axes[0].semilogy(history["pde_loss"], "--", label=f"{label}: raw PDE", alpha=0.8)
    axes[0].set(xlabel="Adam epoch", ylabel="Unweighted loss", title="Training losses")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.25)

    labels = list(EQUATIONS)
    x = np.arange(len(labels))
    width = 0.8 / len(results)
    for index, result in enumerate(results):
        values = [result["pde_term_mse"][key] for key in labels]
        axes[1].bar(x + index * width, values, width, label=f"{result['method']} s{result['seed']}")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x + width * (len(results) - 1) / 2, labels, rotation=15)
    axes[1].set(ylabel="Evaluation MSE", title="Per-equation residual error")
    axes[1].legend(fontsize=7)
    axes[1].grid(axis="y", alpha=0.25)

    fields = ("u", "v", "p", "velocity_magnitude")
    x = np.arange(len(fields))
    width = 0.8 / len(results)
    for index, result in enumerate(results):
        values = [result["fem_error"][field]["relative_l2"] for field in fields]
        axes[2].bar(x + index * width, values, width, label=result["method"])
    axes[2].set_yscale("log")
    axes[2].set_xticks(x + width * (len(results) - 1) / 2, ("u", "v", "p", "|V|"))
    axes[2].set(ylabel="Relative L2 error", title="Error against FEM")
    axes[2].legend(fontsize=7)
    axes[2].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_weight_history(results: list[dict[str, object]], epochs: int, output: Path) -> None:
    """Plot the piecewise-constant PDE equation weights over optimizer epochs."""
    balanced_runs = [result for result in results if result["method"] == "balanced"]
    if not balanced_runs:
        return

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_index, result in enumerate(balanced_runs):
        diagnostics = result.get("balancing_diagnostics", [])
        update_epochs = [0] + [int(item["epoch"]) for item in diagnostics]
        weight_rows = [[1.0] * len(EQUATIONS)] + [item["weights"] for item in diagnostics]

        # Extend the last frozen values to the final plotted epoch.
        if update_epochs[-1] < epochs:
            update_epochs.append(epochs)
            weight_rows.append(weight_rows[-1])

        weights = np.asarray(weight_rows, dtype=float)
        for equation_index, equation in enumerate(EQUATIONS):
            label = equation if run_index == 0 else f"{equation} (seed {result['seed']})"
            ax.step(
                update_epochs,
                weights[:, equation_index],
                where="post",
                linewidth=2,
                label=label,
            )

    ax.axhline(1.0, color="black", linestyle=":", linewidth=1, alpha=0.6)
    ax.set(
        xlabel="Adam epoch",
        ylabel="PDE loss weight",
        title="Per-equation gradient-balancing weight history",
        xlim=(0, epochs),
    )
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def write_report(results: list[dict[str, object]], args, fem, output: Path) -> None:
    methods = {
        name: [result for result in results if result["method"] == name]
        for name in ("default", "fixed", "balanced")
    }

    def mean(items, key):
        return float(np.mean([item[key] for item in items]))

    lines = [
        "# Cylinder-flow PDE loss-balancing benchmark",
        "",
        f"Re = 5 (`U=1`, `mu=0.2`, `rho=1`, `L=1`), Adam epochs = {args.epochs}, "
        f"points = 6x{args.boundary_points} boundary + {args.interior_points} interior.",
        f"Balancer: scope = `{args.scope}`, update after every {args.balance_every} optimizer epochs, "
        f"weight smoothing alpha = {args.alpha}.",
        f"FEM: NGSolve, mesh size {args.fem_mesh_size}, {int(np.asarray(fem['mesh_elements']))} elements, "
        f"final nonlinear residual {float(np.asarray(fem['final_residual'])):.3e}, "
        f"pressure gauge `{np.asarray(fem['pressure_gauge']).item()}`.",
        "",
        "All PDE losses below are raw, unweighted losses evaluated on a fresh paired collocation set.",
        "",
        "| Method | Eval PDE loss | Continuity MSE | X-momentum MSE | Y-momentum MSE | Time (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method_results in methods.values():
        name = method_results[0]["method"]
        term_means = {
            equation: float(np.mean([item["pde_term_mse"][equation] for item in method_results]))
            for equation in EQUATIONS
        }
        lines.append(
            f"| {name} | {mean(method_results, 'pde_loss'):.6g} | "
            f"{term_means['continuity']:.6g} | {term_means['x_momentum']:.6g} | "
            f"{term_means['y_momentum']:.6g} | {mean(method_results, 'train_time_s'):.2f} |"
        )

    fixed_ratio = mean(methods["fixed"], "pde_loss") / mean(methods["default"], "pde_loss")
    balanced_ratio = mean(methods["balanced"], "pde_loss") / mean(methods["default"], "pde_loss")
    time_ratio = mean(methods["balanced"], "train_time_s") / mean(methods["default"], "train_time_s")
    lines.extend(
        [
            "",
            f"Fixed/default evaluation PDE-loss ratio: **{fixed_ratio:.3f}**.",
            f"Balanced/default evaluation PDE-loss ratio: **{balanced_ratio:.3f}**.",
            f"Balanced/default training-time ratio: **{time_ratio:.2f}x**.",
            "",
            "## FEM solution errors",
            "",
            "| Method | u rel-L2 | v rel-L2 | p rel-L2 | |V| rel-L2 | u MAE | v MAE | p MAE |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method_results in methods.values():
        name = method_results[0]["method"]
        def fem_mean(field, metric):
            return float(np.mean([item["fem_error"][field][metric] for item in method_results]))
        lines.append(
            f"| {name} | {fem_mean('u', 'relative_l2'):.6g} | "
            f"{fem_mean('v', 'relative_l2'):.6g} | {fem_mean('p', 'relative_l2'):.6g} | "
            f"{fem_mean('velocity_magnitude', 'relative_l2'):.6g} | "
            f"{fem_mean('u', 'mae'):.6g} | {fem_mean('v', 'mae'):.6g} | "
            f"{fem_mean('p', 'mae'):.6g} |"
        )
    lines.extend([
        "",
        f"At each update, applied weights satisfy `lambda_new = {args.alpha} * "
        f"lambda_old + {1.0 - args.alpha:.1f} * lambda_hat_new`, followed by "
        "mean-one normalization. Weights are detached and frozen between the "
        "stated optimizer-epoch boundaries.",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--boundary-points", type=int, default=200)
    parser.add_argument("--interior-points", type=int, default=1000)
    parser.add_argument("--evaluation-points", type=int, default=4000)
    parser.add_argument("--learning-rate", type=float, default=0.004)
    parser.add_argument("--seeds", type=int, nargs="+", default=[69])
    parser.add_argument("--scope", choices=("full", "last_layer"), default="full")
    parser.add_argument("--balance-every", type=int, default=500)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--fem-mesh-size", type=float, default=0.05)
    parser.add_argument("--fem-boundary-resolution", type=int, default=64)
    parser.add_argument("--fem-grid", type=int, nargs=2, default=[180, 70])
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument(
        "--methods",
        choices=("default", "fixed", "balanced"),
        nargs="+",
        default=["default", "fixed", "balanced"],
        help="Methods to ensure are present; matching cached results are reused.",
    )
    parser.add_argument("--force", action="store_true", help="Retrain selected methods.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"DeepFlow device: {df.device}")
    fem = fem_reference(args)
    json_path = args.output_dir / "benchmark_results.json"
    results = []
    if json_path.exists() and not args.force:
        results = json.loads(json_path.read_text(encoding="utf-8"))

    for seed in args.seeds:
        for method in args.methods:
            config = method_config(args, method, seed)
            cached = next(
                (item for item in results if item.get("benchmark_config") == config),
                None,
            )
            if cached is not None and not args.force:
                print(f"Reusing cached {method} result for seed {seed}")
                continue
            results = [
                item for item in results
                if not (item.get("method") == method and item.get("seed") == seed)
            ]
            _, metrics, _ = train_one(args, method, seed, fem)
            metrics["benchmark_config"] = config
            results.append(metrics)

    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    plot_results(results, args.output_dir / "benchmark_comparison.png")
    plot_weight_history(results, args.epochs, args.output_dir / "pde_weight_history.png")
    write_report(results, args, fem, args.output_dir / "benchmark_report.md")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
