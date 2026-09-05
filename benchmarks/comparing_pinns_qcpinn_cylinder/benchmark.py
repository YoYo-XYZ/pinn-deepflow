"""Shared-harness cylinder PINN/QCPINN benchmark."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Callable, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, PROJECT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import deepflow as df  # noqa: E402
from benchmarks.shared_harness import (  # noqa: E402
    BenchmarkConfig,
    aggregate_metrics,
    build_cylinder_domain,
    build_flow_pde,
    collect_metrics,
    collect_reference_metrics,
    evaluate_area,
    evaluate_line,
    plot_results,
    representative_run_index,
    save_model,
    train_one,
    write_markdown_report,
)


CHANNEL_X = (0.0, 1.1)
CHANNEL_Y = (0.0, 0.41)
CYLINDER = (0.2, 0.2, 0.05)
U_INF = 1.0
L_CHAR = 1.0
MU = 0.1
RHO = 1.0
REYNOLDS = RHO * U_INF * L_CHAR / MU
PROFILE_POINTS = 50
PROFILE_EPSILON = 1.0e-6
FEM_MESH_SIZE = 0.05
FEM_BOUNDARY_RESOLUTION = 128
FEM_TOLERANCE = 1.0e-5
FEM_MAX_ITERATIONS = 200
FEM_CACHE_NAME = "cfd_reference.npz"

QC_PRE = [32]
QC_POST = [32]
QC_NQUBITS = 4
QC_ITERATIONS = 10

RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_NAME = "REPORT.md"

DEFAULT_CONFIG = BenchmarkConfig(
    width=48,
    depth=4,
    learning_rate=0.004,
    epochs_adam=0,
    epochs_lbfgs=100,
    seed=69,
    boundary_points=[50, 50, 50, 50, 50, 50],
    interior_points=[[50, 50]],
    eval_grid=[50, 50],
    sampling="uniform",
)

SMOKE_CONFIG = BenchmarkConfig(
    width=8,
    depth=2,
    learning_rate=0.004,
    epochs_adam=2,
    epochs_lbfgs=0,
    seed=69,
    boundary_points=[4, 4, 4, 4, 4, 4],
    interior_points=[[4, 4]],
    eval_grid=[5, 5],
    sampling="uniform",
)

VARIANTS = (
    "PINN-UVP",
    "QCPINN-UVP",
    "PINN-PSIP",
    "QCPINN-PSIP",
)
FORMULATIONS = {
    "PINN-UVP": "uvp",
    "QCPINN-UVP": "uvp",
    "PINN-PSIP": "psip",
    "QCPINN-PSIP": "psip",
}


def _variant_slug(variant: str) -> str:
    return variant.lower().replace("-", "_")


def _validate_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown cylinder benchmark variant: {variant!r}")


def available_variants() -> tuple[str, ...]:
    """Return variants whose optional model backends are installed."""
    if hasattr(df, "QCPINN"):
        return VARIANTS
    return tuple(variant for variant in VARIANTS if not variant.startswith("QCPINN"))


def build_pde(formulation: str):
    """Build the shared cylinder flow PDE for one formulation."""
    return build_flow_pde(
        formulation,
        U=U_INF,
        L=L_CHAR,
        mu=MU,
        rho=RHO,
    )


def build_domain(
    formulation: str,
    config: BenchmarkConfig = DEFAULT_CONFIG,
):
    """Build one sampled cylinder domain through the shared builder."""
    df.manual_seed(config.seed)
    return build_cylinder_domain(
        formulation=formulation,
        boundary_points=list(config.boundary_points),
        interior_points=list(config.interior_points),
        sampling=config.sampling,
        channel_x=CHANNEL_X,
        channel_y=CHANNEL_Y,
        cylinder=CYLINDER,
        u_inf=U_INF,
        L=L_CHAR,
        mu=MU,
        rho=RHO,
    )


def _output_vars(formulation: str) -> list[str]:
    if formulation == "uvp":
        return ["u", "v", "p"]
    if formulation == "psip":
        return ["psi", "p"]
    raise ValueError(f"Unknown formulation: {formulation!r}")


def build_pinn_model(
    formulation: str,
    config: BenchmarkConfig = DEFAULT_CONFIG,
):
    """Build the standard PINN variant for one formulation."""
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=_output_vars(formulation),
        width=config.width,
        length=config.depth,
    )


def build_qcpinn_model(
    formulation: str,
    config: BenchmarkConfig = DEFAULT_CONFIG,
):
    """Build the QCPINN variant for one formulation."""
    # FLEX: QCPINN construction requires the optional PennyLane backend.
    try:
        import pennylane  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "PennyLane is required for QCPINN variants; install it with "
            "pip install pennylane."
        ) from exc
    try:
        qcpinn = df.QCPINN
    except AttributeError as exc:
        raise RuntimeError("DeepFlow was imported without QCPINN support.") from exc
    return qcpinn(
        input_vars=["x", "y"],
        output_vars=_output_vars(formulation),
        hidden_layer_pre=QC_PRE,
        hidden_layer_post=QC_POST,
        nqubits=QC_NQUBITS,
        q_layer_iterations=QC_ITERATIONS,
    )


def build_model(
    variant: str,
    config: BenchmarkConfig = DEFAULT_CONFIG,
):
    """Build a named model variant, leaving formulation selection explicit."""
    _validate_variant(variant)
    formulation = FORMULATIONS[variant]
    if variant.startswith("QC"):
        return build_qcpinn_model(formulation, config)
    return build_pinn_model(formulation, config)


def _profile_geometries():
    cx, cy, radius = CYLINDER
    x_min, x_max = CHANNEL_X
    y_min, y_max = CHANNEL_Y
    epsilon = PROFILE_EPSILON
    outlet = df.geometry.line_vertical(
        x=x_max - epsilon,
        range_y=[y_min + epsilon, y_max - epsilon],
    )
    wake = df.geometry.line_horizontal(
        y=cy,
        range_x=[cx + radius + epsilon, x_max - epsilon],
    )
    return outlet, wake


def evaluate_profiles(
    model,
    formulation: str,
    points: int = PROFILE_POINTS,
):
    """Evaluate outlet and wake profiles through one shared path."""
    outlet, wake = _profile_geometries()
    return {
        "outlet": evaluate_line(outlet, model, build_pde(formulation), points),
        "wake": evaluate_line(wake, model, build_pde(formulation), points),
    }


def _profile_reference_metrics(profiles: Mapping[str, object], reference_solution):
    metrics = {}
    for name, evaluator in profiles.items():
        values = collect_reference_metrics(
            evaluator,
            reference_solution,
            field="u",
            final_coordinate=None,
        )
        metrics.update({f"{name}_{key}": value for key, value in values.items()})
    return metrics


def run_once(
    variant: str,
    config: BenchmarkConfig,
    reference_solution=None,
    model_factory: Callable[[BenchmarkConfig], object] | None = None,
):
    """Train, evaluate, and collect metrics for one seeded variant."""
    _validate_variant(variant)
    formulation = FORMULATIONS[variant]
    domain = build_domain(formulation, config)
    factory = (
        (lambda: model_factory(config))
        if model_factory is not None
        else (lambda: build_model(variant, config))
    )
    model, training_info = train_one(domain, factory, config)
    evaluator = evaluate_area(domain, model, list(config.eval_grid))
    profiles = evaluate_profiles(model, formulation)
    metrics = {
        **training_info,
        **collect_metrics(evaluator, model, reference_solution=reference_solution),
        "variant": variant,
        "formulation": formulation,
        "model_type": "QCPINN" if variant.startswith("QC") else "PINN",
        "seed": config.seed,
        "trainable_parameters": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "reynolds": REYNOLDS,
    }
    if reference_solution is not None:
        metrics.update(_profile_reference_metrics(profiles, reference_solution))
    return {
        "variant": variant,
        "formulation": formulation,
        "model": model,
        "domain": domain,
        "evaluator": evaluator,
        "profiles": profiles,
        "metrics": metrics,
    }


def run_variant(
    variant: str,
    config: BenchmarkConfig,
    output_dir: Path,
    reference_solution=None,
    model_factory: Callable[[BenchmarkConfig], object] | None = None,
):
    """Run, persist, plot, and report one cylinder model variant."""
    _validate_variant(variant)
    output_dir = Path(output_dir)
    runs = [
        run_once(
            variant,
            replace(config, seed=seed, seeds=[seed]),
            reference_solution=reference_solution,
            model_factory=model_factory,
        )
        for seed in config.seeds
    ]
    representative_index = representative_run_index(
        [run["metrics"] for run in runs]
    )
    representative = runs[representative_index]
    metrics = {
        **representative["metrics"],
        **aggregate_metrics([run["metrics"] for run in runs]),
        "num_runs": len(runs),
        "representative_run_idx": representative_index,
    }
    prefix = f"cylinder_{_variant_slug(variant)}"
    model_path = save_model(representative["model"], output_dir / prefix)
    artifacts = [model_path]
    artifacts.extend(
        plot_results(representative["evaluator"], output_dir, prefix=prefix)
    )
    return {
        **representative,
        "metrics": metrics,
        "artifacts": artifacts,
        "model_path": model_path,
        "runs": runs,
        "representative_run_idx": representative_index,
    }


def run_suite(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
    variants: Sequence[str] | None = None,
    reference_solution=None,
):
    """Run available cylinder variants through the shared harness."""
    variants = available_variants() if variants is None else tuple(variants)
    if not variants:
        raise ValueError("At least one cylinder benchmark variant is required.")
    if len(set(variants)) != len(variants):
        raise ValueError("Cylinder benchmark variants must be unique.")
    for variant in variants:
        _validate_variant(variant)

    output_dir = Path(output_dir)
    results = {
        variant: run_variant(
            variant,
            config,
            output_dir,
            reference_solution=reference_solution,
        )
        for variant in variants
    }
    report_metrics = {
        "variants": ", ".join(variants),
        "reynolds": REYNOLDS,
        "channel_x": CHANNEL_X,
        "channel_y": CHANNEL_Y,
        "cylinder": CYLINDER,
    }
    artifacts = []
    for variant, result in results.items():
        report_metrics.update(
            {
                f"{_variant_slug(variant)}_{key}": value
                for key, value in result["metrics"].items()
            }
        )
        artifacts.extend(result["artifacts"])
    report = write_markdown_report(
        output_dir / REPORT_NAME,
        "Cylinder PINN/QCPINN benchmark",
        config,
        report_metrics,
        artifacts,
    )
    return {"variants": results, "report": report, "artifacts": artifacts}


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a non-negative integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def _config_from_args(args) -> BenchmarkConfig:
    config = SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG
    if args.num_runs is not None:
        config = replace(
            config,
            seeds=[config.seed + index for index in range(args.num_runs)],
        )
    if args.epochs_adam is not None:
        config = replace(config, epochs_adam=args.epochs_adam)
    if args.epochs_lbfgs is not None:
        config = replace(config, epochs_lbfgs=args.epochs_lbfgs)
    return config


def run_variant_cli(
    variant: str,
    model_factory: Callable[[BenchmarkConfig], object] | None = None,
    argv=None,
):
    """Run one variant with the legacy per-cell command-line options."""
    parser = argparse.ArgumentParser(description=f"Run the {variant} cylinder benchmark.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--num_runs", type=_positive_int)
    parser.add_argument("--epochs_adam", type=_nonnegative_int)
    parser.add_argument("--epochs_lbfgs", type=_nonnegative_int)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)
    config = _config_from_args(args)
    result = run_variant(
        variant,
        config,
        args.output_dir,
        model_factory=model_factory,
    )
    metrics = result["metrics"]
    print(
        f"{variant}: final total loss={metrics['final_total_loss']:.6e}, "
        f"PDE residual={metrics.get('mean_abs_pde_residual', float('nan')):.6e}"
    )
    print(f"Model: {result['model_path']}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="PINN-UVP")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run_suite(
        SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG,
        args.output_dir,
        variants=(args.variant,),
    )
