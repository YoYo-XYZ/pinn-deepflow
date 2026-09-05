"""Shared-harness Burgers benchmark for PINN and RFFPINN."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import deepflow as df  # noqa: E402
from benchmarks.shared_harness import (  # noqa: E402
    BenchmarkConfig,
    build_burgers_domain,
    collect_metrics,
    evaluate_area,
    plot_results,
    save_model,
    train_one,
    write_markdown_report,
)


NU = 0.01 / math.pi
RFF_EMBED_DIM = 256
RFF_ALPHA = 5.0
FEM_MESH_SIZE = 0.02
FEM_TIME_STEP = 0.01
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "REPORT.md"
VARIANTS = ("PINN", "RFFPINN")

DEFAULT_CONFIG = BenchmarkConfig(
    width=16,
    depth=4,
    learning_rate=0.004,
    epochs_adam=500,
    epochs_lbfgs=100,
    r3_interval=100,
    seed=69,
    boundary_points=[512, 256, 256],
    interior_points=[1024],
    eval_grid=[161, 81],
    sampling="lhs",
)

SMOKE_CONFIG = BenchmarkConfig(
    width=8,
    depth=2,
    learning_rate=0.004,
    epochs_adam=2,
    epochs_lbfgs=0,
    seed=69,
    boundary_points=[8, 4, 4],
    interior_points=[16],
    eval_grid=[8, 8],
    sampling="lhs",
)


def build_domain(config: BenchmarkConfig):
    """Build one sampled Burgers domain through the shared builder."""
    df.manual_seed(config.seed)
    return build_burgers_domain(
        nu=NU,
        boundary_points=list(config.boundary_points),
        interior_points=list(config.interior_points),
        sampling=config.sampling,
    )


def build_pinn(config: BenchmarkConfig):
    """Construct the standard model variant."""
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=config.width,
        length=config.depth,
    )


def build_rffpinn(config: BenchmarkConfig):
    """Construct the random-feature model variant."""
    return df.RFFPINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=config.width,
        length=config.depth,
        embed_dim=RFF_EMBED_DIM,
        alpha=RFF_ALPHA,
    )


MODEL_BUILDERS: dict[str, Callable[[BenchmarkConfig], object]] = {
    "PINN": build_pinn,
    "RFFPINN": build_rffpinn,
}


def build_model(variant: str, config: BenchmarkConfig):
    """Construct a named model variant."""
    try:
        builder = MODEL_BUILDERS[variant]
    except KeyError as exc:
        raise ValueError(f"Unknown model variant: {variant!r}") from exc
    return builder(config)


def run_variant(
    variant: str,
    config: BenchmarkConfig,
    output_dir: Path,
    reference_solution=None,
) -> dict:
    """Train, evaluate, save, and plot one model variant."""
    domain = build_domain(config)
    model, training_info = train_one(
        domain,
        lambda: build_model(variant, config),
        config,
    )
    evaluator = evaluate_area(domain, model, list(config.eval_grid))
    evaluation_metrics = (
        collect_metrics(
            evaluator,
            model,
            reference_solution=reference_solution,
        )
        if reference_solution is not None
        else collect_metrics(evaluator, model)
    )
    metrics = {
        **training_info,
        **evaluation_metrics,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    model_path = save_model(model, Path(output_dir) / variant.lower())
    artifacts = [model_path]
    artifacts.extend(plot_results(evaluator, output_dir, prefix=variant.lower()))
    return {
        "variant": variant,
        "model": model,
        "domain": domain,
        "evaluator": evaluator,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def run_suite(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
    reference_solution=None,
) -> dict:
    """Run both model variants through the shared benchmark harness."""
    output_dir = Path(output_dir)
    results = {
        variant: run_variant(
            variant,
            config,
            output_dir,
            reference_solution=reference_solution,
        )
        for variant in VARIANTS
    }

    report_metrics = {}
    artifacts = []
    for variant, result in results.items():
        report_metrics.update(
            {
                f"{variant.lower()}_{key}": value
                for key, value in result["metrics"].items()
            }
        )
        artifacts.extend(result["artifacts"])
    report = write_markdown_report(
        output_dir / REPORT_PATH.name,
        "Burgers PINN vs RFFPINN benchmark",
        config,
        report_metrics,
        artifacts,
    )
    return {"variants": results, "report": report, "artifacts": artifacts}


def solve_reference(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Solve the shared Burgers domain with the optional FEM backend."""
    reference = build_domain(config).solve_fem(
        mesh_size=FEM_MESH_SIZE,
        time_step=FEM_TIME_STEP,
        tolerance=1e-8,
        max_iterations=50,
    )
    if not reference.metadata.get("converged", True):
        raise RuntimeError("FEM reference did not converge")
    return reference


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the small CPU-friendly suite without solving FEM.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the optional FEM reference solve.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for native models, plots, and the report.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG
    reference = None
    if not args.smoke and not args.no_reference:
        try:
            print("Solving FEM reference...")
            reference = solve_reference(config)
        except ImportError as exc:
            print(f"Skipping optional FEM reference: {exc}")

    result = run_suite(config, args.output_dir, reference_solution=reference)
    print("\nBenchmark results")
    for variant, values in result["variants"].items():
        metrics = values["metrics"]
        summary = (
            f"{variant:7s} total loss={metrics['final_total_loss']:.6e}, "
            f"PDE residual={metrics['mean_abs_pde_residual']:.6e}"
        )
        if "relative_l2" in metrics:
            summary += f", relative L2={metrics['relative_l2']:.6e}"
        print(summary)
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
