"""Shared-harness Burgers RFFPINN sampling benchmark."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path


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
R3_INTERVAL = 100
FEM_MESH_SIZE = 0.02
FEM_TIME_STEP = 0.01
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "REPORT.md"
SCHEMES = ("lhs", "uniform", "r3")

DEFAULT_CONFIG = BenchmarkConfig(
    width=16,
    depth=4,
    learning_rate=0.004,
    epochs_adam=1000,
    epochs_lbfgs=0,
    r3_interval=R3_INTERVAL,
    seed=69,
    boundary_points=[512, 256, 256],
    interior_points=[[32, 32]],
    eval_grid=[161, 81],
    sampling="lhs",
)

SMOKE_CONFIG = replace(
    DEFAULT_CONFIG,
    width=8,
    depth=2,
    epochs_adam=2,
    boundary_points=[8, 4, 4],
    interior_points=[[4, 4]],
    eval_grid=[8, 8],
    r3_interval=1,
)


def _check_scheme(scheme: str) -> None:
    if scheme not in SCHEMES:
        raise ValueError(f"Unknown sampling scheme: {scheme!r}")


def _initial_sampling(scheme: str) -> str:
    _check_scheme(scheme)
    return "lhs" if scheme == "r3" else scheme


def config_for_scheme(scheme: str, config: BenchmarkConfig) -> BenchmarkConfig:
    """Return the shared config with only this scheme's sampling changed."""
    initial_sampling = _initial_sampling(scheme)
    if scheme == "r3":
        r3_interval = config.r3_interval or R3_INTERVAL
    else:
        r3_interval = 0
    return replace(
        config,
        sampling=initial_sampling,
        r3_interval=r3_interval,
    )


def build_domain(
    scheme: str = "lhs", config: BenchmarkConfig = DEFAULT_CONFIG
):
    """Build one sampled Burgers domain through the shared builder."""
    effective_config = config_for_scheme(scheme, config)
    df.manual_seed(effective_config.seed)
    return build_burgers_domain(
        nu=NU,
        boundary_points=list(effective_config.boundary_points),
        interior_points=list(effective_config.interior_points),
        sampling=effective_config.sampling,
    )


def build_model(config: BenchmarkConfig):
    """Construct the one RFFPINN model used by every sampling scheme."""
    return df.RFFPINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=config.width,
        length=config.depth,
        embed_dim=RFF_EMBED_DIM,
        alpha=RFF_ALPHA,
    )


def run_variant(
    scheme: str,
    config: BenchmarkConfig,
    output_dir: Path,
    reference_solution=None,
) -> dict:
    """Train, evaluate, save, and report one sampling scheme."""
    effective_config = config_for_scheme(scheme, config)
    domain = build_domain(scheme, config)
    model, training_info = train_one(
        domain,
        lambda: build_model(effective_config),
        effective_config,
    )
    evaluator = evaluate_area(
        domain, model, list(effective_config.eval_grid)
    )
    metrics = {
        **training_info,
        **collect_metrics(
            evaluator,
            model,
            reference_solution=reference_solution,
        ),
        "sampling_scheme": scheme,
        "initial_sampling": effective_config.sampling,
        "r3_interval": effective_config.r3_interval,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    model_path = save_model(
        model, Path(output_dir) / f"rffpinn_{scheme}"
    )
    artifacts = [model_path]
    artifacts.extend(
        plot_results(evaluator, output_dir, prefix=f"rffpinn_{scheme}")
    )
    return {
        "scheme": scheme,
        "config": effective_config,
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
    """Run all sampling schemes through the shared benchmark harness."""
    output_dir = Path(output_dir)
    results = {
        scheme: run_variant(
            scheme,
            config,
            output_dir,
            reference_solution=reference_solution,
        )
        for scheme in SCHEMES
    }

    report_metrics = {
        "sampling_schemes": ", ".join(SCHEMES),
        "rff_embed_dim": RFF_EMBED_DIM,
        "rff_alpha": RFF_ALPHA,
        "nu": NU,
    }
    artifacts = []
    for scheme, result in results.items():
        report_metrics.update(
            {
                f"{scheme}_{key}": value
                for key, value in result["metrics"].items()
            }
        )
        artifacts.extend(result["artifacts"])
    report = write_markdown_report(
        output_dir / REPORT_PATH.name,
        "Burgers RFFPINN sampling benchmark",
        config,
        report_metrics,
        artifacts,
    )
    return {"variants": results, "report": report, "artifacts": artifacts}


def solve_reference(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Solve the shared Burgers domain with the optional FEM backend."""
    reference = build_domain("lhs", config).solve_fem(
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
        except (ImportError, OSError) as exc:
            print(f"Skipping optional FEM reference: {exc}")

    result = run_suite(config, args.output_dir, reference_solution=reference)
    print("\nSampling comparison")
    for scheme, values in result["variants"].items():
        metrics = values["metrics"]
        summary = (
            f"{scheme:7s} total loss={metrics['final_total_loss']:.6e}, "
            f"PDE residual={metrics.get('mean_abs_pde_residual', float('nan')):.6e}"
        )
        if "relative_l2" in metrics:
            summary += f", relative L2={metrics['relative_l2']:.6e}"
        print(summary)
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
