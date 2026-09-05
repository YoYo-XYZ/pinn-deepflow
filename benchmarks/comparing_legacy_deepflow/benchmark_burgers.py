#!/usr/bin/env python3
"""Run one version of the shared Burgers benchmark."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path


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
    build_burgers_domain,
    collect_metrics,
    evaluate_area,
    plot_results,
    representative_run_index,
    save_model,
    train_one,
    write_markdown_report,
)


NU = 0.01 / math.pi
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_NAME = "REPORT.md"
DEFAULT_VERSION = "new"

DEFAULT_CONFIG = BenchmarkConfig(
    width=16,
    depth=4,
    learning_rate=0.004,
    epochs_adam=2000,
    epochs_lbfgs=0,
    seed=69,
    boundary_points=[1000, 500, 500],
    interior_points=[4000],
    eval_grid=[500, 250],
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


def build_domain(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build one sampled Burgers domain through the shared builder."""
    df.manual_seed(config.seed)
    return build_burgers_domain(
        nu=NU,
        boundary_points=list(config.boundary_points),
        interior_points=list(config.interior_points),
        sampling=config.sampling,
    )


def build_model(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build the PINN used for both version runs."""
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=config.width,
        length=config.depth,
    )


def run_once(config: BenchmarkConfig) -> dict:
    """Train and evaluate one seeded run."""
    domain = build_domain(config)
    model, training_info = train_one(
        domain,
        lambda: build_model(config),
        config,
    )
    evaluator = evaluate_area(domain, model, list(config.eval_grid))
    metrics = {
        **training_info,
        **collect_metrics(evaluator, model),
        "seed": config.seed,
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    return {
        "model": model,
        "domain": domain,
        "evaluator": evaluator,
        "metrics": metrics,
    }


def run_suite(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
    version: str = DEFAULT_VERSION,
) -> dict:
    """Run, persist, plot, and report one named library version."""
    if not version or not version.strip():
        raise ValueError("version must be a non-empty string")

    output_dir = Path(output_dir)
    runs = [
        run_once(replace(config, seed=seed, seeds=[seed]))
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
        "version": version,
        "nu": NU,
    }

    model_path = save_model(representative["model"], output_dir / version)
    artifacts = [model_path]
    artifacts.extend(
        plot_results(
            representative["evaluator"],
            output_dir,
            prefix=f"burgers_{version}",
        )
    )
    result = {
        **representative,
        "version": version,
        "metrics": metrics,
        "artifacts": artifacts,
        "model_path": model_path,
        "runs": runs,
        "representative_run_idx": representative_index,
    }
    report_metrics = {f"{version}_{key}": value for key, value in metrics.items()}
    report = write_markdown_report(
        output_dir / REPORT_NAME,
        f"Burgers version benchmark ({version})",
        config,
        report_metrics,
        artifacts,
    )
    return {
        "variants": {version: result},
        "report": report,
        "artifacts": artifacts,
        "representative_run_idx": representative_index,
    }


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the small CPU-friendly suite.",
    )
    parser.add_argument(
        "--num_runs",
        type=_positive_int,
        help="Number of independent seeds to run.",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        help="Override training with this many Adam epochs.",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"Label for the native model output (default: {DEFAULT_VERSION}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for the native model, plots, and report.",
    )
    return parser.parse_args(argv)


def _config_from_args(args) -> BenchmarkConfig:
    config = SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG
    if args.num_runs is not None:
        config = replace(
            config,
            seeds=[config.seed + index for index in range(args.num_runs)],
        )
    if args.epochs is not None:
        config = replace(config, epochs_adam=args.epochs, epochs_lbfgs=0)
    return config


def main(argv=None):
    args = _parse_args(argv)
    result = run_suite(
        _config_from_args(args),
        args.output_dir,
        version=args.version,
    )
    values = result["variants"][args.version]
    metrics = values["metrics"]
    print(
        f"{args.version}: final total loss={metrics['final_total_loss']:.6e}, "
        f"PDE residual={metrics.get('mean_abs_pde_residual', float('nan')):.6e}"
    )
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
