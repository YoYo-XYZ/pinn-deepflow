#!/usr/bin/env python3
"""Shared-harness DeepFlow counterpart for the channel-flow benchmark."""

from __future__ import annotations

import argparse
import sys
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
    build_channel_domain,
    collect_metrics,
    evaluate_area,
    perimeter_weighted_boundary_counts,
    plot_results,
    save_model,
    train_one,
    write_markdown_report,
)

try:  # Package execution.
    from .benchmark_common import RESULTS_DIR  # noqa: E402
    from .common_config import (  # noqa: E402
        BOUNDARY_POINTS,
        DEPTH,
        EPOCHS,
        EVAL_GRID,
        INTERIOR_POINTS,
        Lx,
        Ly,
        LR,
        Re,
        SEED,
        WIDTH,
    )
except ImportError:  # Direct script execution.
    from benchmark_common import RESULTS_DIR  # type: ignore  # noqa: E402
    from common_config import (  # type: ignore  # noqa: E402
        BOUNDARY_POINTS,
        DEPTH,
        EPOCHS,
        EVAL_GRID,
        INTERIOR_POINTS,
        Lx,
        Ly,
        LR,
        Re,
        SEED,
        WIDTH,
    )


U = 0.0001
L = 1.0
MU = 0.001
RHO = 1000.0
REPORT_NAME = "REPORT.md"

DEFAULT_CONFIG = BenchmarkConfig(
    width=WIDTH,
    depth=DEPTH,
    learning_rate=LR,
    epochs_adam=EPOCHS,
    epochs_lbfgs=0,
    seed=SEED,
    boundary_points=perimeter_weighted_boundary_counts(sum(BOUNDARY_POINTS), Lx, Ly),
    interior_points=[INTERIOR_POINTS],
    eval_grid=list(EVAL_GRID),
    sampling="random",
)

SMOKE_CONFIG = BenchmarkConfig(
    width=8,
    depth=2,
    learning_rate=LR,
    epochs_adam=2,
    epochs_lbfgs=0,
    seed=SEED,
    boundary_points=[4, 4, 4, 4],
    interior_points=[16],
    eval_grid=[8, 4],
    sampling="random",
)


def build_domain(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build the channel domain through the shared DeepFlow builder."""
    df.manual_seed(config.seed)
    return build_channel_domain(
        lx=Lx,
        ly=Ly,
        boundary_points=list(config.boundary_points),
        interior_points=list(config.interior_points),
        sampling=config.sampling,
        U=U,
        L=L,
        mu=MU,
        rho=RHO,
    )


def build_model(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build the standard DeepFlow model for the channel benchmark."""
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        width=config.width,
        length=config.depth,
    )


def run_variant(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """Train, evaluate, persist, and plot one DeepFlow model."""
    output_dir = Path(output_dir)
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
        "reynolds": Re,
    }
    model_path = save_model(model, output_dir / "deepflow")
    artifacts = [model_path]
    artifacts.extend(plot_results(evaluator, output_dir, prefix="deepflow"))
    return {
        "variant": "DeepFlow",
        "model": model,
        "domain": domain,
        "evaluator": evaluator,
        "metrics": metrics,
        "artifacts": artifacts,
        "model_path": model_path,
    }


def run_suite(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """Run the DeepFlow counterpart through the shared benchmark harness."""
    output_dir = Path(output_dir)
    result = run_variant(config, output_dir)
    report_metrics = {
        "method": "DeepFlow",
        "geometry": f"rectangle [0, {Lx}] x [0, {Ly}]",
        "U": U,
        "L": L,
        "mu": MU,
        "rho": RHO,
        **result["metrics"],
    }
    report = write_markdown_report(
        output_dir / REPORT_NAME,
        "DeepFlow channel-flow benchmark",
        config,
        report_metrics,
        result["artifacts"],
    )
    return {
        "variants": {"DeepFlow": result},
        "report": report,
        "artifacts": result["artifacts"],
    }


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the small CPU-friendly shared-harness configuration.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for the native model, plots, and report.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    result = run_suite(SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG, args.output_dir)
    metrics = result["variants"]["DeepFlow"]["metrics"]
    print(
        f"DeepFlow: final total loss={metrics['final_total_loss']:.6e}, "
        f"PDE residual={metrics.get('mean_abs_pde_residual', float('nan')):.6e}"
    )
    print(f"Model: {result['variants']['DeepFlow']['model_path']}")
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
