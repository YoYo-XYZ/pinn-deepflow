"""Shared-harness FP32/FP64 precision benchmark for channel flow."""

from __future__ import annotations

import argparse
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
    PRECISION_DTYPES,
    build_channel_domain,
    run_precision_suite,
)


LX = 5.0
LY = 1.0
U = 0.0001
L = 1.0
MU = 0.001
RHO = 1000.0
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "REPORT.md"
PRECISIONS = ("FP32", "FP64")

DEFAULT_CONFIG = BenchmarkConfig(
    width=32,
    depth=4,
    learning_rate=0.004,
    epochs_adam=0,
    epochs_lbfgs=200,
    seed=69,
    boundary_points=[100, 500, 100, 500],
    interior_points=[2000],
    eval_grid=[500, 100],
    sampling="random",
)

SMOKE_CONFIG = BenchmarkConfig(
    width=8,
    depth=2,
    learning_rate=0.004,
    epochs_adam=2,
    epochs_lbfgs=0,
    seed=69,
    boundary_points=[4, 4, 4, 4],
    interior_points=[16],
    eval_grid=[8, 4],
    sampling="random",
)


def build_domain(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build the channel domain through the shared Navier-Stokes builder."""
    return build_channel_domain(
        lx=LX,
        ly=LY,
        boundary_points=list(config.boundary_points),
        interior_points=list(config.interior_points),
        sampling=config.sampling,
        U=U,
        L=L,
        mu=MU,
        rho=RHO,
    )


def build_model(config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build the one channel-flow PINN used by both precision variants."""
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        width=config.width,
        length=config.depth,
    )


def run_suite(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """Run both precision variants through the shared precision harness."""
    return run_precision_suite(
        config,
        output_dir,
        build_domain,
        build_model,
        title="Channel-flow precision benchmark",
        prefix="channel_precision",
        dtypes=PRECISION_DTYPES,
    )


def _positive_int(value):
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
        help="Number of paired seeds to run.",
    )
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        help="Override training with this many L-BFGS epochs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory for native models, plots, and the report.",
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
        config = replace(config, epochs_adam=0, epochs_lbfgs=args.epochs)
    return config


def main(argv=None):
    args = _parse_args(argv)
    result = run_suite(_config_from_args(args), args.output_dir)
    print("\nChannel-flow precision comparison")
    for precision, values in result["variants"].items():
        metrics = values["metrics"]
        print(
            f"{precision}: final total loss={metrics['final_total_loss']:.6e}, "
            f"PDE residual={metrics.get('mean_abs_pde_residual', float('nan')):.6e}"
        )
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
