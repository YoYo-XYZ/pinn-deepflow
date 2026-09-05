"""Shared-harness channel-flow initialization benchmark."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import deepflow as df  # noqa: E402
from benchmarks.shared_harness import (  # noqa: E402
    BenchmarkConfig,
    build_channel_domain,
    collect_metrics,
    evaluate_area,
    plot_results,
    save_model,
    train_one,
    write_markdown_report,
)


Lx = 5.0
Ly = 1.0
U = 0.0001
L = 1.0
MU = 0.001
RHO = 1000.0
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_PATH = RESULTS_DIR / "REPORT.md"
INITIALIZATIONS = ("Kaiming-uniform", "Glorot-normal")
WEIGHT_INITS = {
    "Kaiming-uniform": "kaiming",
    "Glorot-normal": "glorot",
}

DEFAULT_CONFIG = BenchmarkConfig(
    width=32,
    depth=4,
    learning_rate=0.004,
    epochs_adam=2000,
    epochs_lbfgs=0,
    seed=69,
    boundary_points=[100, 500, 100, 500],
    interior_points=[2000],
    eval_grid=[200, 40],
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
    """Build one paired channel domain through the shared domain builder."""
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


def build_model(initialization: str, config: BenchmarkConfig = DEFAULT_CONFIG):
    """Construct the PINN variant for one initialization arm."""
    try:
        weight_init = WEIGHT_INITS[initialization]
    except KeyError as exc:
        raise ValueError(f"Unknown initialization: {initialization!r}") from exc
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        width=config.width,
        length=config.depth,
        weight_init=weight_init,
    )


def run_variant(
    initialization: str,
    config: BenchmarkConfig,
    output_dir: Path,
) -> dict:
    """Train, evaluate, persist, and report one initialization arm."""
    domain = build_domain(config)
    model, training_info = train_one(
        domain,
        lambda: build_model(initialization, config),
        config,
    )
    evaluator = evaluate_area(domain, model, list(config.eval_grid))
    metrics = {
        **training_info,
        **collect_metrics(evaluator, model),
        "initialization": initialization,
        "weight_init": WEIGHT_INITS[initialization],
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
        ),
    }
    model_path = save_model(
        model,
        Path(output_dir) / f"channel_{WEIGHT_INITS[initialization]}",
    )
    artifacts = [model_path]
    artifacts.extend(
        plot_results(
            evaluator,
            output_dir,
            prefix=f"channel_{WEIGHT_INITS[initialization]}",
        )
    )
    return {
        "initialization": initialization,
        "model": model,
        "domain": domain,
        "evaluator": evaluator,
        "metrics": metrics,
        "artifacts": artifacts,
    }


def run_suite(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
) -> dict:
    """Run both initialization arms through the shared harness."""
    output_dir = Path(output_dir)
    results = {
        initialization: run_variant(initialization, config, output_dir)
        for initialization in INITIALIZATIONS
    }

    report_metrics = {
        "initializations": ", ".join(INITIALIZATIONS),
        "geometry": f"[{0.0}, {Lx}] x [{0.0}, {Ly}]",
        "U": U,
        "L": L,
        "mu": MU,
        "rho": RHO,
    }
    artifacts = []
    for initialization, result in results.items():
        report_metrics.update(
            {
                f"{WEIGHT_INITS[initialization]}_{key}": value
                for key, value in result["metrics"].items()
            }
        )
        artifacts.extend(result["artifacts"])
    report = write_markdown_report(
        output_dir / REPORT_PATH.name,
        "Channel-flow initialization benchmark",
        config,
        report_metrics,
        artifacts,
    )
    return {"variants": results, "report": report, "artifacts": artifacts}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run the small CPU-friendly suite.",
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
    result = run_suite(config, args.output_dir)
    print("\nChannel-flow initialization comparison")
    for initialization, values in result["variants"].items():
        metrics = values["metrics"]
        print(
            f"{initialization:16s} total loss="
            f"{metrics['final_total_loss']:.6e}, "
            f"PDE residual="
            f"{metrics.get('mean_abs_continuity_residual', float('nan')):.6e}"
        )
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
