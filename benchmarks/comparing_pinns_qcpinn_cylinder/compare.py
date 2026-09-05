"""Compare native cylinder models through shared evaluation and reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Package execution.
    from .benchmark import (  # noqa: E402
        DEFAULT_CONFIG,
        FORMULATIONS,
        REPORT_NAME,
        RESULTS_DIR,
        SMOKE_CONFIG,
        VARIANTS,
        available_variants,
        evaluate_profiles,
        build_domain,
    )
    from .reference import load_cached_reference, solve_reference  # noqa: E402
except ImportError:  # Direct script execution.
    from benchmark import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG,
        FORMULATIONS,
        REPORT_NAME,
        RESULTS_DIR,
        SMOKE_CONFIG,
        VARIANTS,
        available_variants,
        evaluate_profiles,
        build_domain,
    )
    from reference import load_cached_reference, solve_reference  # type: ignore  # noqa: E402

from benchmarks.shared_harness import (  # noqa: E402
    collect_metrics,
    collect_reference_metrics,
    evaluate_area,
    load_model,
    plot_results,
    save_model,
    write_markdown_report,
)


DEFAULT_MODEL_PATHS = {
    variant: RESULTS_DIR / f"cylinder_{variant.lower().replace('-', '_')}.pkl"
    for variant in VARIANTS
}


def _model_path(path: Path, label: str) -> Path:
    path = Path(path)
    if path.suffix != ".pkl":
        path = Path(f"{path}.pkl")
    if not path.is_file():
        raise FileNotFoundError(f"{label} model not found: {path}")
    return path


def _profile_reference_metrics(profiles, reference_solution):
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


def evaluate_variant(
    variant: str,
    model_path: Path,
    config=DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR / "comparison",
    reference_solution=None,
):
    """Load and evaluate one native model through the shared path."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown cylinder benchmark variant: {variant!r}")
    model_path = _model_path(model_path, variant)
    model = load_model(model_path)
    domain = build_domain(FORMULATIONS[variant], config)
    evaluator = evaluate_area(domain, model, list(config.eval_grid))
    profiles = evaluate_profiles(model, FORMULATIONS[variant])
    metrics = {
        **collect_metrics(evaluator, model, reference_solution=reference_solution),
        "variant": variant,
        "formulation": FORMULATIONS[variant],
        "source_model": str(model_path),
    }
    if reference_solution is not None:
        metrics.update(_profile_reference_metrics(profiles, reference_solution))
    output_dir = Path(output_dir)
    persisted_path = save_model(model, output_dir / model_path.stem)
    artifacts = [persisted_path]
    artifacts.extend(
        plot_results(
            evaluator,
            output_dir,
            prefix=f"comparison_{variant.lower().replace('-', '_')}",
        )
    )
    return {
        "variant": variant,
        "model": model,
        "domain": domain,
        "evaluator": evaluator,
        "profiles": profiles,
        "metrics": metrics,
        "artifacts": artifacts,
        "model_path": persisted_path,
        "source_model_path": model_path,
    }


def run_comparison(
    model_paths=None,
    config=DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR / "comparison",
    variants=None,
    reference_solution=None,
    offline_reference_path: Path | None = None,
):
    """Compare selected native models and write a shared-harness report."""
    if reference_solution is not None and offline_reference_path is not None:
        raise ValueError("Choose a FEM reference or an offline reference, not both.")
    if offline_reference_path is not None:
        reference_solution = load_cached_reference(offline_reference_path)
    variants = available_variants() if variants is None else tuple(variants)
    if not variants:
        raise ValueError("At least one cylinder model is required for comparison.")
    model_paths = DEFAULT_MODEL_PATHS if model_paths is None else model_paths
    output_dir = Path(output_dir)
    results = {
        variant: evaluate_variant(
            variant,
            model_paths[variant],
            config,
            output_dir,
            reference_solution=reference_solution,
        )
        for variant in variants
    }
    report_metrics = {
        "variants": ", ".join(variants),
        "reference": (
            "offline cache"
            if offline_reference_path is not None
            else "FEM"
            if reference_solution is not None
            else "not requested"
        ),
    }
    artifacts = []
    for variant, result in results.items():
        report_metrics.update(
            {
                f"{variant.lower().replace('-', '_')}_{key}": value
                for key, value in result["metrics"].items()
            }
        )
        artifacts.extend(result["artifacts"])
    report = write_markdown_report(
        output_dir / REPORT_NAME,
        "Cylinder model comparison",
        config,
        report_metrics,
        artifacts,
    )
    return {"variants": results, "report": report, "artifacts": artifacts}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=VARIANTS,
        help="Variant to compare; may be repeated.",
    )
    parser.add_argument(
        "--offline-reference",
        type=Path,
        help="Explicitly use a cached reference archive instead of solving FEM.",
    )
    parser.add_argument(
        "--no-reference",
        action="store_true",
        help="Skip the FEM reference comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "comparison",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.offline_reference is not None and args.no_reference:
        raise SystemExit("--offline-reference and --no-reference are mutually exclusive")
    config = SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG
    reference = None
    if not args.no_reference and args.offline_reference is None:
        reference = solve_reference(config)
    result = run_comparison(
        config=config,
        output_dir=args.output_dir,
        variants=args.variants,
        reference_solution=reference,
        offline_reference_path=args.offline_reference,
    )
    for variant, values in result["variants"].items():
        metrics = values["metrics"]
        print(
            f"{variant}: total loss="
            f"{metrics.get('history_last_total_loss', float('nan')):.6e}, "
            f"PDE residual="
            f"{metrics.get('mean_abs_pde_residual', float('nan')):.6e}"
        )
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
