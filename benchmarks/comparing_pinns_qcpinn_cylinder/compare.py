"""Compare native cylinder models through shared evaluation and reporting."""

from __future__ import annotations

import argparse
from pathlib import Path

try:  # Package execution.
    from .benchmark import (  # noqa: E402
        DEFAULT_CONFIG,
        HARNESS,
        RESULTS_DIR,
        SMOKE_CONFIG,
        VARIANTS,
    )
    from .reference import load_cached_reference, solve_reference  # noqa: E402
except ImportError:  # Direct script execution.
    from benchmark import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG,
        HARNESS,
        RESULTS_DIR,
        SMOKE_CONFIG,
        VARIANTS,
    )
    from reference import load_cached_reference, solve_reference  # type: ignore  # noqa: E402

DEFAULT_MODEL_PATHS = {
    variant: RESULTS_DIR / f"cylinder_{variant.lower().replace('-', '_')}.pkl"
    for variant in VARIANTS
}
_AUTO_REFERENCE = object()

def evaluate_variant(
    variant: str,
    model_path: Path,
    config=DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR / "comparison",
    reference_solution=None,
):
    """Load and evaluate one native model through the shared path."""
    return HARNESS.compare_variant(
        variant,
        model_path,
        config=config,
        output_dir=output_dir,
        reference_solution=reference_solution,
    )


def run_comparison(
    model_paths=None,
    config=DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR / "comparison",
    variants=None,
    reference_solution=_AUTO_REFERENCE,
    offline_reference_path: Path | None = None,
):
    """Compare models, solving a fresh FEM reference unless explicitly skipped."""
    if (
        reference_solution is not _AUTO_REFERENCE
        and reference_solution is not None
        and offline_reference_path is not None
    ):
        raise ValueError("Choose a FEM reference or an offline reference, not both.")
    if offline_reference_path is not None:
        reference_solution = load_cached_reference(offline_reference_path)
    elif reference_solution is _AUTO_REFERENCE:
        reference_solution = solve_reference(config)
    model_paths = DEFAULT_MODEL_PATHS if model_paths is None else model_paths
    return HARNESS.compare_suite(
        model_paths,
        config=config,
        output_dir=output_dir,
        variants=variants,
        reference_solution=reference_solution,
        reference_label=(
            "offline cache"
            if offline_reference_path is not None
            else "FEM"
            if reference_solution is not None
            else None
        ),
    )


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
