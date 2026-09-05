"""Regenerate Burgers plots from native shared-harness model files."""

from __future__ import annotations

import sys
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
    evaluate_area,
    plot_results as plot_evaluation,
)

try:  # Direct script execution has no package context.
    from .benchmark import DEFAULT_CONFIG, VARIANTS, build_domain  # noqa: E402
except ImportError:  # pragma: no cover - exercised by direct execution
    from benchmark import DEFAULT_CONFIG, VARIANTS, build_domain  # noqa: E402


RESULTS_DIR = SCRIPT_DIR / "results"


def main(output_dir: Path = RESULTS_DIR):
    """Evaluate saved models and regenerate visualizer plots."""
    output_dir = Path(output_dir)
    written = []
    for variant in VARIANTS:
        model = df.load_from_pickle(str(output_dir / variant.lower()))
        evaluator = evaluate_area(
            build_domain(DEFAULT_CONFIG),
            model,
            list(DEFAULT_CONFIG.eval_grid),
        )
        written.extend(
            plot_evaluation(evaluator, output_dir, prefix=variant.lower())
        )
    for path in written:
        print(path)
    return written


if __name__ == "__main__":
    main()
