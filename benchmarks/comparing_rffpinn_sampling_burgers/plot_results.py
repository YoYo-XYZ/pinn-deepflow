"""Regenerate sampling-comparison plots from native shared-harness models."""

from __future__ import annotations

import argparse
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
    from .benchmark import (  # noqa: E402
        DEFAULT_CONFIG,
        SCHEMES,
        build_domain,
        config_for_scheme,
    )
except ImportError:  # pragma: no cover - exercised by direct execution
    from benchmark import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG,
        SCHEMES,
        build_domain,
        config_for_scheme,
    )


RESULTS_DIR = SCRIPT_DIR / "results"


def main(output_dir: Path = RESULTS_DIR):
    """Evaluate saved models and regenerate visualizer plots."""
    output_dir = Path(output_dir)
    written = []
    for scheme in SCHEMES:
        model = df.load_from_pickle(
            str(output_dir / f"rffpinn_{scheme}")
        )
        config = config_for_scheme(scheme, DEFAULT_CONFIG)
        evaluator = evaluate_area(
            build_domain(scheme, DEFAULT_CONFIG),
            model,
            list(config.eval_grid),
        )
        written.extend(
            plot_evaluation(
                evaluator,
                output_dir,
                prefix=f"rffpinn_{scheme}",
            )
        )
    for path in written:
        print(path)
    return written


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory containing native models and receiving plots.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(_parse_args().output_dir)
