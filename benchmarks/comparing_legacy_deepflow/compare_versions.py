#!/usr/bin/env python3
"""Compare native Burgers models produced by two DeepFlow versions."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Package execution.
    from .benchmark_burgers import (  # noqa: E402
        DEFAULT_CONFIG,
        DEFAULT_VERSION,
        RESULTS_DIR,
        SMOKE_CONFIG,
        build_domain,
    )
except ImportError:  # Direct script execution.
    from benchmark_burgers import (  # type: ignore  # noqa: E402
        DEFAULT_CONFIG,
        DEFAULT_VERSION,
        RESULTS_DIR,
        SMOKE_CONFIG,
        build_domain,
    )

from benchmarks.shared_harness import (  # noqa: E402
    BenchmarkConfig,
    collect_metrics,
    evaluate_area,
    load_model,
    plot_results,
    save_model,
    write_markdown_report,
)


VERSION_LABELS = ("old", "new")
DEFAULT_MODEL_PATHS = {
    "old": RESULTS_DIR / "old.pkl",
    "new": RESULTS_DIR / f"{DEFAULT_VERSION}.pkl",
}


def _model_path(path: Path, label: str) -> Path:
    path = Path(path)
    if path.suffix != ".pkl":
        path = Path(f"{path}.pkl")
    if not path.is_file():
        raise FileNotFoundError(f"{label} model not found: {path}")
    return path


def evaluate_version(
    label: str,
    model_path: Path,
    config: BenchmarkConfig,
    output_dir: Path,
) -> dict:
    """Load, evaluate, persist, and plot one version through shared helpers."""
    if label not in VERSION_LABELS:
        raise ValueError(f"Unknown version label: {label!r}")
    model_path = _model_path(model_path, label)
    model = load_model(model_path)
    domain = build_domain(config)
    evaluator = evaluate_area(domain, model, list(config.eval_grid))
    metrics = {
        **collect_metrics(evaluator, model),
        "version": label,
        "source_model": str(model_path),
    }
    persisted_path = save_model(model, Path(output_dir) / label)
    artifacts = [persisted_path]
    artifacts.extend(
        plot_results(
            evaluator,
            output_dir,
            prefix=f"burgers_{label}",
        )
    )
    return {
        "version": label,
        "model": model,
        "domain": domain,
        "evaluator": evaluator,
        "metrics": metrics,
        "artifacts": artifacts,
        "model_path": persisted_path,
        "source_model_path": model_path,
    }


def run_comparison(
    old_model_path: Path = DEFAULT_MODEL_PATHS["old"],
    new_model_path: Path = DEFAULT_MODEL_PATHS["new"],
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR / "comparison",
) -> dict:
    """Compare old and new native models using one shared evaluation path."""
    output_dir = Path(output_dir)
    model_paths = {
        "old": old_model_path,
        "new": new_model_path,
    }
    results = {
        label: evaluate_version(label, model_paths[label], config, output_dir)
        for label in VERSION_LABELS
    }
    report_metrics = {}
    artifacts = []
    for label, result in results.items():
        report_metrics.update(
            {f"{label}_{key}": value for key, value in result["metrics"].items()}
        )
        artifacts.extend(result["artifacts"])
    report = write_markdown_report(
        output_dir / "REPORT.md",
        "Burgers version comparison",
        config,
        report_metrics,
        artifacts,
    )
    return {"variants": results, "report": report, "artifacts": artifacts}


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--old",
        type=Path,
        default=DEFAULT_MODEL_PATHS["old"],
        help=f"Old native model (default: {DEFAULT_MODEL_PATHS['old']}).",
    )
    parser.add_argument(
        "--new",
        type=Path,
        default=DEFAULT_MODEL_PATHS["new"],
        help=f"New native model (default: {DEFAULT_MODEL_PATHS['new']}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use the small evaluation grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR / "comparison",
        help="Directory for comparison models, plots, and report.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    config = SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG
    try:
        result = run_comparison(
            args.old,
            args.new,
            config,
            args.output_dir,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    for label, values in result["variants"].items():
        metrics = values["metrics"]
        print(
            f"{label}: final total loss="
            f"{metrics.get('history_last_total_loss', float('nan')):.6e}, "
            f"PDE residual="
            f"{metrics.get('mean_abs_pde_residual', float('nan')):.6e}"
        )
    print(f"Report: {result['report']}")
    return result


if __name__ == "__main__":
    main()
