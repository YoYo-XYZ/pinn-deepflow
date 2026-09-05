"""Shared harness package (beside per-suite clones, no suite modified)."""

from .config import BenchmarkConfig
from .domains import (
    build_burgers_domain,
    build_cavity_domain,
    build_channel_domain,
    build_cylinder_domain,
    perimeter_weighted_boundary_counts,
)
from .reporting import (
    collect_metrics,
    collect_reference_metrics,
    evaluate_area,
    plot_results,
    save_model,
    train_one,
    write_markdown_report,
)
from .precision import (
    PRECISION_DTYPES,
    PrecisionBaseline,
    build_precision_baseline,
    run_precision_suite,
    run_precision_variant,
)

__all__ = [
    "BenchmarkConfig",
    "build_burgers_domain",
    "build_cavity_domain",
    "build_channel_domain",
    "build_cylinder_domain",
    "collect_metrics",
    "collect_reference_metrics",
    "evaluate_area",
    "perimeter_weighted_boundary_counts",
    "plot_results",
    "save_model",
    "train_one",
    "write_markdown_report",
    "PRECISION_DTYPES",
    "PrecisionBaseline",
    "build_precision_baseline",
    "run_precision_suite",
    "run_precision_variant",
]
