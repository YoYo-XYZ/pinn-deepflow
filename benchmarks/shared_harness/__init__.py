"""Shared benchmark configuration, domains, and reporting helpers."""

from .config import BenchmarkConfig
from .domains import (
    build_burgers_domain,
    build_cavity_domain,
    build_channel_domain,
    build_cylinder_domain,
    build_flow_pde,
    perimeter_weighted_boundary_counts,
)
from .reporting import (
    aggregate_metrics,
    collect_metrics,
    collect_reference_metrics,
    evaluate_area,
    evaluate_line,
    load_model,
    plot_results,
    representative_run_index,
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
    "build_flow_pde",
    "aggregate_metrics",
    "collect_metrics",
    "collect_reference_metrics",
    "evaluate_area",
    "evaluate_line",
    "load_model",
    "perimeter_weighted_boundary_counts",
    "plot_results",
    "representative_run_index",
    "save_model",
    "train_one",
    "write_markdown_report",
    "PRECISION_DTYPES",
    "PrecisionBaseline",
    "build_precision_baseline",
    "run_precision_suite",
    "run_precision_variant",
]
