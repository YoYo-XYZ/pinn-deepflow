"""Shared FP32/FP64 benchmark workflow.

The precision suites provide only a domain factory and a model factory. This
module owns paired baseline construction, training, evaluation, aggregation,
native model persistence, plots, and reports.
"""

from __future__ import annotations

import copy
import sys
import time
from dataclasses import dataclass, replace
from numbers import Real
from pathlib import Path
from statistics import fmean, stdev
from typing import Callable, Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch

import deepflow as df

from .config import BenchmarkConfig
from .reporting import (
    collect_metrics,
    evaluate_area,
    plot_results,
    save_model,
    train_one,
    write_markdown_report,
)


PRECISION_DTYPES = (torch.float32, torch.float64)
_PRECISION_LABELS = {
    torch.float32: "FP32",
    torch.float64: "FP64",
}


@dataclass(frozen=True)
class PrecisionBaseline:
    """FP32 sampled coordinates and model used to pair precision runs."""

    coordinates: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    evaluation_coordinates: tuple[torch.Tensor, torch.Tensor]
    model: torch.nn.Module


def _precision_label(dtype: torch.dtype) -> str:
    try:
        return _PRECISION_LABELS[dtype]
    except KeyError as exc:
        raise ValueError(
            f"dtype must be torch.float32 or torch.float64, got {dtype!r}"
        ) from exc


def _unique_geometries(domain) -> list:
    geometries = []
    seen = set()
    for geometry in (*domain.bound_list, *domain.area_list):
        if id(geometry) not in seen:
            seen.add(id(geometry))
            geometries.append(geometry)
    return geometries


def _capture_coordinates(domain):
    coordinates = []
    for geometry in _unique_geometries(domain):
        if geometry.X is None or geometry.Y is None:
            raise ValueError("Every baseline geometry must have sampled coordinates.")
        coordinates.append(
            (
                geometry.X.detach().cpu().clone(),
                geometry.Y.detach().cpu().clone(),
            )
        )
    return tuple(coordinates)


def _apply_coordinates(domain, coordinates, dtype: torch.dtype) -> None:
    geometries = _unique_geometries(domain)
    if len(geometries) != len(coordinates):
        raise ValueError("Baseline and target domains have different geometry layouts.")

    for geometry, (x, y) in zip(geometries, coordinates):
        # FLEX: paired precision runs must cast the shared sampled coordinates.
        geometry.set_coordinates(x.to(dtype=dtype).clone(), y.to(dtype=dtype).clone())
    for geometry in geometries:
        geometry.process_coordinates()


def _capture_evaluation_coordinates(domain, eval_grid: Sequence[int]):
    area = domain.area_list[0]
    area.sampling_area(list(eval_grid))
    return area.X.detach().cpu().clone(), area.Y.detach().cpu().clone()


def _apply_evaluation_coordinates(evaluator, coordinates, dtype: torch.dtype) -> None:
    x, y = coordinates
    # FLEX: the canonical FP32 evaluation grid is cast for the FP64 run.
    evaluator.geometry.set_coordinates(
        x.to(dtype=dtype).clone(),
        y.to(dtype=dtype).clone(),
    )
    evaluator.postprocess()


def _set_precision(dtype: torch.dtype) -> None:
    _precision_label(dtype)
    # FLEX: precision is the controlled model-variant difference.
    df.set_dtype(dtype)


def _synchronize_device() -> None:
    # FLEX: CUDA work is asynchronous, so timing needs explicit synchronization.
    if torch.cuda.is_available() and str(df.device).startswith("cuda"):
        torch.cuda.synchronize()


def build_precision_baseline(
    config: BenchmarkConfig,
    domain_factory: Callable[[BenchmarkConfig], object],
    model_factory: Callable[[BenchmarkConfig], torch.nn.Module],
    *,
    seed: int | None = None,
) -> PrecisionBaseline:
    """Build one FP32 domain/model baseline for a paired seed."""
    seed = config.seed if seed is None else seed
    _set_precision(torch.float32)
    df.manual_seed(seed, deterministic=True)
    domain = domain_factory(config)
    model = model_factory(config)
    coordinates = _capture_coordinates(domain)
    evaluation_coordinates = _capture_evaluation_coordinates(
        domain, config.eval_grid
    )
    return PrecisionBaseline(
        coordinates=coordinates,
        evaluation_coordinates=evaluation_coordinates,
        model=copy.deepcopy(model),
    )


def run_precision_variant(
    dtype: torch.dtype,
    config: BenchmarkConfig,
    baseline: PrecisionBaseline,
    domain_factory: Callable[[BenchmarkConfig], object],
) -> dict:
    """Run one precision from a shared baseline through the shared harness."""
    label = _precision_label(dtype)
    _set_precision(dtype)
    df.manual_seed(config.seed, deterministic=True)
    domain = domain_factory(config)
    _apply_coordinates(domain, baseline.coordinates, dtype)

    # FLEX: a copied FP32 baseline is cast to the requested precision.
    model = copy.deepcopy(baseline.model).to(dtype=dtype)

    _synchronize_device()
    start = time.perf_counter()
    best_model, training_info = train_one(
        domain,
        lambda: model,
        config,
    )
    _synchronize_device()
    train_time = time.perf_counter() - start

    evaluator = evaluate_area(domain, best_model, list(config.eval_grid))
    _apply_evaluation_coordinates(
        evaluator,
        baseline.evaluation_coordinates,
        dtype,
    )
    metrics = {
        **training_info,
        "train_time_s": float(train_time),
        "total_time_s": float(train_time),
        **collect_metrics(evaluator, best_model),
        "trainable_parameters": sum(
            parameter.numel() for parameter in best_model.parameters()
        ),
    }
    return {
        "precision": label,
        "dtype": str(dtype),
        "seed": int(config.seed),
        "model": best_model,
        "domain": domain,
        "evaluator": evaluator,
        "metrics": metrics,
        "artifacts": [],
    }


def _aggregate_metrics(runs: Sequence[dict]) -> Dict[str, float]:
    if not runs:
        raise ValueError("At least one precision run is required.")
    names = sorted(
        {
            key
            for run in runs
            for key, value in run["metrics"].items()
            if isinstance(value, Real) and not isinstance(value, bool)
        }
    )
    aggregate = {}
    for name in names:
        values = [float(run["metrics"][name]) for run in runs]
        aggregate[f"{name}_mean"] = float(fmean(values))
        aggregate[f"{name}_std"] = float(stdev(values)) if len(values) > 1 else 0.0
    return aggregate


def _representative_run_index(paired_runs: Sequence[dict], labels: Sequence[str]) -> int:
    scores = []
    for index, pair in enumerate(paired_runs):
        losses = [pair[label]["metrics"]["final_total_loss"] for label in labels]
        scores.append((float(fmean(losses)), index))
    return sorted(scores)[len(scores) // 2][1]


def run_precision_suite(
    config: BenchmarkConfig,
    output_dir: Path,
    domain_factory: Callable[[BenchmarkConfig], object],
    model_factory: Callable[[BenchmarkConfig], torch.nn.Module],
    *,
    title: str,
    prefix: str,
    dtypes: Sequence[torch.dtype] = PRECISION_DTYPES,
) -> dict:
    """Run paired precision variants and produce native shared-harness output."""
    dtypes = tuple(dtypes)
    if not dtypes:
        raise ValueError("At least one precision dtype is required.")
    labels = tuple(_precision_label(dtype) for dtype in dtypes)
    if len(set(labels)) != len(labels):
        raise ValueError("Precision dtypes must be unique.")
    if not config.seeds:
        raise ValueError("BenchmarkConfig.seeds must contain at least one seed.")

    previous_dtype = df.dtype
    paired_runs = []
    try:
        for seed in config.seeds:
            run_config = replace(config, seed=seed, seeds=[seed])
            baseline = build_precision_baseline(
                run_config,
                domain_factory,
                model_factory,
                seed=seed,
            )
            paired_runs.append(
                {
                    label: run_precision_variant(
                        dtype,
                        run_config,
                        baseline,
                        domain_factory,
                    )
                    for dtype, label in zip(dtypes, labels)
                }
            )

        representative_index = _representative_run_index(paired_runs, labels)
        variants = {}
        artifacts = []
        report_metrics = {
            "precisions": ", ".join(labels),
            "num_runs": len(config.seeds),
            "representative_run_idx": representative_index,
            "representative_seed": config.seeds[representative_index],
        }
        output_dir = Path(output_dir)
        for label in labels:
            runs = [pair[label] for pair in paired_runs]
            representative = runs[representative_index]
            metrics = {
                **representative["metrics"],
                **_aggregate_metrics(runs),
                "num_runs": len(runs),
                "representative_run_idx": representative_index,
                "representative_seed": representative["seed"],
            }
            variant_prefix = f"{prefix}_{label.lower()}"
            model_path = save_model(
                representative["model"],
                output_dir / variant_prefix,
            )
            variant_artifacts = [model_path]
            variant_artifacts.extend(
                plot_results(
                    representative["evaluator"],
                    output_dir,
                    prefix=variant_prefix,
                )
            )
            result = {
                **representative,
                "metrics": metrics,
                "artifacts": variant_artifacts,
                "runs": runs,
                "representative_run_idx": representative_index,
            }
            variants[label] = result
            artifacts.extend(variant_artifacts)
            report_metrics.update(
                {f"{label.lower()}_{key}": value for key, value in metrics.items()}
            )

        report = write_markdown_report(
            output_dir / "REPORT.md",
            title,
            config,
            report_metrics,
            artifacts,
        )
        return {
            "variants": variants,
            "report": report,
            "artifacts": artifacts,
            "representative_run_idx": representative_index,
        }
    finally:
        _set_precision(previous_dtype)
