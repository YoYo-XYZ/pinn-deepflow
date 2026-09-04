"""Shared reporting: train-one, evaluate, persist, plot, and report.

Reporting rules for the rework:
- Persist via the library native save (``save_as_pickle``).
- Derive metrics from evaluation results (``data_dict``) and loss histories.
- Produce plots via visualizer calls (``plot_color``/``plot_loss_curve``).
- Write a markdown report.
"""

from __future__ import annotations

import sys
import time
from numbers import Real
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import ultraplot as plt  # noqa: E402

import deepflow as df  # noqa: E402

from .config import BenchmarkConfig  # noqa: E402


def train_one(domain, model_factory: Callable, config: BenchmarkConfig):
    """Train one model with Adam then L-BFGS via the simple-loss helper.

    Args:
        domain: Sampled problem domain.
        model_factory: Zero-arg callable returning a fresh model.
        config: Shared training knobs (seed, lr, epochs).

    Returns:
        Tuple ``(best_model, info)`` where info holds timing and final
        losses from the loss call.
    """
    df.manual_seed(config.seed)
    model = model_factory()
    calc_loss = df.calc_loss_simple(domain)

    if config.epochs_adam > 0:
        start = time.perf_counter()
        _, adam_best = model.train_adam(
            calc_loss=calc_loss,
            learning_rate=config.learning_rate,
            epochs=config.epochs_adam,
            print_every=max(1, config.epochs_adam // 10),
        )
        adam_time = time.perf_counter() - start
    else:
        adam_best = model
        adam_time = 0.0

    if config.epochs_lbfgs > 0:
        start = time.perf_counter()
        _, best_model = adam_best.train_lbfgs(
            calc_loss=calc_loss,
            epochs=config.epochs_lbfgs,
            print_every=max(1, config.epochs_lbfgs // 10),
        )
        lbfgs_time = time.perf_counter() - start
    else:
        best_model = adam_best
        lbfgs_time = 0.0

    final = calc_loss(best_model)
    info = {
        "adam_time_s": float(adam_time),
        "lbfgs_time_s": float(lbfgs_time),
        "total_time_s": float(adam_time + lbfgs_time),
        "final_total_loss": float(final["total_loss"].detach().cpu().item()),
        "final_bc_loss": float(final["bc_loss"].detach().cpu().item()),
        "final_pde_loss": float(final["pde_loss"].detach().cpu().item()),
    }
    return best_model, info


def evaluate_area(domain, model, eval_grid: List[int]):
    """Evaluate the PDE area on a fresh grid via the evaluator API."""
    area_eval = domain.area_list[0].evaluate(model)
    area_eval.sampling_area(list(eval_grid))
    return area_eval


def _last(history_values) -> float:
    values = np.asarray(history_values, dtype=np.float64).reshape(-1)
    return float(values[-1]) if values.size else float("nan")


def collect_metrics(evaluator, model) -> Dict[str, float]:
    """Derive metrics from evaluation results and loss histories.

    Uses ``evaluator.data_dict`` residual fields and
    ``model.loss_history`` tails; no hand-rolled field L2 against cached
    arrays.
    """
    data = evaluator.data_dict
    metrics: Dict[str, float] = {}
    for key in ("total_loss", "bc_loss", "pde_loss", "ic_loss"):
        if key in model.loss_history:
            metrics[f"history_last_{key}"] = _last(model.loss_history[key])
    for key, values in data.items():
        if not key.endswith("_residual"):
            continue
        field = np.asarray(values, dtype=np.float64).reshape(-1)
        if field.size == 0:
            continue
        metrics[f"max_{key}"] = float(np.max(np.abs(field)))
        metrics[f"mean_abs_{key}"] = float(np.mean(np.abs(field)))
    return metrics


def save_model(model, path: Path) -> Path:
    """Persist a model via the library native pickle-style save."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_as_pickle(str(path))
    resolved = path if path.name.endswith(".pkl") else Path(f"{path}.pkl")
    return resolved


def _first_field(data: dict) -> str | None:
    for key in ("u", "v", "p", "psi"):
        if key in data:
            return key
    for key in data:
        if key not in ("x", "y", "t") and not key.endswith(
            ("_residual", "_loss", "loss")
        ):
            values = np.asarray(data[key]).reshape(-1)
            if values.size and np.issubdtype(values.dtype, np.number):
                return key
    return None


def plot_results(evaluator, out_dir: Path, prefix: str = "harness") -> List[Path]:
    """Produce plots via visualizer calls and save them.

    Returns:
        List of written image paths (field plot + loss curve when present).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    field = _first_field(evaluator.data_dict)
    if field is not None:
        fig = evaluator.plot_color(field)
        field_path = out_dir / f"{prefix}_{field}_field.png"
        fig.savefig(field_path)
        plt.close(fig)
        written.append(field_path)
    loss_values = np.asarray(evaluator.data_dict.get("total_loss", [])).reshape(-1)
    if loss_values.size:
        fig = evaluator.plot_loss_curve()
        loss_path = out_dir / f"{prefix}_loss_curve.png"
        fig.savefig(loss_path)
        plt.close(fig)
        written.append(loss_path)
    return written


def write_markdown_report(
    path: Path,
    title: str,
    config: BenchmarkConfig,
    metrics: Mapping[str, Any],
    artifacts: Sequence[Path],
) -> Path:
    """Write a markdown report listing config, metrics, and artifacts."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Config", ""]
    for key, value in config.to_dict().items():
        lines.append(f"- {key}: `{value}`")
    lines += ["", "## Metrics", ""]
    if metrics:
        for key in sorted(metrics):
            value = metrics[key]
            if isinstance(value, Real) and not isinstance(value, bool):
                rendered = f"{float(value):.6e}"
            else:
                rendered = str(value)
            lines.append(f"- {key}: `{rendered}`")
    else:
        lines.append("- _no metrics_")
    lines += ["", "## Artifacts", ""]
    if artifacts:
        for artifact in artifacts:
            lines.append(f"- `{Path(artifact).name}`")
    else:
        lines.append("- _no artifacts_")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
