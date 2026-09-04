"""Boundary smoke for shared reporting (native save, evaluator metrics, plots)."""

import matplotlib

matplotlib.use("Agg")

import deepflow as df
from benchmarks.shared_harness.config import BenchmarkConfig
from benchmarks.shared_harness.domains import build_burgers_domain
from benchmarks.shared_harness.reporting import (
    collect_metrics,
    evaluate_area,
    plot_results,
    save_model,
    train_one,
    write_markdown_report,
)


def _tiny_config(**overrides):
    payload = {
        "width": 8,
        "depth": 2,
        "learning_rate": 0.004,
        "epochs_adam": 2,
        "epochs_lbfgs": 0,
        "seed": 69,
        "boundary_points": [8, 4, 4],
        "interior_points": [16],
        "eval_grid": [8, 8],
        "sampling": "lhs",
    }
    payload.update(overrides)
    return BenchmarkConfig(**payload)


def test_reporting_smoke(tmp_path):
    config = _tiny_config()
    domain = build_burgers_domain(
        boundary_points=list(config.boundary_points),
        interior_points=list(config.interior_points),
        sampling=config.sampling,
    )

    def factory():
        return df.PINN(
            input_vars=["x", "y"],
            output_vars=["u"],
            width=config.width,
            length=config.depth,
        )

    best_model, info = train_one(domain, factory, config)
    assert info["final_total_loss"] == info["final_total_loss"]
    assert len(best_model.loss_history["total_loss"]) == config.epochs_adam

    evaluator = evaluate_area(domain, best_model, list(config.eval_grid))
    assert evaluator.data_dict["u"].size > 0

    metrics = collect_metrics(evaluator, best_model)
    assert "history_last_total_loss" in metrics
    assert any(key.startswith("max_") for key in metrics)

    model_path = save_model(best_model, tmp_path / "smoke_model")
    assert model_path.exists()
    reloaded = df.load_from_pickle(str(model_path))
    assert reloaded.loss_history["total_loss"] == best_model.loss_history["total_loss"]

    named_model_path = save_model(best_model, tmp_path / "smoke_model.checkpoint")
    assert named_model_path == tmp_path / "smoke_model.checkpoint.pkl"
    assert named_model_path.exists()

    plots = plot_results(evaluator, tmp_path / "plots", prefix="smoke")
    assert len(plots) == 2
    assert all(path.exists() for path in plots)

    report = write_markdown_report(
        tmp_path / "report.md",
        "Shared harness smoke",
        config,
        {**info, **metrics, "formulation": "burgers"},
        [model_path, *plots],
    )
    text = report.read_text(encoding="utf-8")
    assert "Shared harness smoke" in text
    assert "final_total_loss" in text
    assert "smoke_model" in text
