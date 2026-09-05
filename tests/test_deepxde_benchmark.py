"""Public smoke seams for the DeepFlow/DeepXDE benchmark."""

import numpy as np

from benchmarks.comparing_deepxde.benchmark_deepflow import (
    SMOKE_CONFIG,
    run_suite,
)
from benchmarks.comparing_deepxde.compare import run_comparison


def test_deepflow_counterpart_runs_shared_smoke_path(tmp_path):
    result = run_suite(SMOKE_CONFIG, tmp_path)

    values = result["variants"]["DeepFlow"]
    assert result["report"].exists()
    assert values["model_path"].suffix == ".pkl"
    assert values["evaluator"].data_dict["u"].size > 0
    assert np.isfinite(values["metrics"]["final_total_loss"])
    assert all(path.exists() for path in values["artifacts"])


def test_comparison_reads_native_deepflow_model_and_evaluator_metrics(tmp_path):
    suite = run_suite(SMOKE_CONFIG, tmp_path / "suite")
    deepflow_values = suite["variants"]["DeepFlow"]
    data = deepflow_values["evaluator"].data_dict
    deepxde_results = tmp_path / "deepxde_results.npz"
    # FLEX: this fixture represents the raw archive written by the untouched
    # external competitor, so the comparison reader can be tested offline.
    np.savez(
        deepxde_results,
        x=data["x"],
        y=data["y"],
        u=data["u"],
        v=data["v"],
        p=data["p"],
        continuity_residual=data["continuity_residual"],
        x_momentum_residual=data["x_momentum_residual"],
        y_momentum_residual=data["y_momentum_residual"],
        loss_train=np.array([1.0]),
        loss_test=np.array([1.0]),
        loss_steps=np.array([1]),
        final_total_loss=np.array(1.0),
        best_loss_train=np.array(1.0),
        best_loss_test=np.array(1.0),
        train_time_s=np.array(0.0),
    )

    comparison = run_comparison(
        deepflow_values["model_path"],
        deepxde_results,
        SMOKE_CONFIG,
        tmp_path / "comparison",
    )

    native = comparison["variants"]["DeepFlow"]
    assert comparison["report"].exists()
    assert native["evaluator"].data_dict["u"].size > 0
    assert "history_last_total_loss" in native["metrics"]
    assert np.isfinite(comparison["metrics"]["DeepFlow"]["max_pde_residual"])
