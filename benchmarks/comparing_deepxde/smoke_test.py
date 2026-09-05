"""Run the shared-harness smoke path for the DeepFlow/DeepXDE comparison."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Package execution.
    from .benchmark_deepflow import SMOKE_CONFIG, run_suite  # noqa: E402
    from .compare import run_comparison  # noqa: E402
except ImportError:  # Direct script execution.
    from benchmark_deepflow import SMOKE_CONFIG, run_suite  # type: ignore  # noqa: E402
    from compare import run_comparison  # type: ignore  # noqa: E402


def _write_deepxde_smoke_archive(path: Path, data: dict) -> None:
    """Create a tiny raw-reader fixture without importing DeepXDE."""
    # FLEX: the untouched competitor writes this archive format; the smoke
    # path needs a tiny fixture without requiring the optional competitor.
    np.savez(
        path,
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
        train_time_s=np.array(0.0),
        final_total_loss=np.array(1.0),
        best_loss_train=np.array(1.0),
        best_loss_test=np.array(1.0),
    )


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="deepflow-deepxde-smoke-") as directory:
        root = Path(directory)
        suite = run_suite(SMOKE_CONFIG, root / "suite")
        values = suite["variants"]["DeepFlow"]
        assert suite["report"].exists()
        assert values["evaluator"].data_dict["u"].size > 0
        assert np.isfinite(values["metrics"]["final_total_loss"])
        assert all(path.exists() for path in values["artifacts"])

        deepxde_results = root / "deepxde_results.npz"
        _write_deepxde_smoke_archive(
            deepxde_results,
            values["evaluator"].data_dict,
        )
        comparison = run_comparison(
            values["model_path"],
            deepxde_results,
            SMOKE_CONFIG,
            root / "comparison",
        )
        assert comparison["report"].exists()
        assert "history_last_total_loss" in comparison["metrics"]["DeepFlow"]
        assert all(path.exists() for path in comparison["artifacts"])
        print("DeepFlow/DeepXDE shared-harness smoke passed")


if __name__ == "__main__":
    main()
