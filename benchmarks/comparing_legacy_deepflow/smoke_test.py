"""Run the shared-harness smoke path for the version comparison."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Direct script execution has no package context.
    from .benchmark_burgers import DEFAULT_VERSION, SMOKE_CONFIG, run_suite
    from .compare_versions import run_comparison
except ImportError:  # pragma: no cover - exercised by direct execution
    from benchmark_burgers import DEFAULT_VERSION, SMOKE_CONFIG, run_suite
    from compare_versions import run_comparison


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="deepflow-legacy-burgers-smoke-") as directory:
        output_dir = Path(directory)
        result = run_suite(SMOKE_CONFIG, output_dir, version=DEFAULT_VERSION)
        values = result["variants"][DEFAULT_VERSION]
        assert result["report"].exists()
        assert values["evaluator"].data_dict["u"].size > 0
        assert "history_last_total_loss" in values["metrics"]
        assert all(path.exists() for path in values["artifacts"])

        comparison = run_comparison(
            values["model_path"],
            values["model_path"],
            SMOKE_CONFIG,
            output_dir / "comparison",
        )
        assert set(comparison["variants"]) == {"old", "new"}
        assert comparison["report"].exists()
        assert all(
            path.exists()
            for values in comparison["variants"].values()
            for path in values["artifacts"]
        )
        print("Legacy-version Burgers shared-harness smoke passed")


if __name__ == "__main__":
    main()
