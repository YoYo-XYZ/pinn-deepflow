"""Run the shared-harness smoke path for the Burgers model comparison."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Direct script execution has no package context.
    from .benchmark import SMOKE_CONFIG, run_suite  # noqa: E402
except ImportError:  # pragma: no cover - exercised by direct execution
    from benchmark import SMOKE_CONFIG, run_suite  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="deepflow-burgers-smoke-") as directory:
        result = run_suite(SMOKE_CONFIG, Path(directory))
        assert set(result["variants"]) == {"PINN", "RFFPINN"}
        assert result["report"].exists()
        for variant, values in result["variants"].items():
            assert values["evaluator"].data_dict["u"].size > 0
            total_loss = values["metrics"]["history_last_total_loss"]
            assert total_loss == total_loss
            assert all(path.exists() for path in values["artifacts"])
            print(f"{variant}: shared-harness smoke passed")


if __name__ == "__main__":
    main()
