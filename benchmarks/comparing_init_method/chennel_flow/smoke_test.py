"""Run the shared-harness smoke path for channel initializations."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # Direct script execution has no package context.
    from .compare_init import INITIALIZATIONS, SMOKE_CONFIG, run_suite  # noqa: E402
except ImportError:  # pragma: no cover - exercised by direct execution
    from compare_init import INITIALIZATIONS, SMOKE_CONFIG, run_suite  # noqa: E402


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="deepflow-channel-init-smoke-") as directory:
        result = run_suite(SMOKE_CONFIG, Path(directory))
        assert set(result["variants"]) == set(INITIALIZATIONS)
        assert result["report"].exists()
        for initialization, values in result["variants"].items():
            data = values["evaluator"].data_dict
            assert all(data[field].size > 0 for field in ("u", "v", "p"))
            assert "history_last_total_loss" in values["metrics"]
            assert all(path.exists() for path in values["artifacts"])
            print(f"{initialization}: shared-harness smoke passed")


if __name__ == "__main__":
    main()
