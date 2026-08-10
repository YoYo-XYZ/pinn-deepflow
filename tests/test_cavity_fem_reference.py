import sys
from pathlib import Path

import numpy as np
import pytest


pytest.importorskip("ngsolve")

_BENCHMARK_DIR = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "comparing_pinns_qcpinn_formulations_cavity"
)
if str(_BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCHMARK_DIR))

import reference  # noqa: E402


def test_fem_reference_exports_legacy_grid_schema_and_pressure_gauge():
    payload = reference.solve_reference((5, 5), 0.1)

    assert payload["converged"] == 1
    assert payload["nx"] == 5
    assert payload["ny"] == 5
    assert payload["u"].shape == (5, 5)
    assert payload["v"].shape == (5, 5)
    assert payload["p"].shape == (5, 5)
    assert np.isfinite(payload["u"]).all()
    assert np.isfinite(payload["v"]).all()
    assert np.isfinite(payload["p"]).all()
    assert payload["pressure_gauge"].item() == "corner_anchored"
    assert np.isfinite(payload["pressure_offset"])
