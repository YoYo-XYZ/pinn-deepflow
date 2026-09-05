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

import deepflow as df  # noqa: E402

import reference  # noqa: E402


def test_fem_reference_returns_native_reference_evaluator():
    result = reference.solve_reference(
        mesh_size=0.1,
        boundary_resolution=16,
    )

    assert isinstance(result, df.ReferenceGroupEvaluator)
    assert result.metadata["converged"]
    values = result.reference_solution.evaluate(
        np.array([0.25, 0.75]),
        np.array([0.25, 0.75]),
        fields=("u", "v", "p"),
    )
    assert all(np.isfinite(values[field]).all() for field in ("u", "v", "p"))
