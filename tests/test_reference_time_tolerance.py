import numpy as np
import pytest


def _transient_solution():
    import deepflow as df
    from deepflow.reference import ReferenceSolution

    area = df.geometry.rectangle([0, 1], [0, 1])
    return ReferenceSolution(
        area=area,
        mesh=None,
        snapshots=[{"u": 0.0}, {"u": 1.0}],
        times=[0.0, 0.1],
        field_evaluator=lambda field, x, y: np.full_like(x, field, dtype=float),
    )


def test_float32_transient_endpoint_is_accepted_and_clamped():
    solution = _transient_solution()

    values = solution.evaluate(
        0.5,
        0.5,
        t=np.float32(0.1),
        fields=["u"],
    )

    np.testing.assert_allclose(values["u"], 1.0)


def test_genuinely_out_of_range_transient_time_is_rejected():
    solution = _transient_solution()

    with pytest.raises(ValueError, match="Transient query times"):
        solution.evaluate(0.5, 0.5, t=0.1001, fields=["u"])
