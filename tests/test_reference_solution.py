import json

import numpy as np
import pytest
from deepflow.geometry import rectangle
from deepflow.reference import ReferenceSolution


def make_solution(**kwargs):
    return ReferenceSolution(
        area=rectangle([0, 1], [0, 1]),
        mesh=None,
        field_evaluator=lambda field, x, y: field(x, y),
        **kwargs,
    )


def test_steady_queries_broadcast_select_and_copy_cached_values():
    solution = make_solution(fields={"u": lambda x, y: x + 2 * y, "p": lambda x, y: 3})
    x = np.array([[0.25], [0.75]])
    y = np.array([0.2, 0.4, 0.6])
    values = solution.evaluate(x, y)
    np.testing.assert_allclose(values["u"], x + 2 * y)
    np.testing.assert_array_equal(values["p"], np.full((2, 3), 3))
    values["u"][:] = -100
    np.testing.assert_allclose(solution.evaluate(x, y)["u"], x + 2 * y)
    assert tuple(solution.evaluate(x, y, fields="p")) == ("p",)
    assert solution.evaluate(0.5, 0.5)["p"].shape == ()
    assert solution.evaluate([], [])["u"].shape == (0,)
    assert solution.evaluate(x, y, fields=[]) == {}
    assert solution.evaluate(0.5, 0.5, t=[0, 1])["u"].shape == (2,)
    with pytest.raises(KeyError, match="Unknown reference field"):
        solution.evaluate(0.5, 0.5, fields="missing")
    with pytest.raises(ValueError, match="inside the PDE domain"):
        solution.evaluate(1.1, 0.5)


@pytest.mark.parametrize("times", [[0.0, 0.4, 1.0], [0.0]])
def test_transient_interpolation_endpoints_and_time_tolerance(times):
    snapshots = [{"u": lambda x, y, t=t: x + y + t} for t in times]
    solution = make_solution(snapshots=snapshots, times=times)
    query_times = np.linspace(times[0], times[-1], 9)
    np.testing.assert_allclose(
        solution.evaluate(0.2, 0.3, t=query_times)["u"], 0.5 + query_times
    )
    np.testing.assert_allclose(
        solution.evaluate(0.2, 0.3, t=[times[0] - 1e-8, times[-1] + 1e-8])["u"],
        0.5 + np.array([times[0], times[-1]]),
    )
    copied_times = solution.times
    copied_times[:] = -1
    np.testing.assert_array_equal(solution.times, times)
    with pytest.raises(ValueError, match="require t"):
        solution.evaluate(0.2, 0.3)
    with pytest.raises(ValueError, match="times must lie"):
        solution.evaluate(0.2, 0.3, t=2)


def test_exact_endpoints_do_not_mix_nonfinite_neighbor_fields():
    solution = make_solution(
        snapshots=[{"u": lambda x, y: 2}, {"u": lambda x, y: np.inf}],
        times=[0, 1],
    )
    with np.errstate(invalid="ignore"):
        assert solution.evaluate(0.5, 0.5, t=0)["u"] == 2
        assert solution.evaluate(0.5, 0.5, t=1)["u"] == np.inf


def test_nonfinite_query_time_returns_nan():
    solution = make_solution(
        snapshots=[{"u": lambda x, y: 1}, {"u": lambda x, y: 2}], times=[0, 1]
    )
    assert np.isnan(solution.evaluate(0.5, 0.5, t=np.nan)["u"])


def test_time_from_y_and_npz_export(tmp_path):
    solution = make_solution(
        snapshots=[{"u": lambda x, y: x + y}, {"u": lambda x, y: x + y + 1}],
        times=[0, 1],
        time_from_y=True,
        metadata={"derived_fields": ["u"], "tag": "test"},
    )
    np.testing.assert_allclose(solution.evaluate([0.2, 0.4], 0.25)["u"], [0.7, 0.9])
    np.testing.assert_allclose(
        solution.evaluate([0.2, 0.4], 10, t=0.25)["u"], [0.7, 0.9]
    )
    path = tmp_path / "fields.npz"
    solution.export_npz(path, [0.2, 0.4], 10, t=0.25, fields="u")
    with np.load(path) as data:
        assert set(data.files) == {"x", "y", "t", "u", "time_values", "metadata"}
        np.testing.assert_allclose(data["u"], [0.7, 0.9])
        np.testing.assert_array_equal(data["y"], [10, 10])
        np.testing.assert_array_equal(data["t"], [0.25, 0.25])
        assert json.loads(str(data["metadata"])) == solution.metadata
    solution.export_npz(path, 0.2, 0.25)
    with np.load(path) as data:
        assert "t" not in data
        assert data["u"].shape == ()


def test_sampled_hole_boundary_is_queryable():
    from deepflow.geometry import circle

    hole = circle(0.5, 0.5, 0.2)
    area = rectangle([0, 1], [0, 1]) - hole
    for bound in hole.bound_list:
        bound.sampling_line(12)
    solution = ReferenceSolution(
        area=area, mesh=None, fields={"u": 1}, field_evaluator=lambda field, x, y: field
    )
    bound = hole.bound_list[0]
    np.testing.assert_array_equal(solution.evaluate(bound.X, bound.Y)["u"], 1)
    with pytest.raises(ValueError, match="inside the PDE domain"):
        solution.evaluate(0.5, 0.5)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({}, "must contain fields"),
        ({"snapshots": []}, "require corresponding times"),
        ({"snapshots": [], "times": []}, "strictly increasing"),
        ({"snapshots": [{"u": 0}], "times": [0, 1]}, "must match"),
        ({"snapshots": [{"u": 0}] * 2, "times": [1, 0]}, "strictly increasing"),
        ({"snapshots": [{"u": 0}, {"v": 0}], "times": [0, 1]}, "same fields"),
    ],
)
def test_invalid_solution_data(kwargs, message):
    with pytest.raises(ValueError, match=message):
        make_solution(**kwargs)
