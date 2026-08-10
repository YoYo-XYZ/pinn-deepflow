import os
import sys

import numpy as np
import pytest
import torch

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import deepflow as df
import deepflow.reference as reference_module
from deepflow.visualization import Visualizer


class _FakeReferenceSolution:
    def __init__(self, *, transient=False, time_from_y=False):
        self.fields = ("u", "v", "p")
        self.is_transient = transient
        self.time_from_y = time_from_y
        self.metadata = {
            "backend": "fake",
            "fields": list(self.fields),
            "transient": transient,
        }

    def evaluate(self, x, y, t=None, fields=None):
        fields = self.fields if fields is None else tuple(fields)
        if self.is_transient:
            assert t is not None
            base = np.asarray(t, dtype=float)
        else:
            base = np.zeros_like(np.asarray(x, dtype=float))
        values = {}
        for index, name in enumerate(fields):
            values[name] = base + np.asarray(x, dtype=float) + index
        return values


class _FakeReferenceSolver:
    calls = []
    solution = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def solve(self, domain):
        type(self).calls.append((self.kwargs, domain))
        return type(self).solution


@pytest.fixture
def fake_reference_solver(monkeypatch):
    _FakeReferenceSolver.calls = []
    _FakeReferenceSolver.solution = _FakeReferenceSolution()
    monkeypatch.setattr(reference_module, "ReferenceSolver", _FakeReferenceSolver)
    return _FakeReferenceSolver


def _domain_with_pde():
    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.NavierStokes(mu=1.0, rho=1.0))
    return area, df.domain(area)


def test_solve_fem_returns_reference_group_with_suffixed_fields(fake_reference_solver):
    area, domain = _domain_with_pde()

    result = domain.solve_fem(
        area_sampling_res=[3, 4],
        bound_sampling_res=5,
        mesh_size=0.05,
        boundary_resolution=64,
        max_iterations=7,
    )

    assert isinstance(result, df.GroupEvaluator)
    assert result.reference_solution is fake_reference_solver.solution
    assert result.metadata["backend"] == "fake"
    assert len(result.area_evaluators) == 1
    assert len(result.bound_evaluators) == 4

    area_data = result.area_evaluators[0].data_dict
    assert {"u_ref", "v_ref", "p_ref", "x", "y"} <= set(area_data)
    assert not any(
        key.endswith("_residual") or "loss" in key for key in area_data
    )
    assert area_data["u_ref"].shape == area_data["x"].shape
    assert len(fake_reference_solver.calls) == 1
    assert fake_reference_solver.calls[0][0]["mesh_size"] == 0.05
    assert fake_reference_solver.calls[0][0]["boundary_resolution"] == 64


def test_solve_fem_reuses_existing_coordinates_when_resolutions_are_omitted(
    fake_reference_solver,
):
    area, domain = _domain_with_pde()
    domain.sampling_uniform(
        bound_sampling_res=[4, 5, 6, 7],
        area_sampling_res=[[3, 3]],
    )
    original_area_x = area.X.clone()
    original_area_y = area.Y.clone()
    original_bound_x = [bound.X.clone() for bound in domain.bound_list]

    domain.solve_fem()

    assert np.array_equal(area.X.numpy(), original_area_x.numpy())
    assert np.array_equal(area.Y.numpy(), original_area_y.numpy())
    assert all(
        np.array_equal(bound.X.numpy(), original.numpy())
        for bound, original in zip(domain.bound_list, original_bound_x)
    )


def test_solve_fem_preserves_existing_time_when_processed_coordinates_are_missing(
    monkeypatch,
):
    solution = _FakeReferenceSolution(transient=True)
    _FakeReferenceSolver.solution = solution
    monkeypatch.setattr(reference_module, "ReferenceSolver", _FakeReferenceSolver)

    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.HeatEquation(alpha=0.1))
    domain = df.domain(area)
    domain.sampling_uniform(
        bound_sampling_res=[3, 3, 3, 3],
        area_sampling_res=[[2, 2]],
    )
    for geometry in [*domain.bound_list, area]:
        geometry.define_time([0.0, 1.0], sampling_scheme="uniform")
        geometry.process_coordinates()
    original_time = area.T.clone()
    area.X_ = None
    area.Y_ = None

    result = domain.solve_fem()

    assert np.array_equal(
        result.area_evaluators[0].data_dict["t"],
        original_time.numpy(),
    )


def test_solve_fem_regenerates_time_when_resampling_is_requested(monkeypatch):
    solution = _FakeReferenceSolution(transient=True)
    _FakeReferenceSolver.solution = solution
    monkeypatch.setattr(reference_module, "ReferenceSolver", _FakeReferenceSolver)

    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.HeatEquation(alpha=0.1))
    domain = df.domain(area)
    domain.sampling_uniform(
        bound_sampling_res=[3, 3, 3, 3],
        area_sampling_res=[[2, 2]],
    )
    for geometry in [*domain.bound_list, area]:
        geometry.define_time([0.0, 1.0], sampling_scheme="uniform")
        geometry.process_coordinates()
        geometry.t = torch.full_like(geometry.X, 0.25)
        geometry.T = geometry.t
        geometry.T_ = geometry.t.detach().clone().requires_grad_()
        geometry.inputs_tensor_dict["t"] = geometry.T_

    result = domain.solve_fem(
        area_sampling_res=[2, 2],
        bound_sampling_res=3,
    )

    regenerated = result.area_evaluators[0].data_dict["t"]
    assert regenerated.shape == (4,)
    assert not np.all(regenerated == 0.25)
    np.testing.assert_allclose(regenerated, np.linspace(0.0, 1.0, 4))


def test_solve_fem_accepts_nested_resolutions_for_multiple_areas(
    fake_reference_solver,
):
    area_a = df.rectangle([0, 1], [0, 1])
    area_b = df.rectangle([2, 3], [0, 1])
    area_a.define_pde(df.NavierStokes(mu=1.0, rho=1.0))
    domain = df.domain(area_a, area_b)

    result = domain.solve_fem(
        area_sampling_res=[[2, 3], [3, 2]],
        bound_sampling_res=4,
    )

    assert [evaluator.data_dict["x"].shape[0] for evaluator in result.area_evaluators] == [
        6,
        6,
    ]


def test_solve_fem_rejects_unsampled_geometry_before_backend(monkeypatch):
    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.NavierStokes(mu=1.0, rho=1.0))
    domain = df.domain(area)
    calls = []

    class ShouldNotRun:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def solve(self, domain):
            calls.append(("solve", domain))
            raise AssertionError("backend should not run")

    monkeypatch.setattr(reference_module, "ReferenceSolver", ShouldNotRun)

    with pytest.raises(ValueError, match="unsampled domain geometries"):
        domain.solve_fem()
    assert calls == []


def test_reference_refresh_and_time_updates_requery_solution(monkeypatch):
    solution = _FakeReferenceSolution(transient=True)
    _FakeReferenceSolver.solution = solution
    monkeypatch.setattr(reference_module, "ReferenceSolver", _FakeReferenceSolver)

    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.HeatEquation(alpha=0.1))
    domain = df.domain(area)
    for geometry in [*domain.bound_list, area]:
        geometry.define_time([0.0, 1.0], sampling_scheme="uniform")

    result = domain.solve_fem(area_sampling_res=[2, 2], bound_sampling_res=3)
    evaluator = result.area_evaluators[0]
    initial_values = evaluator.data_dict["u_ref"].copy()

    evaluator.define_time(0.5)
    assert np.all(evaluator.data_dict["t"] == 0.5)
    assert not np.array_equal(evaluator.data_dict["u_ref"], initial_values)

    area.sampling_area([3, 2])
    result.refresh()
    assert result.area_evaluators[0].data_dict["u_ref"].shape == (6,)


def test_reference_group_aggregate_plot_uses_suffixed_field(
    monkeypatch,
    fake_reference_solver,
):
    area, domain = _domain_with_pde()
    result = domain.solve_fem(area_sampling_res=[2, 2], bound_sampling_res=3)
    captured = {}

    def fake_plot_color(self, **kwargs):
        captured.update(kwargs)
        return self.data_dict

    monkeypatch.setattr(Visualizer, "plot_color", fake_plot_color)
    data = result.plot_color("u_ref")

    assert data["u_ref"].shape[0] == sum(
        evaluator.data_dict["u_ref"].shape[0] for evaluator in result
    )
    assert captured["color_axis"] == "u_ref"


def test_solve_fem_ngsolve_steady_smoke():
    pytest.importorskip("ngsolve")

    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.NavierStokes(mu=1.0, rho=1.0))
    for bound in area.bound_list:
        bound.define_bc({"u": 0.0, "v": 0.0})

    result = df.domain(area).solve_fem(
        mesh_size=0.75,
        boundary_resolution=8,
        max_iterations=3,
        area_sampling_res=[3, 3],
        bound_sampling_res=4,
    )

    assert isinstance(result, df.GroupEvaluator)
    assert result.area_evaluators[0].data_dict["u_ref"].shape == (9,)
