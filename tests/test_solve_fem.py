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


def test_solve_fem_returns_reference_solution_without_sampling(fake_reference_solver):
    _, domain = _domain_with_pde()

    result = domain.solve_fem(
        mesh_size=0.05,
        boundary_resolution=64,
        max_iterations=7,
    )

    assert isinstance(result, df.ReferenceGroupEvaluator)
    assert result.reference_solution is fake_reference_solver.solution
    assert result.metadata["backend"] == "fake"
    values = result.evaluate(np.array([0.25]), np.array([0.5]))
    np.testing.assert_allclose(values["u"], [0.25])
    assert len(fake_reference_solver.calls) == 1
    assert fake_reference_solver.calls[0][0] == {
        "mesh_size": 0.05,
        "boundary_resolution": 64,
        "time_step": None,
        "tolerance": 1e-8,
        "max_iterations": 7,
    }


def test_solve_fem_does_not_require_sampled_coordinates(monkeypatch):
    area, domain = _domain_with_pde()
    calls = []
    solution = _FakeReferenceSolution()

    class ShouldRun:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def solve(self, solved_domain):
            calls.append(("solve", solved_domain))
            return solution

    monkeypatch.setattr(reference_module, "ReferenceSolver", ShouldRun)

    result = domain.solve_fem()
    assert isinstance(result, df.ReferenceGroupEvaluator)
    assert result.reference_solution is solution
    assert calls[0] == (
        "init",
        {
            "mesh_size": None,
            "boundary_resolution": 128,
            "time_step": None,
            "tolerance": 1e-8,
            "max_iterations": 200,
        },
    )
    assert calls[1] == ("solve", domain)
    assert area.X is None


def test_solve_fem_returns_transient_reference_solution_without_sampling(monkeypatch):
    solution = _FakeReferenceSolution(transient=True)
    _FakeReferenceSolver.solution = solution
    monkeypatch.setattr(reference_module, "ReferenceSolver", _FakeReferenceSolver)

    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.HeatEquation(alpha=0.1))
    domain = df.domain(area)

    result = domain.solve_fem()

    values = result.evaluate(
        np.array([0.25, 0.75]),
        np.array([0.5, 0.5]),
        t=np.array([0.0, 1.0]),
    )
    np.testing.assert_allclose(values["u"], [0.25, 1.75])


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
    )

    assert isinstance(result, df.ReferenceGroupEvaluator)
    assert result.metadata["mesh"]["dimension"] == 2
    values = result.evaluate(np.array([0.5]), np.array([0.5]))
    assert values["u"].shape == (1,)


def test_solve_fem_burgers_is_steady_2d():
    pytest.importorskip("ngsolve")

    area = df.rectangle([0, 1], [0, 1])
    area.define_pde(df.pde.BurgersEquation1D(nu=0.1))
    for bound in area.bound_list:
        bound.define_bc({"u": 1.0})

    result = df.domain(area).solve_fem(
        mesh_size=0.5,
        boundary_resolution=8,
        max_iterations=5,
    )

    assert not result.reference_solution.is_transient
    assert not result.reference_solution.time_from_y
    assert result.metadata["mesh"]["dimension"] == 2
    assert "time_values" not in result.metadata
    values = result.evaluate(
        np.array([0.25, 0.75]), np.array([0.25, 0.75])
    )
    np.testing.assert_allclose(values["u"], 1.0, atol=1e-8)


def test_solve_fem_accepts_external_matching_boundary_conditions():
    pytest.importorskip("ngsolve")

    area = df.rectangle([-1, 1], [0, 1])
    line_ic = df.geometry.line_horizontal(y=0, range_x=[-1, 1])
    line_bc1 = df.geometry.line_vertical(x=-1, range_y=[0, 1])
    line_bc2 = df.geometry.line_vertical(x=1, range_y=[0, 1])
    domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)

    area.define_pde(df.pde.BurgersEquation1D(nu=0.1))
    domain.bound_list[0].define_bc(
        {"u": ["x", lambda x: -torch.sin(torch.pi * x)]}
    )
    domain.bound_list[1].define_bc({"u": 0.0})
    domain.bound_list[2].define_bc({"u": 0.0})

    result = domain.solve_fem(
        mesh_size=0.25,
        boundary_resolution=16,
        max_iterations=25,
    )
    x = np.array([-0.75, 0.0, 0.75])
    values = result.evaluate(x, np.zeros_like(x))["u"]

    np.testing.assert_allclose(values, -np.sin(np.pi * x), atol=5e-4)
