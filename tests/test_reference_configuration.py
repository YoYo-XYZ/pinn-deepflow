import subprocess
import sys

import numpy as np
import pytest

import deepflow as df
from deepflow.reference import ReferenceConfigurationError, ReferenceSolver, UnsupportedReferencePDE


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mesh_size": 0}, {"mesh_size": np.nan}, {"mesh_size": np.inf},
        {"boundary_resolution": 3}, {"boundary_resolution": 4.5},
        {"time_step": 0}, {"time_step": np.nan},
        {"tolerance": 0}, {"tolerance": np.inf},
        {"max_iterations": 0}, {"max_iterations": 1.5},
    ],
)
def test_invalid_solver_options(kwargs):
    with pytest.raises(ValueError):
        ReferenceSolver(**kwargs)


def test_optional_backend_is_lazy_and_missing_dependency_has_install_hint():
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockNGSolve(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'ngsolve', 'netgen'}:
            raise ImportError('blocked for optional dependency check')

sys.meta_path.insert(0, BlockNGSolve())
import deepflow as df
from deepflow.reference import ReferenceSolver, ReferenceSolution
assert 'ngsolve' not in sys.modules
assert df.ReferenceSolver is ReferenceSolver
assert df.ReferenceSolution is ReferenceSolution
solver = ReferenceSolver(mesh_size=0.5, boundary_resolution=4)
assert solver.clear_cache() is None
area = df.geometry.rectangle([0, 1], [0, 1])
area.define_pde(df.NavierStokes(1, 1))
try:
    solver.solve(df.domain(area))
except ImportError as exc:
    assert 'pip install deepflow[cfd]' in str(exc)
else:
    raise AssertionError('Expected a missing optional dependency error')
"""
    subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["missing_pde", "multiple_pdes", "heat_without_time", "wave_without_time", "stream_with_time", "custom", "derivative_bc", "unknown_bc", "bad_condition", "bad_time", "unknown_ic"])
def test_unsupported_configurations_fail_before_loading_backend(kind, monkeypatch):
    from deepflow.reference import solver as module

    def unexpected_load():
        pytest.fail("Invalid configuration attempted to load NGSolve")

    monkeypatch.setattr(module, "_load_ngsolve", unexpected_load)
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.NavierStokes(1, 1))
    domain = df.domain(area)
    expected = ReferenceConfigurationError
    if kind == "missing_pde":
        domain = df.domain(df.geometry.rectangle([0, 1], [0, 1]))
    elif kind == "multiple_pdes":
        domain.area_list.append(area)
    elif kind in {"heat_without_time", "wave_without_time"}:
        area.define_pde(df.pde.HeatEquation(0.1) if kind.startswith("heat") else df.pde.WaveEquation(1))
    elif kind == "stream_with_time":
        area.define_pde(df.StreamFunctionNavierStokes(1, 1))
        area.range_t = [0, 1]
    elif kind == "custom":
        area.define_pde(df.pde.CustomPDE(lambda values: values))
        expected = UnsupportedReferencePDE
    elif kind == "derivative_bc":
        area.bound_list[0].define_bc({"u_x": 0})
    elif kind == "unknown_bc":
        area.bound_list[0].define_bc({"temperature": 0})
    elif kind == "bad_condition":
        area.bound_list[0].define_bc({"u": ("t", lambda t: t)})
    elif kind == "bad_time":
        area.range_t = [1, 0]
    elif kind == "unknown_ic":
        initial = df.geometry.rectangle([0, 1], [0, 1])
        initial.define_ic({"temperature": 0})
        domain.area_list.append(initial)
    with pytest.raises(expected):
        ReferenceSolver().solve(domain)
