import ast
import json
from pathlib import Path

import numpy as np
import pytest

import deepflow as df
from deepflow.reference import ReferenceSolver


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
WORKFLOWS = [
    ("BFS_flow_steady/BFS_flow_steady.ipynb", dict(mesh_size=0.2, boundary_resolution=32, max_iterations=20)),
    ("burgers_eq/burgers_eq.ipynb", dict(mesh_size=0.5, boundary_resolution=16, max_iterations=3)),
    ("cavity_flow_steady/cavity_flow_steady.ipynb", dict(mesh_size=0.2, boundary_resolution=32, max_iterations=20)),
    ("channel_flow_transient/channel_flow_transient.ipynb", dict(mesh_size=0.4, boundary_resolution=32, time_step=1.0, max_iterations=10)),
    ("cylinder_flow_steady/cylinder_flow_steady.ipynb", dict(mesh_size=0.05, boundary_resolution=64, max_iterations=200)),
    ("heat_eq/heat_eq.ipynb", dict(mesh_size=0.1, boundary_resolution=32, time_step=0.1, max_iterations=1)),
    ("reference_solution/cylinder_flow_reference.ipynb", dict(mesh_size=0.05, boundary_resolution=64, max_iterations=200)),
]


def notebook_domain(path):
    namespace = {}
    notebook = json.loads((EXAMPLES / path).read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if "domain.sampling_" in source or "domain.solve_fem(" in source:
            break
        tree = ast.parse(source)
        tree.body = [
            statement for statement in tree.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Attribute)
                and statement.value.func.attr in {"show_setup", "show_coordinates"}
            )
        ]
        exec(compile(tree, str(EXAMPLES / path), "exec"), namespace)
    return namespace["domain"]


@pytest.fixture(autouse=True)
def ngsolve_backend():
    ngs = pytest.importorskip("ngsolve")
    ngs.SetNumThreads(1)


@pytest.mark.parametrize("path,options", WORKFLOWS, ids=[p.split('/')[0] for p, _ in WORKFLOWS])
def test_notebook_fem_workflow(path, options, tmp_path):
    domain = notebook_domain(path)
    result = domain.solve_fem(**options)
    solution = result.reference_solution
    area = result.area_list[0]
    area.sampling_area([11, 9])
    if solution.is_transient:
        area.define_time(float(solution.times[-1]))
    assert area["u_ref"].shape == area["x"].shape
    assert np.isfinite(area["u_ref"]).all()
    query = {"t": float(solution.times[-1])} if solution.is_transient else {}
    values = solution.evaluate(area["x"], area["y"], **query)
    for name, value in values.items():
        assert np.isfinite(value).all()
        np.testing.assert_allclose(value, area[f"{name}_ref"])
    solution.export_npz(tmp_path / "reference.npz", area["x"], area["y"], **query)
    assert solution.metadata["backend"] == "NGSolve/Netgen"
    assert solution.metadata["mesh"]["elements"] > 0


@pytest.mark.parametrize("kind", ["heat", "wave", "steady_navier", "transient_navier", "stream", "burgers"])
def test_constant_exact_solutions(kind):
    area = df.geometry.rectangle([0, 1], [0, 1])
    initial = df.geometry.rectangle([0, 1], [0, 1])
    domain = df.domain(area, initial.area_list)
    if kind in {"heat", "wave", "burgers"}:
        pde = {"heat": df.pde.HeatEquation(0.1), "wave": df.pde.WaveEquation(1), "burgers": df.pde.BurgersEquation1D(0.1)}[kind]
        conditions = {"u": 2}
    else:
        pde = df.pde.StreamFunctionNavierStokes(1, 1) if kind == "stream" else df.pde.NavierStokes(1, 1)
        conditions = {"u": 0, "v": 0}
    area.define_pde(pde)
    for bound in area.bound_list:
        bound.define_bc(conditions)
    if kind in {"heat", "wave", "transient_navier"}:
        for geometry in domain:
            geometry.define_time([0, 0.2])
        initial.define_ic(conditions)
    solution = ReferenceSolver(mesh_size=0.3, boundary_resolution=8, time_step=0.1).solve(domain)
    query = {"t": [0, 0.05, 0.2]} if solution.is_transient else {}
    values = solution.evaluate(0.5, 0.5, **query)
    for name, value in values.items():
        np.testing.assert_allclose(value, 2 if kind in {"heat", "wave", "burgers"} else 0, atol=1e-10)
    if kind == "stream":
        assert solution.fields == ("psi", "u", "v", "p")
        assert solution.derived_fields == ("psi",)


@pytest.mark.parametrize("kind", ["heat", "wave"])
def test_diffusion_and_wave_neumann_eigenmode(kind):
    area = df.geometry.rectangle([0, 1], [0, 1])
    initial = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.pde.HeatEquation(0.1) if kind == "heat" else df.pde.WaveEquation(1))
    initial.define_ic({"u": ("x", lambda x: np.cos(np.pi * x))})
    domain = df.domain(area, initial.area_list)
    for geometry in domain:
        geometry.define_time([0, 0.1])
    solution = ReferenceSolver(mesh_size=0.15, boundary_resolution=8, time_step=0.01).solve(domain)
    x = np.array([0.2, 0.4, 0.7])
    factor = np.exp(-0.1 * np.pi**2 * 0.1) if kind == "heat" else np.cos(np.pi * 0.1)
    np.testing.assert_allclose(solution.evaluate(x, 0.5, t=0.1)["u"], factor * np.cos(np.pi * x), atol=0.003)


def test_wave_initial_velocity():
    area = df.geometry.rectangle([0, 1], [0, 1])
    initial = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.pde.WaveEquation(1))
    initial.define_ic({"u": 1, "u_t": 2})
    domain = df.domain(area, initial.area_list)
    for geometry in domain:
        geometry.define_time([0, 0.2])
    solution = ReferenceSolver(mesh_size=0.3, boundary_resolution=8, time_step=0.1).solve(domain)
    np.testing.assert_allclose(
        solution.evaluate(0.5, 0.5, t=[0, 0.05, 0.2])["u"], [1, 1.1, 1.4], atol=1e-10
    )


def test_stream_function_reconstructs_uniform_velocity():
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.pde.StreamFunctionNavierStokes(1, 1))
    for bound in area.bound_list:
        bound.define_bc({"u": 1, "v": 0, "psi": ("y", lambda y: y)})
    solution = ReferenceSolver(mesh_size=0.3, boundary_resolution=8).solve(df.domain(area))
    values = solution.evaluate([0.2, 0.5, 0.8], [0.3, 0.5, 0.7])
    np.testing.assert_allclose(values["psi"], [0.3, 0.5, 0.7], atol=1e-10)
    np.testing.assert_allclose(values["u"], 1, atol=1e-10)
    np.testing.assert_allclose(values["v"], 0, atol=1e-10)
