import os
import subprocess
import sys

import numpy as np
import pytest


def test_reference_import_is_lazy():
    source = (
        "import sys; import deepflow; "
        "assert 'ngsolve' not in sys.modules; "
        "from deepflow.reference import ReferenceSolver, ReferenceSolution; "
        "assert 'ngsolve' not in sys.modules; "
        "assert deepflow.ReferenceSolver is ReferenceSolver"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
    subprocess.run([sys.executable, "-c", source], check=True, env=env)


def test_custom_pde_is_rejected_before_optional_backend_load():
    import deepflow as df

    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.CustomPDE(lambda _: (0,)))
    with pytest.raises(NotImplementedError, match="CustomPDE"):
        df.ReferenceSolver().solve(df.domain(area))


@pytest.mark.parametrize("boundary_resolution", [3, 4.5, None])
def test_reference_solver_validates_boundary_resolution(boundary_resolution):
    import deepflow as df

    with pytest.raises(
        ValueError,
        match="boundary_resolution must be an integer >= 4",
    ):
        df.ReferenceSolver(boundary_resolution=boundary_resolution)


def test_reference_solution_accepts_float32_curved_boundary_queries():
    import deepflow as df
    from deepflow.reference import ReferenceSolution

    area = df.geometry.circle(0.2, 0.2, 0.05)
    for bound in area.bound_list:
        bound.sampling_line(64)
        bound.process_coordinates()

    solution = ReferenceSolution(
        area=area,
        mesh=None,
        fields={"u": object()},
        field_evaluator=lambda field, x, y: np.zeros_like(x),
    )

    for bound in area.bound_list:
        values = solution.evaluate(
            bound.X.numpy(),
            bound.Y.numpy(),
            fields=["u"],
        )
        assert values["u"].shape == bound.X.shape


def test_reference_solution_uses_a_single_defensive_query_cache():
    import deepflow as df
    from deepflow.reference import ReferenceSolution

    area = df.geometry.rectangle([0, 1], [0, 1])
    calls = []

    def evaluate_field(field, x, y):
        calls.append((x.copy(), y.copy()))
        return np.full_like(x, field, dtype=float)

    solution = ReferenceSolution(
        area=area,
        mesh=None,
        fields={"u": 2.0},
        field_evaluator=evaluate_field,
    )

    first = solution.evaluate([0.25], [0.25])
    first["u"][0] = 99.0
    repeated = solution.evaluate([0.25], [0.25])
    np.testing.assert_allclose(repeated["u"], [2.0])
    assert len(calls) == 1

    solution.evaluate([0.5], [0.5])
    solution.evaluate([0.25], [0.25])
    assert len(calls) == 3


def test_reference_solution_rejects_inconsistent_snapshot_fields():
    import deepflow as df
    from deepflow.reference import ReferenceSolution

    area = df.geometry.rectangle([0, 1], [0, 1])
    with pytest.raises(
        ValueError,
        match="All transient snapshots must contain the same fields",
    ):
        ReferenceSolution(
            area=area,
            mesh=None,
            snapshots=[{"u": 0.0}, {"v": 1.0}],
            times=[0.0, 1.0],
            field_evaluator=lambda field, x, y: np.full_like(x, field),
        )


def test_rectangle_circle_polygon_and_hole_meshes():
    ngsolve = pytest.importorskip("ngsolve")
    import deepflow as df
    from deepflow.reference.geometry import NetgenGeometryAdapter

    adapter = NetgenGeometryAdapter(boundary_resolution=16)
    geometries = [
        df.geometry.rectangle([0, 1], [0, 1]),
        df.geometry.circle(0, 0, 1),
        df.geometry.polygon([0, 0], [2, 0], [2, 1], [0, 1]),
        df.geometry.rectangle([-2, 2], [-2, 2]) - df.geometry.circle(0, 0, 0.5),
    ]
    meshes = [adapter.build(geometry, ngsolve, 0.35).mesh for geometry in geometries]
    assert all(mesh.ne > 0 for mesh in meshes)
    assert len(meshes[-1].GetBoundaries()) > 4


def test_open_boundary_is_rejected():
    ngsolve = pytest.importorskip("ngsolve")
    import deepflow as df
    from deepflow.reference.geometry import NetgenGeometryAdapter, ReferenceGeometryError

    open_area = df.geometry.Area(
        [df.geometry.line([0, 0], [1, 0]), df.geometry.line([1, 0], [1, 1])],
        ranges={0: (0, 1), 1: (0, 1)},
    )
    with pytest.raises(ReferenceGeometryError, match="closed loop"):
        NetgenGeometryAdapter(8).build(open_area, ngsolve, 0.25)
