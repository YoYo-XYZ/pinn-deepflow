import os
import subprocess
import sys

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
