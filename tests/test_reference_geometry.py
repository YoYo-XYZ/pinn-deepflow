import numpy as np
import pytest
import torch

import deepflow as df
from deepflow.reference import ReferenceGeometryError, ReferenceSolver
from deepflow.reference.geometry import NetgenGeometryAdapter, _loop_points, _signed_area


@pytest.fixture
def ngs():
    module = pytest.importorskip("ngsolve")
    module.SetNumThreads(1)
    return module


def test_shuffled_reversed_outer_loop_and_hole_keep_labels_and_orientation(ngs):
    area = df.geometry.rectangle([0, 2], [0, 2]) - df.geometry.circle(1, 1, 0.2)
    area.bound_list[:] = area.bound_list[::-1]
    geometry = NetgenGeometryAdapter(12).build(area, ngs, 0.4)
    assert len(geometry.loops) == 2
    assert _signed_area(_loop_points(geometry.loops[0])) > 0
    assert _signed_area(_loop_points(geometry.loops[1])) < 0
    for index, bound in enumerate(area.bound_list):
        assert geometry.labels_by_bound[id(bound)] == f"outer_{index:04d}"
    for loop in geometry.loops:
        for edge in loop.edges:
            assert geometry.boundary_info[edge.label]["hole"] is loop.hole
            np.testing.assert_array_equal(geometry.points_by_label[edge.label], edge.points)


@pytest.mark.parametrize("kind,message", [
    ("open", "closed loop"),
    ("ambiguous", "ambiguous junction"),
    ("disconnected", "exactly one connected"),
    ("outside_hole", "outside the outer domain"),
    ("implicit", "explicit Bound objects"),
])
def test_invalid_geometry_is_rejected(ngs, kind, message):
    area = df.geometry.rectangle([0, 1], [0, 1])
    if kind == "open":
        area.bound_list.pop()
    elif kind == "ambiguous":
        area.bound_list.append(area.bound_list[1])
    elif kind == "disconnected":
        area.bound_list.extend(df.geometry.rectangle([2, 3], [0, 1]).bound_list)
    elif kind == "outside_hole":
        area.negative_bound_list = df.geometry.circle(2, 2, 0.2).bound_list
    elif kind == "implicit":
        area.bound_list.clear()
    with pytest.raises(ReferenceGeometryError, match=message):
        NetgenGeometryAdapter(8).build(area, ngs, 0.3)


def test_standalone_boundary_matching_preserves_sampled_coordinates(ngs):
    area = df.geometry.rectangle([0, 1], [0, 1])
    standalone = df.geometry.line_vertical(0, [0, 1])
    standalone.define_bc({"u": 0, "v": 0})
    standalone.sampling_line(9, scheme="random")
    standalone.process_coordinates()
    original = {name: value.clone() for name, value in standalone.coords.items()}
    original_x, original_y = standalone.X, standalone.Y
    for bound in area.bound_list[1:]:
        bound.define_bc({"u": 0, "v": 0})
    area.define_pde(df.NavierStokes(1, 1))
    domain = df.domain(area.area_list, standalone, area.bound_list[1:])
    solution = ReferenceSolver(mesh_size=0.3, boundary_resolution=12).solve(domain)
    assert standalone.X is original_x
    assert standalone.Y is original_y
    for name, value in original.items():
        torch.testing.assert_close(standalone.coords[name], value)
    np.testing.assert_allclose(solution.evaluate(0.5, 0.5)["u"], 0, atol=1e-12)
