import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.mark.parametrize("reverse", [False, True])
def test_concave_polygon_membership_includes_edges_and_vertices(reverse):
    vertices = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]]
    area = df.geometry.polygon(*(vertices[::-1] if reverse else vertices))
    x = torch.tensor([[0.5, 1.5, 1.5], [0.0, 1.0, 2.1]])
    y = torch.tensor([[1.5, 0.5, 1.5], [0.0, 1.5, 0.5]])
    assert torch.equal(
        area.contains(x, y), torch.tensor([[True, True, False], [True, True, False]])
    )
    assert area.contains(torch.tensor(0.5), torch.tensor(0.5)).item()
    assert area.contains(torch.empty(0), torch.empty(0)).shape == (0,)


@pytest.mark.parametrize(
    "vertices",
    [[], [[0, 0], [1, 1]], [[0], [1], [2]], [[0, 0], [1, 0], [0, float("nan")]]],
)
def test_invalid_polygon_vertices(vertices):
    with pytest.raises(ValueError, match="vertices"):
        df.geometry.polygon(*vertices)


@pytest.mark.parametrize("radius", [0, -1])
def test_circle_requires_positive_radius(radius):
    with pytest.raises(ValueError, match="radius must be positive"):
        df.geometry.circle(0, 0, radius)


def test_circle_membership_and_boundary_sampling_follow_exact_radius():
    area = df.geometry.circle(2, -1, 0.5)
    x, y = torch.tensor([2.0, 2.5, 2.6]), torch.tensor([-1.0, -1.0, -1.0])
    assert torch.equal(area.contains(x, y), torch.tensor([True, True, False]))
    for boundary in area:
        x, y = boundary.sampling_line(17)
        torch.testing.assert_close(
            (x - 2) ** 2 + (y + 1) ** 2, torch.full_like(x, 0.25)
        )


def test_geometry_composition_preserves_holes_and_operands():
    rectangle = df.geometry.rectangle([0, 2], [0, 2])
    hole = df.geometry.circle(1, 1, 0.3)
    extension = df.geometry.rectangle([2, 3], [0, 1])
    area = (rectangle - hole) | extension
    x, y = torch.tensor([0.2, 1.0, 2.5, 2.5]), torch.tensor([0.2, 1.0, 0.5, 1.5])
    assert torch.equal(area.contains(x, y), torch.tensor([True, False, True, False]))
    assert rectangle.contains(torch.tensor(1.0), torch.tensor(1.0))
    assert len(rectangle.bound_list) == 4
    assert rectangle.negative_bound_list is None
    torch.testing.assert_close(
        (rectangle + extension).contains(x, y), (rectangle | extension).contains(x, y)
    )
    filled = area | hole
    assert filled.contains(torch.tensor(1.0), torch.tensor(1.0))


def test_attaching_boundary_does_not_change_area_membership():
    area = df.geometry.rectangle([0, 1], [0, 1])
    boundary = df.geometry.line_horizontal(3, [0, 4])
    attached = area + boundary
    x, y = torch.tensor([0.5, 2.0]), torch.tensor([0.5, 3.0])
    torch.testing.assert_close(attached.contains(x, y), area.contains(x, y))
    assert attached.bound_list[-1] is boundary
    assert len(area.bound_list) == 4


@pytest.mark.parametrize("scheme", ["uniform", "random", "lhs"])
def test_sampling_respects_rectangular_ranges_and_replays_seed(scheme):
    area = df.geometry.rectangle([-2, 4], [3, 5])
    df.manual_seed(19)
    x, y = area.sampling_area([5, 3], scheme=scheme)
    assert x.shape == y.shape == (15,)
    assert ((x >= -2) & (x <= 4) & (y >= 3) & (y <= 5)).all()
    df.manual_seed(19)
    actual_x, actual_y = area.sampling_area([5, 3], scheme=scheme)
    torch.testing.assert_close(actual_x, x)
    torch.testing.assert_close(actual_y, y)
    if scheme == "uniform":
        torch.testing.assert_close(
            x.reshape(5, 3)[:, 0], torch.tensor([-2 + 1e-6, -0.5, 1.0, 2.5, 4 - 1e-6])
        )
        torch.testing.assert_close(
            y.reshape(5, 3)[0], torch.tensor([3 + 1e-6, 4.0, 5 - 1e-6])
        )


@pytest.mark.parametrize("scheme", ["uniform", "random", "lhs"])
def test_samples_exclude_holes_and_concave_cutouts(scheme):
    area = df.geometry.rectangle([-1, 1], [-1, 1]) - df.geometry.circle(0, 0, 0.6)
    x, y = area.sampling_area([12, 12], scheme=scheme)
    assert 0 < len(x) < 144
    assert (x**2 + y**2 > 0.36).all()
    assert (x.abs() <= 1).all() and (y.abs() <= 1).all()


@pytest.mark.parametrize("scheme", ["uniform", "random", "lhs"])
@pytest.mark.parametrize(
    "endpoints", [([2, -1], [2, 3]), ([-1, 3], [2, 3]), ([2, 3], [-1, -3])]
)
def test_line_samples_lie_on_segment(endpoints, scheme):
    first, last = map(torch.tensor, endpoints)
    boundary = df.geometry.line(first.tolist(), last.tolist())
    x, y = boundary.sampling_line(11, scheme=scheme)
    delta = last - first
    torch.testing.assert_close(
        (x - first[0]) * delta[1] - (y - first[1]) * delta[0], torch.zeros_like(x)
    )
    assert x.shape == y.shape == (11,)
    assert ((x >= min(first[0], last[0])) & (x <= max(first[0], last[0]))).all()
    assert ((y >= min(first[1], last[1])) & (y <= max(first[1], last[1]))).all()


def test_parametric_curve_samples_both_coordinate_functions():
    boundary = df.geometry.curve([0, 2], lambda t: t**2, lambda t: 3 * t, ref_axis="t")
    x, y = boundary.sampling_line(5)
    t = torch.linspace(0, 2, 5)
    torch.testing.assert_close(x, t**2)
    torch.testing.assert_close(y, 3 * t)


def test_sampling_all_boundaries_has_one_row_per_boundary():
    area = df.geometry.rectangle([0, 2], [0, 1])
    x, y = area.sampling_lines(5, scheme="uniform")
    assert x.shape == y.shape == (4, 5)
    torch.testing.assert_close(x[0], torch.zeros(5))
    torch.testing.assert_close(y[1], torch.zeros(5))
    torch.testing.assert_close(x[2], torch.full((5,), 2.0))
    torch.testing.assert_close(y[3], torch.ones(5))


@pytest.mark.parametrize("resolutions", [(), (2, 2), (2, 3, 2, 2)])
def test_invalid_boundary_resolutions(resolutions):
    with pytest.raises(ValueError, match="resolution"):
        df.geometry.rectangle([0, 1], [0, 1]).sampling_lines(*resolutions)


@pytest.mark.parametrize(
    "x,y,error",
    [
        ([0.0], [0.0], TypeError),
        (torch.zeros(2), torch.zeros(3), ValueError),
        (
            torch.zeros(2, dtype=torch.float64),
            torch.zeros(2, dtype=torch.float32),
            ValueError,
        ),
        (
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, dtype=torch.int64),
            TypeError,
        ),
    ],
)
def test_contains_rejects_incompatible_coordinates(x, y, error):
    with pytest.raises(error):
        df.geometry.rectangle([0, 1], [0, 1]).contains(x, y)


@pytest.mark.parametrize(
    "result", [True, torch.ones(2), torch.ones(1, dtype=torch.bool)]
)
def test_contains_rejects_invalid_custom_masks(result):
    area = df.geometry.Area(
        [], contains_fn=lambda x, y: result, ranges={0: (0, 1), 1: (0, 1)}
    )
    with pytest.raises(TypeError, match="contains function"):
        area.contains(torch.zeros(2), torch.zeros(2))


@pytest.mark.parametrize("ref_axis", ["x", "y", "t"])
def test_curve_sampling_supports_each_reference_axis(ref_axis):
    if ref_axis == "x":
        boundary = df.geometry.curve([0, 1], lambda x: x**2, ref_axis="x")
    elif ref_axis == "y":
        boundary = df.geometry.curve([0, 1], lambda y: y.sqrt(), ref_axis="y")
    else:
        boundary = df.geometry.curve([0, 1], lambda t: t, lambda t: t**2, ref_axis="t")
    x, y = boundary.sampling_line(5)
    torch.testing.assert_close(y, x**2)
    torch.testing.assert_close(torch.stack((x[0], y[0])), torch.zeros(2))
    torch.testing.assert_close(torch.stack((x[-1], y[-1])), torch.ones(2))


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Area built directly from bounds accepts points outside both coordinate ranges",
)
def test_area_from_explicit_boundaries_rejects_exterior_points():
    rectangle = df.geometry.rectangle([0, 1], [0, 1])
    area = df.geometry.Area(rectangle.bound_list)
    x, y = torch.tensor([0.5, 2.0, -1.0]), torch.tensor([0.5, 2.0, -1.0])
    torch.testing.assert_close(area.contains(x, y), torch.tensor([True, False, False]))
