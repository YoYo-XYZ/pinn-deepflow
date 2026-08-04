import torch
import pytest

from deepflow import geometry


def _contains(area, points):
    coordinates = torch.tensor(points, dtype=torch.float32)
    return area.contains(coordinates[:, 0], coordinates[:, 1])


def test_rectangle_contains_inside_boundary_and_outside_points():
    area = geometry.rectangle([0, 2], [0, 1])

    mask = _contains(area, [(1, 0.5), (0, 0.5), (-1, 0.5), (1, 2)])

    assert mask.tolist() == [True, True, False, False]


def test_circle_contains_inside_boundary_and_outside_points():
    area = geometry.circle(1, 2, 0.5)

    mask = _contains(area, [(1, 2), (1.5, 2), (1.5, 2.1)])

    assert mask.tolist() == [True, True, False]


def test_concave_polygon_uses_even_odd_membership():
    area = geometry.polygon(
        [0, 0],
        [2, 0],
        [2, 1],
        [1, 1],
        [1, 2],
        [0, 2],
    )

    mask = _contains(area, [(0.5, 0.5), (0.5, 1.5), (1.5, 0.5), (1.5, 1.5)])

    assert mask.tolist() == [True, True, True, False]


def test_polygon_membership_is_independent_of_vertex_order():
    vertices = ([0, 0], [2, 0], [2, 1], [0, 1])
    clockwise = geometry.polygon(*vertices)
    counter_clockwise = geometry.polygon(*reversed(vertices))
    points = torch.tensor([[1, 0.5], [3, 0.5], [0, 0.5]], dtype=torch.float64)

    clockwise_mask = clockwise.contains(points[:, 0], points[:, 1])
    counter_clockwise_mask = counter_clockwise.contains(points[:, 0], points[:, 1])

    assert torch.equal(clockwise_mask, counter_clockwise_mask)
    assert clockwise_mask.tolist() == [True, False, True]


def test_subtraction_composes_membership_and_preserves_previous_holes():
    outer = geometry.rectangle([0, 4], [0, 2])
    first_hole = geometry.circle(1, 1, 0.25)
    second_hole = geometry.circle(3, 1, 0.25)
    area = outer - first_hole - second_hole

    mask = _contains(area, [(2, 1), (1, 1), (3, 1), (5, 1)])

    assert mask.tolist() == [True, False, False, False]
    assert len(area.negative_bound_list) == 4


def test_area_addition_matches_boolean_union():
    left = geometry.rectangle([0, 2], [0, 1])
    right = geometry.rectangle([1, 3], [0, 1])
    added = left + right
    union = left | right
    points = torch.tensor(
        [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5], [3.5, 0.5]],
        dtype=torch.float32,
    )

    added_mask = added.contains(points[:, 0], points[:, 1])
    union_mask = union.contains(points[:, 0], points[:, 1])

    assert torch.equal(added_mask, union_mask)
    assert added_mask.tolist() == [True, True, True, False]
    assert added.ranges == {0: (0.0, 3.0), 1: (0.0, 1.0)}


def test_disjoint_union_excludes_gap_between_areas():
    left = geometry.rectangle([0, 1], [0, 1])
    right = geometry.rectangle([2, 3], [0, 1])
    area = left + right

    mask = _contains(area, [(0.5, 0.5), (1.5, 0.5), (2.5, 0.5)])

    assert mask.tolist() == [True, False, True]


def test_area_add_bound_retains_legacy_behavior():
    area = geometry.rectangle([0, 1], [0, 1])
    extra_bound = geometry.line_horizontal(0.5, [0, 1])

    combined = area + extra_bound

    assert len(combined.bound_list) == len(area.bound_list) + 1
    assert _contains(combined, [(0.5, 0.25), (2, 2)]).tolist() == [True, False]


def test_area_or_bound_matches_area_add_bound():
    area = geometry.rectangle([0, 1], [0, 1])
    extra_bound = geometry.line_horizontal(0.5, [0, 1])
    added = area + extra_bound
    combined = area | extra_bound
    points = torch.tensor([[0.5, 0.25], [2, 2]], dtype=torch.float32)

    assert len(combined.bound_list) == len(added.bound_list)
    assert torch.equal(
        combined.contains(points[:, 0], points[:, 1]),
        added.contains(points[:, 0], points[:, 1]),
    )


def test_sampling_area_filters_candidates_through_contains():
    area = geometry.rectangle([0, 2], [0, 2]) - geometry.circle(1, 1, 0.5)

    x, y = area.sampling_area([21, 21], scheme="uniform")

    assert area.contains(x, y).all()
    assert not geometry.circle(1, 1, 0.5).contains(x, y).any()


def test_legacy_area_construction_retains_boundary_masking_fallback():
    explicit_area = geometry.rectangle([0, 2], [0, 1])
    legacy_area = geometry.Area(explicit_area.bound_list.copy())
    points = torch.tensor([[1, 0.5], [-1, 0.5], [1, 2]], dtype=torch.float32)

    mask = legacy_area.contains(points[:, 0], points[:, 1])

    assert mask.tolist() == [True, False, False]


def test_contains_rejects_mismatched_shapes():
    area = geometry.rectangle([0, 1], [0, 1])

    with pytest.raises(ValueError, match="same shape"):
        area.contains(torch.zeros(2), torch.zeros(3))


def test_geometry_factories_validate_invalid_shapes():
    with pytest.raises(ValueError, match="positive"):
        geometry.circle(0, 0, 0)

    with pytest.raises(ValueError, match="three vertices"):
        geometry.polygon([0, 0], [1, 0])
