"""Boundary smoke for shared domain builders (no suite modified)."""

import numpy as np

import deepflow as df
from benchmarks.shared_harness.domains import (
    build_burgers_domain,
    build_cavity_domain,
    build_channel_domain,
    build_cylinder_domain,
    perimeter_weighted_boundary_counts,
)


def _assert_sampled_finite(domain):
    assert domain.bound_list and domain.area_list
    for geometry in (*domain.bound_list, *domain.area_list):
        assert geometry.X is not None and len(geometry.X) > 0
    value = float(
        df.calc_loss_simple(domain)(
            df.PINN(
                input_vars=["x", "y"],
                output_vars=["u", "v", "p"],
                width=4,
                length=1,
            )
        )["total_loss"]
        .detach()
        .cpu()
        .item()
    )
    assert np.isfinite(value)


def test_perimeter_rule_matches_cloned_channel_counts():
    assert perimeter_weighted_boundary_counts(1200, 5.0, 1.0) == [100, 500, 100, 500]


def test_burgers_builder_smoke():
    domain = build_burgers_domain(
        boundary_points=[8, 4, 4], interior_points=[16], sampling="lhs"
    )
    assert len(domain.bound_list) == 3
    value = float(
        df.calc_loss_simple(domain)(
            df.PINN(input_vars=["x", "y"], output_vars=["u"], width=4, length=1)
        )["total_loss"]
        .detach()
        .cpu()
        .item()
    )
    assert np.isfinite(value)


def test_channel_builder_smoke():
    domain = build_channel_domain(
        boundary_points=[4, 4, 4, 4], interior_points=[16], sampling="random"
    )
    _assert_sampled_finite(domain)


def test_cavity_builders_smoke():
    for formulation in ("uvp", "psip"):
        domain = build_cavity_domain(
            formulation=formulation,
            boundary_points=[4, 4, 4, 4, 1],
            interior_points=[[4, 4]],
            sampling="uniform",
        )
        assert len(domain.bound_list) == 5
        assert domain.area_list[0].PDE is not None


def test_cylinder_builders_smoke():
    for formulation in ("uvp", "psip"):
        domain = build_cylinder_domain(
            formulation=formulation,
            boundary_points=[4, 4, 4, 4, 4, 4],
            interior_points=[[4, 4]],
            sampling="uniform",
        )
        assert len(domain.bound_list) == 6
        assert domain.area_list[0].PDE is not None
