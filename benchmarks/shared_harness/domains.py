"""Shared domain builders for Burgers, channel, cavity, and cylinder cases.

All builders use the public DeepFlow API only (geometry, domain,
define_bc/define_pde, sampling calls). The one shared perimeter-weighting
rule lives in :func:`perimeter_weighted_boundary_counts`.
"""

from __future__ import annotations

import math
import sys
from numbers import Integral, Real
from pathlib import Path
from typing import List, Optional, Union

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import deepflow as df  # noqa: E402
from torch import pi, sin  # noqa: E402

InteriorRes = Union[int, List[int]]

_BURGERS_X = [-1.0, 1.0]
_BURGERS_Y = [0.0, 1.0]
_CHANNEL_DEFAULT_BOUNDARY_TOTAL = 1200


def perimeter_weighted_boundary_counts(
    total_points: int, lx: float, ly: float
) -> List[int]:
    """Distribute boundary points proportional to side length.

    Shared rule replacing per-suite clones: for a rectangular channel the
    bound order is ``[left, bottom, right, top]`` with side lengths
    ``[Ly, Lx, Ly, Lx]`` over perimeter ``2 * (Lx + Ly)``.

    Args:
        total_points: Total boundary budget to distribute.
        lx: Channel length in x.
        ly: Channel height in y.

    Returns:
        Four integer counts in ``[left, bottom, right, top]`` order.
    """
    if isinstance(total_points, bool) or not isinstance(total_points, Integral):
        raise ValueError(f"total_points must be a positive int, got {total_points!r}")
    if total_points <= 0:
        raise ValueError(f"total_points must be positive, got {total_points!r}")
    if (
        isinstance(lx, bool)
        or not isinstance(lx, Real)
        or not math.isfinite(float(lx))
        or isinstance(ly, bool)
        or not isinstance(ly, Real)
        or not math.isfinite(float(ly))
        or lx <= 0
        or ly <= 0
    ):
        raise ValueError(f"lx and ly must be positive, got {(lx, ly)!r}")
    perimeter = 2.0 * (lx + ly)
    side_lengths = [float(ly), float(lx), float(ly), float(lx)]
    raw_counts = [total_points * length / perimeter for length in side_lengths]
    counts = [math.floor(value) for value in raw_counts]
    remainder = int(total_points) - sum(counts)
    fractions = [value - math.floor(value) for value in raw_counts]
    for index in sorted(range(len(counts)), key=lambda i: (-fractions[i], i))[:remainder]:
        counts[index] += 1
    return counts


def _sampling_counts(values):
    if isinstance(values, Integral) and not isinstance(values, bool):
        return [int(values)]
    try:
        return list(values)
    except TypeError as exc:
        raise ValueError(f"sampling resolutions must be a list, got {values!r}") from exc


def _apply_sampling(domain, boundary_points, interior_points, sampling: str):
    boundary_points = _sampling_counts(boundary_points)
    interior_points = _sampling_counts(interior_points)
    if sampling == "uniform":
        domain.sampling_uniform(list(boundary_points), list(interior_points))
    elif sampling == "random":
        domain.sampling_random(list(boundary_points), list(interior_points))
    elif sampling == "lhs":
        domain.sampling_lhs(list(boundary_points), list(interior_points))
    else:
        raise ValueError(f"Unknown sampling {sampling!r}")
    return domain


def build_burgers_domain(
    nu: float = 0.01 / math.pi,
    boundary_points: Optional[List[int]] = None,
    interior_points: Optional[List[InteriorRes]] = None,
    sampling: str = "lhs",
):
    """Build the shared 1D-Burgers domain (x in [-1, 1], y/time in [0, 1])."""
    boundary_points = [1000, 500, 500] if boundary_points is None else boundary_points
    interior_points = [4000] if interior_points is None else interior_points
    area = df.geometry.rectangle(list(_BURGERS_X), list(_BURGERS_Y))
    line_ic = df.geometry.line_horizontal(y=_BURGERS_Y[0], range_x=list(_BURGERS_X))
    line_left = df.geometry.line_vertical(x=_BURGERS_X[0], range_y=list(_BURGERS_Y))
    line_right = df.geometry.line_vertical(x=_BURGERS_X[1], range_y=list(_BURGERS_Y))
    domain = df.domain(area.area_list, line_ic, line_left, line_right)
    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=nu))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})
    return _apply_sampling(domain, boundary_points, interior_points, sampling)


def build_channel_domain(
    lx: float = 5.0,
    ly: float = 1.0,
    boundary_points: Optional[List[int]] = None,
    interior_points: Optional[List[InteriorRes]] = None,
    sampling: str = "random",
    U: float = 0.0001,
    L: float = 1.0,
    mu: float = 0.001,
    rho: float = 1000.0,
):
    """Build the shared steady channel domain with perimeter-weighted counts."""
    if boundary_points is None:
        boundary_points = perimeter_weighted_boundary_counts(
            _CHANNEL_DEFAULT_BOUNDARY_TOTAL, lx, ly
        )
    interior_points = [2000] if interior_points is None else interior_points
    domain = df.domain(df.geometry.rectangle([0.0, lx], [0.0, ly]))
    domain.bound_list[0].define_bc({"u": 1, "v": 0})
    domain.bound_list[1].define_bc({"u": 0, "v": 0})
    domain.bound_list[2].define_bc({"p": 0})
    domain.bound_list[3].define_bc({"u": 0, "v": 0})
    domain.area_list[0].define_pde(
        df.pde.NavierStokes(U=U, L=L, mu=mu, rho=rho)
    )
    return _apply_sampling(domain, boundary_points, interior_points, sampling)


def build_flow_pde(
    formulation: str,
    *,
    U: float,
    L: float,
    mu: float,
    rho: float,
):
    if formulation == "uvp":
        return df.pde.NavierStokes(U=U, L=L, mu=mu, rho=rho)
    if formulation == "psip":
        return df.pde.StreamFunctionNavierStokes(U=U, L=L, mu=mu, rho=rho)
    raise ValueError(f"Unknown formulation: {formulation!r}")


def build_cavity_domain(
    formulation: str = "uvp",
    boundary_points: Optional[List[int]] = None,
    interior_points: Optional[List[InteriorRes]] = None,
    sampling: str = "uniform",
    u_inf: float = 1.0,
    L: float = 1.0,
    mu: float = 0.1,
    rho: float = 1.0,
):
    """Build the shared lid-driven cavity domain (unit square + pressure point)."""
    boundary_points = (
        [50, 50, 50, 50, 1] if boundary_points is None else boundary_points
    )
    interior_points = [[50, 50]] if interior_points is None else interior_points
    rectangle = df.geometry.rectangle([0.0, 1.0], [0.0, 1.0])
    pressure_point = df.geometry.point(0.0, 0.0)
    domain = df.domain(rectangle, pressure_point)
    domain.area_list[0].define_pde(
        build_flow_pde(formulation, U=u_inf, L=L, mu=mu, rho=rho)
    )
    if formulation == "uvp":
        for index in (0, 1, 2):
            domain.bound_list[index].define_bc({"u": 0, "v": 0})
        domain.bound_list[3].define_bc({"u": u_inf, "v": 0})
    elif formulation == "psip":
        for index in (0, 1, 2):
            domain.bound_list[index].define_bc({"psi_x": 0, "psi_y": 0})
        domain.bound_list[3].define_bc({"psi_x": 0, "psi_y": u_inf})
    else:
        raise ValueError(f"Unknown formulation: {formulation!r}")
    domain.bound_list[4].define_bc({"p": 0})
    return _apply_sampling(domain, boundary_points, interior_points, sampling)


def build_cylinder_domain(
    formulation: str = "uvp",
    boundary_points: Optional[List[int]] = None,
    interior_points: Optional[List[InteriorRes]] = None,
    sampling: str = "uniform",
    channel_x: tuple = (0.0, 1.1),
    channel_y: tuple = (0.0, 0.41),
    cylinder: tuple = (0.2, 0.2, 0.05),
    u_inf: float = 1.0,
    L: float = 1.0,
    mu: float = 0.1,
    rho: float = 1.0,
):
    """Build the shared channel-with-cylinder-obstacle domain."""
    boundary_points = (
        [50, 50, 50, 50, 50, 50] if boundary_points is None else boundary_points
    )
    interior_points = [[50, 50]] if interior_points is None else interior_points
    cx, cy, radius = cylinder
    y_min, y_max = channel_y
    if channel_x[1] <= channel_x[0] or y_max <= y_min or radius <= 0:
        raise ValueError("channel ranges must increase and cylinder radius must be positive")
    height = channel_y[1] - channel_y[0]
    circle = df.geometry.circle(cx, cy, radius)
    rectangle = df.geometry.rectangle(list(channel_x), list(channel_y))
    domain = df.domain(rectangle - circle, circle.bound_list)
    domain.area_list[0].define_pde(
        build_flow_pde(formulation, U=u_inf, L=L, mu=mu, rho=rho)
    )
    inlet_u = [
        "y",
        lambda y, _y0=y_min, _h=height, _u=u_inf: (
            4 * _u * (y - _y0) * (_y0 + _h - y) / _h**2
        ),
    ]
    if formulation == "uvp":
        domain.bound_list[0].define_bc({"u": inlet_u, "v": 0})
        for index in (1, 3, 4, 5):
            domain.bound_list[index].define_bc({"u": 0, "v": 0})
    elif formulation == "psip":
        domain.bound_list[0].define_bc({"psi_x": 0, "psi_y": inlet_u})
        for index in (1, 3, 4, 5):
            domain.bound_list[index].define_bc({"psi_x": 0, "psi_y": 0})
    else:
        raise ValueError(f"Unknown formulation: {formulation!r}")
    domain.bound_list[2].define_bc({"p": 0})
    return _apply_sampling(domain, boundary_points, interior_points, sampling)
