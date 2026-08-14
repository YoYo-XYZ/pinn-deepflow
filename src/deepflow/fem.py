"""FEM evaluation orchestration for :class:`ProblemDomain`."""

from typing import TYPE_CHECKING

from .geometry import Area

if TYPE_CHECKING:
    from .domain import ProblemDomain
    from .evaluation import GroupEvaluator


def solve_fem(
    domain: "ProblemDomain",
    mesh_size=0.05,
    boundary_resolution=128,
    time_step=None,
    tolerance=1e-8,
    max_iterations=200,
    area_sampling_res=None,
    bound_sampling_res=None,
) -> "GroupEvaluator":
    """Solve a domain with the optional NGSolve FEM backend.

    Existing geometry coordinates are reused when sampling resolutions are
    omitted; supplied resolutions use the existing uniform samplers.
    """
    from .evaluation import (
        GroupEvaluator,
        _normalize_resolutions,
        _unique_geometry_entries,
        _validate_sampled_entries,
    )

    bound_entries = _unique_geometry_entries(
        getattr(domain, "bound_list", [])
    )
    area_entries = _unique_geometry_entries(
        getattr(domain, "area_list", [])
    )

    if bound_sampling_res is not None:
        resolutions = _normalize_resolutions(
            bound_sampling_res,
            len(bound_entries),
            "bound_sampling_res",
        )
        for (_, geometry), resolution in zip(bound_entries, resolutions):
            geometry.sampling_line(resolution, scheme="uniform")
            geometry.process_coordinates()

    sampleable_area_entries = [
        (index, geometry)
        for index, geometry in area_entries
        if isinstance(geometry, Area)
    ]
    if area_sampling_res is not None:
        resolutions = _normalize_resolutions(
            area_sampling_res,
            len(sampleable_area_entries),
            "area_sampling_res",
            allow_area_pair=True,
        )
        for (_, geometry), resolution in zip(
            sampleable_area_entries, resolutions
        ):
            geometry.sampling_area(resolution, scheme="uniform")
            geometry.process_coordinates()

    _validate_sampled_entries(bound_entries, area_entries, "solve_fem")

    from .reference import ReferenceSolver

    reference_solution = ReferenceSolver(
        mesh_size=mesh_size,
        boundary_resolution=boundary_resolution,
        time_step=time_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).solve(domain)
    return GroupEvaluator(
        None,
        domain,
        reference_solution=reference_solution,
    )
