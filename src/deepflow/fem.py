"""FEM evaluation orchestration for :class:`ProblemDomain`."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domain import ProblemDomain
    from .reference import ReferenceSolution


def solve_fem(
    domain: "ProblemDomain",
    mesh_size=0.05,
    boundary_resolution=128,
    time_step=None,
    tolerance=1e-8,
    max_iterations=200,
) -> "ReferenceSolution":
    """Solve a domain with the optional NGSolve FEM backend."""
    from .reference import ReferenceSolver

    return ReferenceSolver(
        mesh_size=mesh_size,
        boundary_resolution=boundary_resolution,
        time_step=time_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).solve(domain)
