"""FEM evaluation orchestration for ``ProblemDomain``."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .domain import ProblemDomain
    from .evaluation import ReferenceGroupEvaluator


def solve_fem(
    domain: "ProblemDomain",
    mesh_size=0.05,
    boundary_resolution=128,
    time_step=None,
    tolerance=1e-8,
    max_iterations=200,
) -> "ReferenceGroupEvaluator":
    """Solve with optional NGSolve FEM; return a ``ReferenceGroupEvaluator``.

    Args:
        domain: Domain with exactly one PDE area.
        mesh_size: Target mesh size; ``None`` infers it from the area.
        boundary_resolution: Samples per mesh boundary.
        time_step: Transient snapshot spacing.
        tolerance: Solver convergence tolerance.
        max_iterations: Solver iteration limit.

    Missing NGSolve/Netgen raises ``ImportError``. Invalid input raises
    ``ValueError``.
    """
    from .evaluation import ReferenceGroupEvaluator
    from .reference import ReferenceSolver

    reference = ReferenceSolver(
        mesh_size=mesh_size,
        boundary_resolution=boundary_resolution,
        time_step=time_step,
        tolerance=tolerance,
        max_iterations=max_iterations,
    ).solve(domain)
    return ReferenceGroupEvaluator(reference, domain)
