"""Optional FEM reference solutions; NGSolve loads only on ``ReferenceSolver.solve``."""

from .solution import ReferenceSolution
from .solver import (
    ReferenceConfigurationError,
    ReferenceSolver,
    UnsupportedReferencePDE,
)
from .geometry import ReferenceGeometryError

__all__ = [
    "ReferenceConfigurationError",
    "ReferenceGeometryError",
    "ReferenceSolution",
    "ReferenceSolver",
    "UnsupportedReferencePDE",
]
