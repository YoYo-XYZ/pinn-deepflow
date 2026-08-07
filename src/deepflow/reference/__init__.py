"""Optional NGSolve reference solutions for DeepFlow problems.

The module itself is dependency-light.  NGSolve is imported only when
``ReferenceSolver.solve`` is called.
"""

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
