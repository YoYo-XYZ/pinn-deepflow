"""Shared configuration replacing per-suite config clones.

Covers the knobs suites currently clone: network size (width/depth),
learning rate, epochs (Adam/L-BFGS), seeds, and point counts
(boundary/interior/evaluation). Physics parameters stay with the domain
builders; this object holds only training and sampling knobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Union

InteriorRes = Union[int, List[int]]
SAMPLING_CHOICES = ("uniform", "random", "lhs")


def _check_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive int, got {value!r}")


def _check_point_counts(name: str, values: list) -> None:
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list, got {values!r}")
    for entry in values:
        if isinstance(entry, bool):
            raise ValueError(f"{name} entries must be positive, got {entry!r}")
        if isinstance(entry, int):
            if entry <= 0:
                raise ValueError(f"{name} entries must be positive, got {entry!r}")
        elif isinstance(entry, (list, tuple)):
            if len(entry) != 2 or not all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in entry
            ):
                raise ValueError(
                    f"{name} grid entries must be [nx, ny] of positive ints, "
                    f"got {entry!r}"
                )
        else:
            raise ValueError(f"{name} entries must be int or [nx, ny], got {entry!r}")


@dataclass
class BenchmarkConfig:
    """Single config object for benchmark suites.

    Attributes:
        width: Neurons per hidden layer.
        depth: Number of hidden layers.
        learning_rate: Adam learning rate.
        epochs_adam: Adam epochs (0 skips Adam).
        epochs_lbfgs: L-BFGS epochs (0 skips L-BFGS).
        seed: Manual seed for sampling and model init.
        boundary_points: Points per boundary, in domain bound order.
        interior_points: Points per area (int counts or [nx, ny] grids).
        eval_grid: Evaluation resolution as [nx, ny].
        sampling: Initial sampler: ``uniform``, ``random``, or ``lhs``.
    """

    width: int = 16
    depth: int = 4
    learning_rate: float = 0.004
    epochs_adam: int = 2
    epochs_lbfgs: int = 0
    seed: int = 69
    boundary_points: List[int] = field(default_factory=lambda: [8, 4, 4])
    interior_points: Union[int, List[InteriorRes]] = field(default_factory=lambda: [16])
    eval_grid: List[int] = field(default_factory=lambda: [8, 8])
    sampling: str = "lhs"

    def __post_init__(self) -> None:
        _check_positive_int("width", self.width)
        _check_positive_int("depth", self.depth)
        if not isinstance(self.learning_rate, (int, float)) or not (
            self.learning_rate > 0
        ):
            raise ValueError(
                f"learning_rate must be positive, got {self.learning_rate!r}"
            )
        for name in ("epochs_adam", "epochs_lbfgs"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError(f"seed must be a non-negative int, got {self.seed!r}")
        _check_point_counts("boundary_points", list(self.boundary_points))
        if isinstance(self.interior_points, bool):
            raise ValueError(
                "interior_points entries must be positive, "
                f"got {self.interior_points!r}"
            )
        if isinstance(self.interior_points, int):
            self.interior_points = [self.interior_points]
        _check_point_counts("interior_points", list(self.interior_points))
        if (
            not isinstance(self.eval_grid, (list, tuple))
            or len(self.eval_grid) != 2
            or not all(
                isinstance(v, int) and not isinstance(v, bool) and v > 0
                for v in self.eval_grid
            )
        ):
            raise ValueError(
                f"eval_grid must be [nx, ny] of positive ints, got {self.eval_grid!r}"
            )
        if self.sampling not in SAMPLING_CHOICES:
            raise ValueError(
                f"sampling must be one of {SAMPLING_CHOICES}, got {self.sampling!r}"
            )

    def to_dict(self) -> dict:
        """Return a plain-dict view for reports."""
        return {
            "width": self.width,
            "depth": self.depth,
            "learning_rate": self.learning_rate,
            "epochs_adam": self.epochs_adam,
            "epochs_lbfgs": self.epochs_lbfgs,
            "seed": self.seed,
            "boundary_points": list(self.boundary_points),
            "interior_points": list(self.interior_points),
            "eval_grid": list(self.eval_grid),
            "sampling": self.sampling,
        }
