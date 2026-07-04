"""
Shared configuration and hyperparameters for the 1D Burgers equation benchmark.

This problem matches the setup in ``examples/burgers_eq/burgers_eq.ipynb``:
  - Geometry: rectangle [-1, 1] x [0, 1]
  - 1D Burgers' equation: u_t + u * u_x = nu * u_xx (spatial version uses y as time)
  - nu = 0.01 / pi
  - Network: input=2, output=1 (u), width=16, depth=4 hidden layers, Tanh
  - Optimizer: Adam, lr=0.004, 2000 epochs/iterations
"""

from typing import Tuple

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
X_RANGE: Tuple[float, float] = (-1.0, 1.0)
Y_RANGE: Tuple[float, float] = (0.0, 1.0)
NU: float = 0.01 / 3.14159265358979323846  # nu = 0.01 / pi

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
WIDTH: int = 16
DEPTH: int = 4
ACTIVATION: str = "tanh"

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
LR: float = 0.004
EPOCHS: int = 4000

# ---------------------------------------------------------------------------
# Sampling – boundary points and interior collocation points
# Order: [IC line, left wall, right wall]
# ---------------------------------------------------------------------------
BOUNDARY_POINTS: list = [1000, 500, 500]
INTERIOR_POINTS: list = [4000]

# ---------------------------------------------------------------------------
# Evaluation grid resolution (nx, ny)
# ---------------------------------------------------------------------------
EVAL_GRID: list = [500, 250]

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
SEED: int = 69
