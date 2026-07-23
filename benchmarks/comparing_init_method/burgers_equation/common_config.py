"""
Shared configuration and hyperparameters for the 1D Burgers equation
initialization benchmark.

The problem definition follows ``examples/burgers_eq/burgers_eq.ipynb``, but
this standalone benchmark is not an exact notebook reproduction:
  - Geometry: rectangle [-1, 1] x [0, 1]
  - 1D Burgers' equation: u_t + u * u_x = nu * u_xx (spatial version uses y as time)
  - nu = 0.01 / pi
  - Network: input=2, output=1 (u), width=16, depth=4 hidden layers, Tanh
  - Optimizer: Adam, lr=0.004, 4000 epochs/iterations

Sampling is one Latin-hypercube draw at the start of each run with the counts
below; the notebook's adaptive R3 resampling callback is not used here. The
seed is reset before each initializer so the two runs share the same sampled
domain.
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
ACTIVATION: str = "tanh"  # descriptive metadata; DeepFlow's PINN uses Tanh

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
LR: float = 0.004
EPOCHS: int = 4000

# ---------------------------------------------------------------------------
# Sampling – one LHS draw at run start; no adaptive resampling
# Order: [IC line, left wall, right wall], then [interior]
# ---------------------------------------------------------------------------
BOUNDARY_POINTS: list = [1000, 500, 500]
INTERIOR_POINTS: list = [4000]

# ---------------------------------------------------------------------------
# Uniform visualization/residual grid resolution (nx, ny); not the training set
# ---------------------------------------------------------------------------
EVAL_GRID: list = [500, 250]

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
SEED: int = 69

BENCHMARK_METADATA = {
    "Sampling protocol": "One LHS draw: boundary [IC, left, right] = "
    f"{BOUNDARY_POINTS}, interior = {INTERIOR_POINTS}; no R3 resampling",
    "Loss evaluation": f"Best-model losses on the fixed training domain; fields/residuals on uniform grid {EVAL_GRID}",
    "Seed protocol": f"Seed {SEED} reset before each initializer (paired comparison)",
}
