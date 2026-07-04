"""
Shared configuration and hyperparameters for the 2D steady channel flow benchmark.

This problem matches the setup in ``static/quickstart/code.ipynb``:
  - Geometry: rectangle [0, Lx] x [0, Ly]
  - 2D steady incompressible Navier-Stokes, nondimensional Re = 100
  - Network: input=2, output=3 (u,v,p), width=32, depth=4 hidden layers, Tanh
  - Optimizer: Adam, lr=0.004, 2000 epochs/iterations
"""

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
Lx: float = 5.0
Ly: float = 1.0

# ---------------------------------------------------------------------------
# Reynolds number (nondimensional)
# ---------------------------------------------------------------------------
Re: float = 100.0

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
WIDTH: int = 32          # neurons per hidden layer
DEPTH: int = 4           # number of hidden layers
ACTIVATION: str = "tanh" # activation function string (for DeepXDE)

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
LR: float = 0.004        # Adam learning rate
EPOCHS: int = 2000       # training iterations / epochs

# ---------------------------------------------------------------------------
# Sampling – boundary points per side and interior collocation points
# Order: [left, bottom, right, top]
# ---------------------------------------------------------------------------
BOUNDARY_POINTS: list = [200, 400, 200, 400]
INTERIOR_POINTS: int = 2000

# ---------------------------------------------------------------------------
# Evaluation grid resolution (nx, ny)
# ---------------------------------------------------------------------------
EVAL_GRID: list = [500, 100]

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
SEED: int = 69
