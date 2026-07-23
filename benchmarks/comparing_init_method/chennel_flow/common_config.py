"""
Shared configuration and hyperparameters for the 2D steady channel-flow
initialization benchmark.

The geometry, PDE coefficients, network, and optimizer follow
``static/quickstart/code.ipynb``, but this standalone benchmark is not an exact
notebook reproduction. In particular, ``compare_init.py`` converts the total
boundary budget below to perimeter-weighted random counts and uses a smaller
evaluation grid.
  - Geometry: rectangle [0, Lx] x [0, Ly]
  - 2D steady incompressible Navier-Stokes, nondimensional Re = 100
  - Network: input=2, output=3 (u,v,p), width=32, depth=4 hidden layers, Tanh
  - Optimizer: Adam, lr=0.004, 2000 epochs/iterations

The seed is reset before each initializer so both runs use the same sampled
domain and training randomness.
"""

# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------
Lx: float = 5.0
Ly: float = 1.0

# ---------------------------------------------------------------------------
# Reynolds number (nondimensional)
# ---------------------------------------------------------------------------
Re: float = 100.0  # implied by the PDE coefficients used in compare_init.py

# ---------------------------------------------------------------------------
# Neural network architecture
# ---------------------------------------------------------------------------
WIDTH: int = 32  # neurons per hidden layer
DEPTH: int = 4  # number of hidden layers
ACTIVATION: str = "tanh"  # descriptive metadata; DeepFlow's PINN uses Tanh

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------
LR: float = 0.004  # Adam learning rate
EPOCHS: int = 2000  # training iterations / epochs

# ---------------------------------------------------------------------------
# Sampling – total boundary budget and interior collocation points
# The benchmark redistributes the boundary total by side length, then samples
# all points once with the random scheme. Actual counts are [100, 500, 100, 500].
# ---------------------------------------------------------------------------
BOUNDARY_POINTS: list = [200, 400, 200, 400]
INTERIOR_POINTS: int = 2000

# ---------------------------------------------------------------------------
# Uniform visualization/residual grid resolution (nx, ny); not the training set
# ---------------------------------------------------------------------------
EVAL_GRID: list = [200, 40]

# ---------------------------------------------------------------------------
# Random seed for reproducibility
# ---------------------------------------------------------------------------
SEED: int = 69

BENCHMARK_METADATA = {
    "Sampling protocol": "One random draw: boundary budget "
    f"{BOUNDARY_POINTS} -> perimeter-weighted [100, 500, 100, 500], "
    f"interior = {INTERIOR_POINTS}; no resampling",
    "Loss evaluation": f"Best-model losses on the fixed training domain; fields/residuals on uniform grid {EVAL_GRID}",
    "Seed protocol": f"Seed {SEED} reset before each initializer (paired comparison)",
}
