"""
Shared configuration for the FP32 vs FP64 2D channel-flow benchmark.

Problem: steady incompressible Navier-Stokes in a rectangular channel.
"""

from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
X_RANGE: Tuple[float, float] = (0.0, 5.0)
Y_RANGE: Tuple[float, float] = (0.0, 1.0)
U: float = 0.0001
L: float = 1.0
MU: float = 0.001
RHO: float = 1000.0

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
WIDTH: int = 32
DEPTH: int = 4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
EPOCHS: int = 200
SEED: int = 69

# ---------------------------------------------------------------------------
# Sampling
# Order: [left inlet, bottom wall, right outlet, top wall]
# ---------------------------------------------------------------------------
BOUNDARY_POINTS: list[int] = [200, 400, 200, 400]
INTERIOR_POINTS: list[int] = [2000]

# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------
EVAL_GRID: list[int] = [500, 100]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
