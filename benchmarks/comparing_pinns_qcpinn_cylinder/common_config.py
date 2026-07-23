"""
Shared configuration for the QCPINN vs PINN cylinder flow benchmark.

Problem: 2D steady incompressible flow around a circular cylinder at Re=50,
following the setup in `examples/cylinder_flow_steady/`.
"""

from pathlib import Path
from typing import Tuple, List

# ---------------------------------------------------------------------------
# Problem definition — geometry
# ---------------------------------------------------------------------------
# Channel: rectangle [0, CHANNEL_X[1]] x [0, CHANNEL_Y[1]]
CHANNEL_X: Tuple[float, float] = (0.0, 1.1)
CHANNEL_Y: Tuple[float, float] = (0.0, 0.41)
# Cylinder: center (CYLINDER_CX, CYLINDER_CY), radius CYLINDER_R
CYLINDER_CX: float = 0.2
CYLINDER_CY: float = 0.2
CYLINDER_R: float = 0.05

# ---------------------------------------------------------------------------
# Problem definition — PDE (Navier-Stokes)
# ---------------------------------------------------------------------------
# Characteristic velocity, length, dynamic viscosity, density.
# Re = rho * U * L / mu = 1 * 1 * 1 / 0.02 = 50
U_INF: float = 1.0
L_CHAR: float = 1.0
MU: float = 0.02
RHO: float = 1.0
REYNOLDS: float = RHO * U_INF * L_CHAR / MU  # 50.0

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
# LHS initial sampling — 1000 points per boundary, 4000 interior.
BOUNDARY_POINTS: List[int] = [1000, 1000, 1000, 1000, 1000, 1000]
INTERIOR_POINTS: List[int] = [4000]

# ---------------------------------------------------------------------------
# Resampling (periodic full LHS resampling)
# ---------------------------------------------------------------------------
RESAMPLE_EVERY: int = 100  # epochs between resamples

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LR_ADAM: float = 0.004
EPOCHS_ADAM: int = 2000
THRESHOLD_ADAM: float = 0.01

EPOCHS_LBFGS: int = 500
THRESHOLD_LBFGS: float = 0.0001

# ---------------------------------------------------------------------------
# Evaluation grid
# ---------------------------------------------------------------------------
EVAL_GRID: List[int] = [300, 150]   # uniform grid for area evaluation
OUTLET_LINE_POINTS: int = 200         # points along the outlet boundary

# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------
# QCPINN: pre=[50], post=[50], nqubits=4, q_layer_iterations=1
#   Pre:  Linear(2,50)=150 + Linear(50,4)=204 = 354
#   Quantum: weights(1,3,4) = 12
#   Post: Linear(4,50)=250 + Linear(50,3)=153 = 403
#   Total = 769 parameters
QC_PRE: List[int] = [50]
QC_POST: List[int] = [50]
QC_NQUBITS: int = 4
QC_ITERATIONS: int = 1

# Classical PINN: width=18, length=3 (parameter-matched to ~795)
#   Linear(2,18)=54 + 2*Linear(18,18)=684 + Linear(18,3)=57 = 795
PINN_WIDTH: int = 18
PINN_LENGTH: int = 3

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
BASE_SEED: int = 69
SEEDS: List[int] = [BASE_SEED, BASE_SEED + 1, BASE_SEED + 2]  # [69, 70, 71]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

PINN_RESULTS_FILE: Path = RESULTS_DIR / "pinn_results.npz"
QCPINN_RESULTS_FILE: Path = RESULTS_DIR / "qcpinn_results.npz"
REPORT_FILE: Path = RESULTS_DIR / "benchmark_report.md"
