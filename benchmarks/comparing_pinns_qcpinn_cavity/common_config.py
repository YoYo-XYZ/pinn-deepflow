"""Shared configuration for the QCPINN vs PINN cavity-flow benchmark."""

from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Problem definition -- geometry
# ---------------------------------------------------------------------------
CAVITY_X: Tuple[float, float] = (0.0, 1.0)
CAVITY_Y: Tuple[float, float] = (0.0, 1.0)
LID_VELOCITY: float = 1.0

# ---------------------------------------------------------------------------
# Problem definition -- PDE (Navier-Stokes)
# ---------------------------------------------------------------------------
# Re = rho * U * L / mu = 1 * 1 * 1 / 0.1 = 10
U_INF: float = LID_VELOCITY
L_CHAR: float = 1.0
MU: float = 0.1
RHO: float = 1.0
REYNOLDS: float = RHO * U_INF * L_CHAR / MU

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
# Four rectangle boundaries plus a small pressure-reference boundary.
BOUNDARY_POINTS: List[int] = [50, 50, 50, 50, 1]
INTERIOR_POINTS: List[List[int]] = [[50, 50]]

# ---------------------------------------------------------------------------
# Resampling and training
# ---------------------------------------------------------------------------
RESAMPLE_EVERY: Optional[int] = None

LR_ADAM: float = 0.004
EPOCHS_ADAM: int = 0
THRESHOLD_ADAM: Optional[float] = None

EPOCHS_LBFGS: int = 100
THRESHOLD_LBFGS: Optional[float] = None

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_GRID: List[int] = [50, 50]
CENTERLINE_POINTS: int = 50

# ---------------------------------------------------------------------------
# Finite-volume CFD reference
# ---------------------------------------------------------------------------
CFD_DEFAULT_GRID: Tuple[int, int] = (101, 101)
CFD_REFINED_GRID: Tuple[int, int] = (201, 201)
CFD_MAX_ITERATIONS: int = 10000
CFD_TOLERANCE: float = 1.0e-8
CFD_LINEAR_TOLERANCE: float = 1.0e-9
CFD_LINEAR_MAX_ITERATIONS: int = 500
CFD_ALPHA_U: float = 0.7
CFD_ALPHA_V: float = 0.7
CFD_ALPHA_P: float = 0.3
CFD_REFERENCE_FILENAME: str = "cfd_reference.npz"
CFD_REFINED_FILENAME: str = "cfd_reference_201.npz"
CFD_GRID_CONVERGENCE_FILENAME: str = "cfd_grid_convergence.npz"

# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------
# QCPINN: pre=[32], post=[32], nqubits=4, q_layer_iterations=10 -> 607 params.
QC_PRE: List[int] = [32]
QC_POST: List[int] = [32]
QC_NQUBITS: int = 4
QC_ITERATIONS: int = 10

# Direct (u, v, p) adaptation with increased capacity for the Re=10 cavity.
PINN_WIDTH: int = 48
PINN_LENGTH: int = 4

# ---------------------------------------------------------------------------
# Reproducibility and paths
# ---------------------------------------------------------------------------
BASE_SEED: int = 69
SEEDS: List[int] = [BASE_SEED, BASE_SEED + 1, BASE_SEED + 2]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

PINN_RESULTS_FILE: Path = RESULTS_DIR / "pinn_results.npz"
QCPINN_RESULTS_FILE: Path = RESULTS_DIR / "qcpinn_results.npz"
REPORT_FILE: Path = RESULTS_DIR / "benchmark_report.md"
