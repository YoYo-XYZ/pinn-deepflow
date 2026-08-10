"""Shared configuration for the four-cell cavity benchmark."""

from pathlib import Path
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
CAVITY_X: Tuple[float, float] = (0.0, 1.0)
CAVITY_Y: Tuple[float, float] = (0.0, 1.0)
LID_VELOCITY: float = 1.0

U_INF: float = LID_VELOCITY
L_CHAR: float = 1.0
MU: float = 0.1
RHO: float = 1.0
REYNOLDS: float = RHO * U_INF * L_CHAR / MU

# ---------------------------------------------------------------------------
# Sampling and training
# ---------------------------------------------------------------------------
BOUNDARY_POINTS: List[int] = [50, 50, 50, 50, 1]
INTERIOR_POINTS: List[List[int]] = [[50, 50]]

LR_ADAM: float = 0.004
EPOCHS_ADAM: int = 0

EPOCHS_LBFGS: int = 100
DEFAULT_NUM_RUNS: int = 1

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
EVAL_GRID: List[int] = [50, 50]
CENTERLINE_POINTS: int = 50

# ---------------------------------------------------------------------------
# FEM reference
# ---------------------------------------------------------------------------
FEM_DEFAULT_GRID: Tuple[int, int] = (101, 101)
FEM_MESH_SIZE: float = 0.05
FEM_BOUNDARY_RESOLUTION: int = 128
FEM_MAX_ITERATIONS: int = 200
FEM_TOLERANCE: float = 1.0e-5

FEM_REFERENCE_FILENAME: str = "cfd_reference.npz"

# ---------------------------------------------------------------------------
# Network architectures
# ---------------------------------------------------------------------------
# These are the configurations used by the existing QCPINN-vs-PINN cavity
# benchmark.  The output dimension changes naturally for the PSIP cells.
PINN_WIDTH: int = 48
PINN_LENGTH: int = 4

QC_PRE: List[int] = [32]
QC_POST: List[int] = [32]
QC_NQUBITS: int = 4
QC_ITERATIONS: int = 10

# ---------------------------------------------------------------------------
# Reproducibility and paths
# ---------------------------------------------------------------------------
BASE_SEED: int = 69
SEEDS: List[int] = [BASE_SEED, BASE_SEED + 1, BASE_SEED + 2]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

PINN_UVP_RESULTS_FILE = RESULTS_DIR / "pinn_uvp_results.npz"
PINN_PSIP_RESULTS_FILE = RESULTS_DIR / "pinn_psip_results.npz"
QCPINN_UVP_RESULTS_FILE = RESULTS_DIR / "qcpinn_uvp_results.npz"
QCPINN_PSIP_RESULTS_FILE = RESULTS_DIR / "qcpinn_psip_results.npz"
REPORT_FILE = RESULTS_DIR / "benchmark_report.md"
