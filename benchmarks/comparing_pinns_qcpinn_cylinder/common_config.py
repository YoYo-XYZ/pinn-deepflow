"""Shared configuration for the four-cell cylinder-flow benchmark."""

from pathlib import Path
from typing import List, Tuple

# Geometry and physics copied from examples/cylinder_flow_steady.
CHANNEL_X: Tuple[float, float] = (0.0, 1.1)
CHANNEL_Y: Tuple[float, float] = (0.0, 0.41)
CYLINDER_CX: float = 0.2
CYLINDER_CY: float = 0.2
CYLINDER_R: float = 0.05

U_INF: float = 1.0
L_CHAR: float = 1.0
MU: float = 0.1  # Re = rho*U*L/mu = 10
RHO: float = 1.0
REYNOLDS: float = RHO * U_INF * L_CHAR / MU

# Match the cavity benchmark workload.
BOUNDARY_POINTS: List[int] = [50, 50, 50, 50, 50, 50]
INTERIOR_POINTS: List[List[int]] = [[50, 50]]
LR_ADAM: float = 0.004
EPOCHS_ADAM: int = 0
EPOCHS_LBFGS: int = 100
DEFAULT_NUM_RUNS: int = 1

EVAL_GRID: List[int] = [50, 50]
PROFILE_POINTS: int = 50

# FEM reference settings follow the cavity benchmark format.
FEM_DEFAULT_GRID: Tuple[int, int] = (101, 101)
FEM_MESH_SIZE: float = 0.05
FEM_BOUNDARY_RESOLUTION: int = 128
FEM_MAX_ITERATIONS: int = 200
FEM_TOLERANCE: float = 1.0e-5
FEM_REFERENCE_FILENAME: str = "cfd_reference.npz"

# Exact network configuration used by the cavity benchmark.
PINN_WIDTH: int = 48
PINN_LENGTH: int = 4
QC_PRE: List[int] = [32]
QC_POST: List[int] = [32]
QC_NQUBITS: int = 4
QC_ITERATIONS: int = 10

BASE_SEED: int = 69
SEEDS: List[int] = [BASE_SEED]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
PINN_UVP_RESULTS_FILE = RESULTS_DIR / "pinn_uvp_results.npz"
PINN_PSIP_RESULTS_FILE = RESULTS_DIR / "pinn_psip_results.npz"
QCPINN_UVP_RESULTS_FILE = RESULTS_DIR / "qcpinn_uvp_results.npz"
QCPINN_PSIP_RESULTS_FILE = RESULTS_DIR / "qcpinn_psip_results.npz"
REPORT_FILE = RESULTS_DIR / "benchmark_report.md"
