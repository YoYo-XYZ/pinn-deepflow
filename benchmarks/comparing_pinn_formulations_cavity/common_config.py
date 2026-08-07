"""Shared configuration for the direct and stream-function cavity benchmark."""

from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Problem definition -- geometry and physical parameters
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
# Sampling and training -- identical to the existing Re=10 cavity benchmark
# ---------------------------------------------------------------------------
BOUNDARY_POINTS: List[int] = [50, 50, 50, 50, 1]
INTERIOR_POINTS: List[List[int]] = [[50, 50]]

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
# Parameter-matched model configurations
# ---------------------------------------------------------------------------
PINN_WIDTH: int = 48
PINN_LENGTH: int = 4

# A 2-output FNN with [48, 48, 48, 49] hidden layers has 7,349 parameters,
# versus 7,347 for the direct 3-output PINN.  This is the closest simple
# parameter match while retaining the existing direct model capacity.
STREAM_HIDDEN_LAYERS: List[int] = [48, 48, 48, 49]

# ---------------------------------------------------------------------------
# Reproducibility and paths
# ---------------------------------------------------------------------------
BASE_SEED: int = 69
SEEDS: List[int] = [BASE_SEED, BASE_SEED + 1, BASE_SEED + 2]

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"

UVP_RESULTS_FILE: Path = RESULTS_DIR / "uvp_results.npz"
PSIP_RESULTS_FILE: Path = RESULTS_DIR / "psip_results.npz"
REPORT_FILE: Path = RESULTS_DIR / "benchmark_report.md"

# The CFD solver and its reference output already belong to the existing
# cavity benchmark.  Both formulation benchmarks use this same reference.
CFD_REFERENCE_FILE: Path = (
    SCRIPT_DIR.parent
    / "comparing_pinns_qcpinn_cavity"
    / "results"
    / "cfd_reference.npz"
)
