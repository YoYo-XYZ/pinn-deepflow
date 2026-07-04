#!/usr/bin/env python3
"""
Common configuration for the 1D Burgers equation benchmark.

Shared between benchmark_burgers.py and compare_versions.py so that
hyperparameters stay consistent across versions of the DeepFlow framework.
"""

from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Problem definition
# ---------------------------------------------------------------------------
X_RANGE: Tuple[float, float] = (-1.0, 1.0)
Y_RANGE: Tuple[float, float] = (0.0, 1.0)
NU: float = 0.01 / 3.14159265358979323846  # nu = 0.01 / pi

# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------
WIDTH: int = 16
DEPTH: int = 4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
LR: float = 0.004
EPOCHS: int = 2000
SEED: int = 69

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
# LHS boundary samples: [interior points on IC line, left wall, right wall]
BOUNDARY_POINTS: list[int] = [1000, 500, 500]
INTERIOR_POINTS: list[int] = [4000]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OLD_RESULTS_FILE = RESULTS_DIR / "burgers_benchmark_old.npz"
NEW_RESULTS_FILE = RESULTS_DIR / "burgers_benchmark_new.npz"
REPORT_FILE = RESULTS_DIR / "benchmark_report.md"
LOSS_CURVES_FILE = RESULTS_DIR / "loss_curves.png"
