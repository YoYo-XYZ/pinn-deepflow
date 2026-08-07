#!/usr/bin/env python3
"""Generate fresh CFD references for the combined cavity benchmark."""

import importlib.util
from pathlib import Path

from common_config import (
    CFD_DEFAULT_GRID,
    CFD_GRID_CONVERGENCE_FILENAME,
    CFD_REFINED_GRID,
    CFD_REFERENCE_FILENAME,
    CFD_REFINED_FILENAME,
    RESULTS_DIR,
)


_SCRIPT_DIR = Path(__file__).resolve().parent
_REFERENCE_SOURCE = (
    _SCRIPT_DIR.parent / "comparing_pinns_qcpinn_cavity" / "reference_cfd.py"
)


def _load_reference_solver():
    """Load the shared finite-volume solver without copying its implementation."""
    spec = importlib.util.spec_from_file_location(
        "_combined_cavity_reference_cfd", _REFERENCE_SOURCE
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load CFD solver from {_REFERENCE_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def generate_reference():
    solver = _load_reference_solver()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    coarse_path = RESULTS_DIR / CFD_REFERENCE_FILENAME
    refined_path = RESULTS_DIR / CFD_REFINED_FILENAME
    convergence_path = RESULTS_DIR / CFD_GRID_CONVERGENCE_FILENAME

    coarse = solver.run_reference(
        CFD_DEFAULT_GRID[0], CFD_DEFAULT_GRID[1], coarse_path
    )
    refined = solver.run_reference(
        CFD_REFINED_GRID[0],
        CFD_REFINED_GRID[1],
        refined_path,
        initial_result=coarse,
    )
    if not bool(coarse["converged"]) or not bool(refined["converged"]):
        raise RuntimeError("One or more CFD reference grids did not converge.")
    solver.save_grid_convergence(coarse_path, refined_path, convergence_path)


if __name__ == "__main__":
    generate_reference()
