"""Generate the fresh NGSolve FEM reference for cylinder flow."""

import time
from pathlib import Path

import numpy as np

from benchmark_common import build_domain
from common_config import (
    CHANNEL_X, CHANNEL_Y, CYLINDER_CY, CYLINDER_CX, CYLINDER_R,
    FEM_BOUNDARY_RESOLUTION, FEM_DEFAULT_GRID, FEM_MAX_ITERATIONS,
    FEM_MESH_SIZE, FEM_REFERENCE_FILENAME, FEM_TOLERANCE, PROFILE_POINTS,
    RESULTS_DIR, REYNOLDS,
)


def _build_reference_domain():
    return build_domain("uvp")


def _sample_solution(reference, grid):
    x_axis = np.linspace(*CHANNEL_X, grid[0])
    y_axis = np.linspace(*CHANNEL_Y, grid[1])
    query_x, query_y = np.meshgrid(x_axis, y_axis, indexing="ij")
    fields = reference.reference_solution.evaluate(query_x, query_y, fields=("u", "v", "p"))
    x = query_x.reshape(-1)
    y = query_y.reshape(-1)

    epsilon = 1.0e-6
    outlet_y = np.linspace(
        CHANNEL_Y[0] + epsilon, CHANNEL_Y[1] - epsilon, PROFILE_POINTS
    )
    outlet = reference.reference_solution.evaluate(
        np.full_like(outlet_y, CHANNEL_X[1] - epsilon),
        outlet_y,
        fields=("u", "v"),
    )
    wake_x = np.linspace(
        CYLINDER_CX + CYLINDER_R + epsilon,
        CHANNEL_X[1] - epsilon,
        PROFILE_POINTS,
    )
    wake = reference.reference_solution.evaluate(
        wake_x, np.full_like(wake_x, CYLINDER_CY), fields=("u", "v")
    )
    return x, y, fields, outlet_y, outlet, wake_x, wake


def solve_reference(grid=FEM_DEFAULT_GRID, mesh_size=FEM_MESH_SIZE, output_path=None):
    print(f"DeepFlow FEM cylinder reference ({grid[0]} x {grid[1]} samples, Re={REYNOLDS:g})")
    start = time.perf_counter()
    domain = _build_reference_domain()
    reference = domain.solve_fem(
        mesh_size=mesh_size,
        boundary_resolution=FEM_BOUNDARY_RESOLUTION,
        tolerance=FEM_TOLERANCE,
        max_iterations=FEM_MAX_ITERATIONS,
    )
    x, y, fields, outlet_y, outlet, wake_x, wake = _sample_solution(reference, grid)
    metadata = reference.metadata
    residuals = np.asarray(metadata.get("solver_residuals", []), dtype=float)
    mesh = metadata.get("mesh", {})
    payload = {
        "x": x,
        "y": y,
        "u": np.asarray(fields["u"]),
        "v": np.asarray(fields["v"]),
        "p": np.asarray(fields["p"]),
        "outlet_y": outlet_y,
        "outlet_u": np.asarray(outlet["u"]),
        "outlet_v": np.asarray(outlet["v"]),
        "wake_x": wake_x,
        "wake_u": np.asarray(wake["u"]),
        "wake_v": np.asarray(wake["v"]),
        "nx": int(grid[0]),
        "ny": int(grid[1]),
        "reynolds": REYNOLDS,
        "converged": int(metadata["converged"]),
        "iterations": int(metadata["iterations"]),
        "final_residual": float(residuals[-1]),
        "runtime_s": time.perf_counter() - start,
        "mesh_size": mesh_size,
        "mesh_elements": mesh.get("elements", -1),
        "mesh_vertices": mesh.get("vertices", -1),
        "pressure_gauge": np.asarray("outlet_zero"),
        "backend": np.asarray(metadata["backend"]),
    }
    if not payload["converged"]:
        raise RuntimeError("DeepFlow FEM cylinder reference did not converge.")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **payload)
    return payload


if __name__ == "__main__":
    solve_reference(output_path=RESULTS_DIR / FEM_REFERENCE_FILENAME)
