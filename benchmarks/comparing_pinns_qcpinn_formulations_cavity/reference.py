#!/usr/bin/env python3
"""Generate DeepFlow FEM references for the combined cavity benchmark."""

import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from benchmark_common import build_domain, df
from common_config import (
    CAVITY_X,
    CAVITY_Y,
    CENTERLINE_POINTS,
    FEM_BOUNDARY_RESOLUTION,
    FEM_COARSE_MESH_SIZE,
    FEM_DEFAULT_GRID,
    FEM_GRID_CONVERGENCE_FILENAME,
    FEM_MAX_ITERATIONS,
    FEM_REFINED_GRID,
    FEM_REFINED_MESH_SIZE,
    FEM_REFERENCE_FILENAME,
    FEM_TOLERANCE,
    REYNOLDS,
    RESULTS_DIR,
)


def _build_reference_domain():
    """Reuse the UVP benchmark setup without its pressure-point bound."""
    return df.domain(build_domain("uvp").area_list[0])


def _sample_solution(reference):
    """Sample the documented FEM solution object on the requested grid."""
    data = reference.area_evaluators[0].data_dict
    x = np.unique(np.asarray(data["x"]).reshape(-1))
    y = np.unique(np.asarray(data["y"]).reshape(-1))
    query_x, query_y = np.meshgrid(x, y, indexing="xy")
    solution = reference.reference_solution
    fields = solution.evaluate(query_x, query_y, fields=("u", "v", "p"))

    pressure_offset = float(
        solution.evaluate([CAVITY_X[0]], [CAVITY_Y[0]], fields=["p"])["p"][0]
    )
    fields["p"] = np.asarray(fields["p"]) - pressure_offset

    vertical_y = np.linspace(*CAVITY_Y, CENTERLINE_POINTS)
    vertical_u = solution.evaluate(
        np.full_like(vertical_y, 0.5), vertical_y, fields=["u"]
    )["u"]
    horizontal_x = np.linspace(*CAVITY_X, CENTERLINE_POINTS)
    horizontal_v = solution.evaluate(
        horizontal_x, np.full_like(horizontal_x, 0.5), fields=["v"]
    )["v"]
    return (
        x,
        y,
        fields,
        vertical_y,
        vertical_u,
        horizontal_x,
        horizontal_v,
        pressure_offset,
    )


def _payload(reference, mesh_size, runtime_s):
    (
        x,
        y,
        fields,
        vertical_y,
        vertical_u,
        horizontal_x,
        horizontal_v,
        pressure_offset,
    ) = _sample_solution(reference)
    metadata = reference.metadata
    residuals = np.asarray(metadata.get("solver_residuals", []), dtype=float)
    mesh = metadata.get("mesh", {})
    return {
        "x": x,
        "y": y,
        **{name: np.asarray(fields[name]) for name in ("u", "v", "p")},
        "vertical_y": vertical_y,
        "vertical_u": np.asarray(vertical_u),
        "horizontal_x": horizontal_x,
        "horizontal_v": np.asarray(horizontal_v),
        "nx": x.size,
        "ny": y.size,
        "dx": np.diff(x).mean(),
        "dy": np.diff(y).mean(),
        "reynolds": REYNOLDS,
        "converged": int(metadata["converged"]),
        "iterations": metadata["iterations"],
        "final_residual": residuals[-1],
        "runtime_s": runtime_s,
        "mesh_size": mesh_size,
        "mesh_elements": mesh["elements"],
        "mesh_vertices": mesh["vertices"],
        "pressure_offset": pressure_offset,
        "pressure_gauge": np.asarray("corner_anchored"),
        "backend": np.asarray(metadata["backend"]),
    }


def solve_reference(grid, mesh_size, output_path=None):
    """Solve and optionally export one FEM reference."""
    print(f"DeepFlow FEM cavity reference ({grid[0]} x {grid[1]} samples)")
    start = time.perf_counter()
    domain = _build_reference_domain()
    reference = domain.solve_fem(
        mesh_size=mesh_size,
        boundary_resolution=FEM_BOUNDARY_RESOLUTION,
        tolerance=FEM_TOLERANCE,
        max_iterations=FEM_MAX_ITERATIONS,
        area_sampling_res=list(grid),
        bound_sampling_res=FEM_BOUNDARY_RESOLUTION,
    )
    payload = _payload(reference, mesh_size, time.perf_counter() - start)
    if not payload["converged"]:
        raise RuntimeError("DeepFlow FEM reference did not converge.")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **payload)
    return payload


def save_grid_convergence(coarse, refined, output_path):
    """Compare coarse and refined FEM payloads and save only the metrics."""
    points = np.stack(
        np.meshgrid(coarse["y"], coarse["x"], indexing="ij"), axis=-1
    ).reshape(-1, 2)
    metrics = {
        "coarse_mesh_size": float(coarse["mesh_size"]),
        "refined_mesh_size": float(refined["mesh_size"]),
        "coarse_nx": int(coarse["nx"]),
        "coarse_ny": int(coarse["ny"]),
        "refined_nx": int(refined["nx"]),
        "refined_ny": int(refined["ny"]),
    }
    for field in ("u", "v", "p"):
        refined_values = RegularGridInterpolator(
            (refined["y"], refined["x"]), refined[field],
            bounds_error=False, fill_value=None,
        )(points).reshape(coarse[field].shape)
        metrics[f"l2_relative_{field}"] = np.linalg.norm(
            coarse[field] - refined_values
        ) / max(np.linalg.norm(refined_values), 1.0e-14)

    for field, coordinate in (
        ("vertical_u", "vertical_y"),
        ("horizontal_v", "horizontal_x"),
    ):
        refined_profile = np.interp(
            coarse[coordinate], refined[coordinate], refined[field]
        )
        metrics[f"rmse_{field}"] = np.sqrt(
            np.mean((coarse[field] - refined_profile) ** 2)
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **metrics)


def generate_reference():
    """Generate the FEM reference and convergence metrics."""
    coarse_path = RESULTS_DIR / FEM_REFERENCE_FILENAME
    coarse = solve_reference(FEM_DEFAULT_GRID, FEM_COARSE_MESH_SIZE, coarse_path)
    refined = solve_reference(FEM_REFINED_GRID, FEM_REFINED_MESH_SIZE)
    if not coarse["converged"] or not refined["converged"]:
        raise RuntimeError("One or more FEM references did not converge.")
    save_grid_convergence(
        coarse, refined, RESULTS_DIR / FEM_GRID_CONVERGENCE_FILENAME
    )


if __name__ == "__main__":
    generate_reference()
