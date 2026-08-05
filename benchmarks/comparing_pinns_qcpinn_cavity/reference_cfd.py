#!/usr/bin/env python3
"""Finite-volume SIMPLE reference solution for the Re=10 cavity benchmark."""

import argparse
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import bicgstab, spsolve
from scipy.interpolate import RegularGridInterpolator

from common_config import (
    CFD_ALPHA_P,
    CFD_ALPHA_U,
    CFD_ALPHA_V,
    CFD_DEFAULT_GRID,
    CFD_GRID_CONVERGENCE_FILENAME,
    CFD_LINEAR_MAX_ITERATIONS,
    CFD_LINEAR_TOLERANCE,
    CFD_MAX_ITERATIONS,
    CFD_REFINED_FILENAME,
    CFD_REFINED_GRID,
    CFD_REFERENCE_FILENAME,
    CFD_TOLERANCE,
    CAVITY_X,
    CAVITY_Y,
    L_CHAR,
    LID_VELOCITY,
    MU,
    RESULTS_DIR,
    RHO,
)


def _solve_sparse(matrix, rhs, initial, tolerance, max_iterations):
    """Solve a sparse linear system with a diagonal preconditioner."""
    diagonal = matrix.diagonal()
    safe_diagonal = np.where(np.abs(diagonal) > 1.0e-14, diagonal, 1.0)
    preconditioner = sp.diags(1.0 / safe_diagonal)
    solution, info = bicgstab(
        matrix,
        rhs,
        x0=initial,
        rtol=tolerance,
        atol=0.0,
        maxiter=max_iterations,
        M=preconditioner,
    )
    if info != 0 or not np.all(np.isfinite(solution)):
        solution = spsolve(matrix, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Sparse linear solve returned non-finite values.")
    return solution


class StaggeredSimpleSolver:
    """Steady staggered-grid finite-volume solver using SIMPLE."""

    def __init__(
        self,
        nx,
        ny,
        max_iterations=CFD_MAX_ITERATIONS,
        tolerance=CFD_TOLERANCE,
        linear_tolerance=CFD_LINEAR_TOLERANCE,
        linear_max_iterations=CFD_LINEAR_MAX_ITERATIONS,
        alpha_u=CFD_ALPHA_U,
        alpha_v=CFD_ALPHA_V,
        alpha_p=CFD_ALPHA_P,
        report_every=100,
    ):
        if nx < 3 or ny < 3:
            raise ValueError("The CFD grid must contain at least 3 cells per direction.")
        if not 0.0 < alpha_u <= 1.0 or not 0.0 < alpha_v <= 1.0:
            raise ValueError("Velocity under-relaxation factors must be in (0, 1].")
        if not 0.0 < alpha_p <= 1.0:
            raise ValueError("Pressure under-relaxation must be in (0, 1].")

        self.nx = int(nx)
        self.ny = int(ny)
        self.dx = (CAVITY_X[1] - CAVITY_X[0]) / self.nx
        self.dy = (CAVITY_Y[1] - CAVITY_Y[0]) / self.ny
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.linear_tolerance = float(linear_tolerance)
        self.linear_max_iterations = int(linear_max_iterations)
        self.alpha_u = float(alpha_u)
        self.alpha_v = float(alpha_v)
        self.alpha_p = float(alpha_p)
        self.report_every = max(1, int(report_every))

        # p is cell-centered; u and v live on vertical and horizontal faces.
        self.p = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.u = np.zeros((self.ny, self.nx + 1), dtype=np.float64)
        self.v = np.zeros((self.ny + 1, self.nx), dtype=np.float64)
        self._apply_velocity_boundaries()

    def _apply_velocity_boundaries(self):
        """Apply the fixed velocity values on the outer staggered faces."""
        self.u[:, 0] = 0.0
        self.u[:, -1] = 0.0
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

    def initialize_from_cell_centered_result(self, result):
        """Prolongate a coarser converged solution onto this fine grid."""
        source_points = np.stack(
            np.meshgrid(result["y"], result["x"], indexing="ij"), axis=-1
        ).reshape(-1, 2)
        target_x = CAVITY_X[0] + (np.arange(self.nx) + 0.5) * self.dx
        target_y = CAVITY_Y[0] + (np.arange(self.ny) + 0.5) * self.dy
        target_points = np.stack(
            np.meshgrid(target_y, target_x, indexing="ij"), axis=-1
        ).reshape(-1, 2)

        def interpolate(field):
            interpolator = RegularGridInterpolator(
                (result["y"], result["x"]),
                result[field],
                bounds_error=False,
                fill_value=None,
            )
            return interpolator(target_points).reshape(self.ny, self.nx)

        u_center = interpolate("u")
        v_center = interpolate("v")
        self.p = interpolate("p")
        self.u[:, 1:-1] = 0.5 * (u_center[:, :-1] + u_center[:, 1:])
        self.v[1:-1, :] = 0.5 * (v_center[:-1, :] + v_center[1:, :])
        self._apply_velocity_boundaries()

    def _u_momentum_system(self):
        """Assemble the central-difference u-momentum system."""
        nxu = self.nx - 1
        u_internal = self.u[:, 1:-1]
        fw = RHO * self.u[:, :-2] * self.dy
        fe = RHO * self.u[:, 2:] * self.dy
        fs = RHO * 0.5 * (self.v[:-1, :-1] + self.v[:-1, 1:]) * self.dx
        fn = RHO * 0.5 * (self.v[1:, :-1] + self.v[1:, 1:]) * self.dx

        dw = np.full_like(u_internal, MU * self.dy / self.dx)
        de = np.full_like(u_internal, MU * self.dy / self.dx)
        ds = np.full_like(u_internal, MU * self.dx / self.dy)
        dn = np.full_like(u_internal, MU * self.dx / self.dy)
        dw[:, 0] *= 2.0
        de[:, -1] *= 2.0
        ds[0, :] *= 2.0
        dn[-1, :] *= 2.0

        aw = dw + fw / 2.0
        ae = de - fe / 2.0
        ass = ds + fs / 2.0
        an = dn - fn / 2.0
        ap = np.maximum(aw + ae + ass + an + (fe - fw + fn - fs), 1.0e-14)
        ap_relaxed = ap / self.alpha_u
        diagonal_coefficients = np.zeros((self.ny, self.nx + 1), dtype=np.float64)
        diagonal_coefficients[:, 1:-1] = ap_relaxed

        east = -ae.copy()
        west = -aw.copy()
        north = -an.copy()
        south = -ass.copy()
        east[:, -1] = 0.0
        west[:, 0] = 0.0
        north[-1, :] = 0.0
        south[0, :] = 0.0
        matrix = sp.diags(
            [
                south.ravel()[nxu:],
                west.ravel()[1:],
                ap_relaxed.ravel(),
                east.ravel()[:-1],
                north.ravel()[:-nxu],
            ],
            [-nxu, -1, 0, 1, nxu],
            shape=(self.ny * nxu, self.ny * nxu),
            format="csr",
        )
        rhs = (
            (1.0 - self.alpha_u) / self.alpha_u * ap * u_internal
            + (self.p[:, :-1] - self.p[:, 1:]) * self.dy
        )
        rhs[-1, :] += an[-1, :] * LID_VELOCITY
        return matrix, rhs.ravel(), diagonal_coefficients

    def _v_momentum_system(self):
        """Assemble the central-difference v-momentum system."""
        nyv = self.ny - 1
        v_internal = self.v[1:-1, :]
        fw = RHO * 0.5 * (self.u[:-1, :-1] + self.u[1:, :-1]) * self.dy
        fe = RHO * 0.5 * (self.u[:-1, 1:] + self.u[1:, 1:]) * self.dy
        fs = RHO * self.v[:-2, :] * self.dx
        fn = RHO * self.v[2:, :] * self.dx

        dw = np.full_like(v_internal, MU * self.dy / self.dx)
        de = np.full_like(v_internal, MU * self.dy / self.dx)
        ds = np.full_like(v_internal, MU * self.dx / self.dy)
        dn = np.full_like(v_internal, MU * self.dx / self.dy)
        dw[:, 0] *= 2.0
        de[:, -1] *= 2.0
        ds[0, :] *= 2.0
        dn[-1, :] *= 2.0

        aw = dw + fw / 2.0
        ae = de - fe / 2.0
        ass = ds + fs / 2.0
        an = dn - fn / 2.0
        ap = np.maximum(aw + ae + ass + an + (fe - fw + fn - fs), 1.0e-14)
        ap_relaxed = ap / self.alpha_v
        diagonal_coefficients = np.zeros((self.ny + 1, self.nx), dtype=np.float64)
        diagonal_coefficients[1:-1, :] = ap_relaxed

        east = -ae.copy()
        west = -aw.copy()
        north = -an.copy()
        south = -ass.copy()
        east[:, -1] = 0.0
        west[:, 0] = 0.0
        north[-1, :] = 0.0
        south[0, :] = 0.0
        matrix = sp.diags(
            [
                south.ravel()[self.nx:],
                west.ravel()[1:],
                ap_relaxed.ravel(),
                east.ravel()[:-1],
                north.ravel()[:-self.nx],
            ],
            [-self.nx, -1, 0, 1, self.nx],
            shape=(nyv * self.nx, nyv * self.nx),
            format="csr",
        )
        rhs = (
            (1.0 - self.alpha_v) / self.alpha_v * ap * v_internal
            + (self.p[:-1, :] - self.p[1:, :]) * self.dx
        )
        return matrix, rhs.ravel(), diagonal_coefficients

    def _pressure_correction_system(self, mass_imbalance, u_diagonal, v_diagonal):
        """Assemble the SIMPLE pressure-correction Poisson system."""
        west = np.zeros_like(mass_imbalance)
        east = np.zeros_like(mass_imbalance)
        south = np.zeros_like(mass_imbalance)
        north = np.zeros_like(mass_imbalance)
        west[:, 1:] = self.dy * self.dy / u_diagonal[:, 1:-1]
        east[:, :-1] = self.dy * self.dy / u_diagonal[:, 1:-1]
        south[1:, :] = self.dx * self.dx / v_diagonal[1:-1, :]
        north[:-1, :] = self.dx * self.dx / v_diagonal[1:-1, :]

        diagonal = west + east + south + north
        # Fix p' in the lower-left cell and remove its row couplings.
        east[0, 0] = 0.0
        north[0, 0] = 0.0
        diagonal[0, 0] = 1.0
        matrix = sp.diags(
            [
                -south.ravel()[self.nx:],
                -west.ravel()[1:],
                diagonal.ravel(),
                -east.ravel()[:-1],
                -north.ravel()[:-self.nx],
            ],
            [-self.nx, -1, 0, 1, self.nx],
            shape=(self.nx * self.ny, self.nx * self.ny),
            format="csr",
        )
        rhs = -mass_imbalance.ravel()
        rhs[0] = 0.0
        return matrix, rhs

    def _correct_velocities(self, pressure_correction, u_diagonal, v_diagonal):
        """Apply the SIMPLE pressure-correction velocity updates."""
        correction = pressure_correction.reshape(self.ny, self.nx)
        for j in range(self.ny):
            for i in range(1, self.nx):
                self.u[j, i] += u_diagonal[j, i] ** -1 * self.dy * (
                    correction[j, i - 1] - correction[j, i]
                )
        for j in range(1, self.ny):
            for i in range(self.nx):
                self.v[j, i] += v_diagonal[j, i] ** -1 * self.dx * (
                    correction[j - 1, i] - correction[j, i]
                )
        self._apply_velocity_boundaries()

    def _cell_center_velocity(self):
        """Interpolate staggered velocities to pressure-cell centers."""
        u_center = 0.5 * (self.u[:, 1:] + self.u[:, :-1])
        v_center = 0.5 * (self.v[1:, :] + self.v[:-1, :])
        return u_center, v_center

    def solve(self):
        """Run SIMPLE and return the converged fields and diagnostics."""
        history = []
        start = time.perf_counter()

        for iteration in range(1, self.max_iterations + 1):
            old_u = self.u.copy()
            old_v = self.v.copy()
            old_p = self.p.copy()

            u_matrix, u_rhs, u_diagonal = self._u_momentum_system()
            u_solution = _solve_sparse(
                u_matrix,
                u_rhs,
                self.u[:, 1:-1].reshape(-1),
                self.linear_tolerance,
                self.linear_max_iterations,
            )
            self.u[:, 1:-1] = u_solution.reshape(self.ny, self.nx - 1)

            v_matrix, v_rhs, v_diagonal = self._v_momentum_system()
            v_solution = _solve_sparse(
                v_matrix,
                v_rhs,
                self.v[1:-1, :].reshape(-1),
                self.linear_tolerance,
                self.linear_max_iterations,
            )
            self.v[1:-1, :] = v_solution.reshape(self.ny - 1, self.nx)
            self._apply_velocity_boundaries()

            mass_imbalance = (
                (self.u[:, 1:] - self.u[:, :-1]) * self.dy
                + (self.v[1:, :] - self.v[:-1, :]) * self.dx
            )
            pressure_matrix, pressure_rhs = self._pressure_correction_system(
                mass_imbalance, u_diagonal, v_diagonal
            )
            pressure_correction = _solve_sparse(
                pressure_matrix,
                pressure_rhs,
                np.zeros(self.nx * self.ny),
                self.linear_tolerance,
                self.linear_max_iterations,
            )
            self.p += self.alpha_p * pressure_correction.reshape(self.ny, self.nx)
            self.p -= self.p[0, 0]
            self._correct_velocities(
                pressure_correction, u_diagonal, v_diagonal
            )

            mass_imbalance = (
                (self.u[:, 1:] - self.u[:, :-1]) * self.dy
                + (self.v[1:, :] - self.v[:-1, :]) * self.dx
            )
            velocity_change = max(
                np.max(np.abs(self.u - old_u)) / max(LID_VELOCITY, 1.0e-14),
                np.max(np.abs(self.v - old_v)) / max(LID_VELOCITY, 1.0e-14),
            )
            pressure_change = np.max(np.abs(self.alpha_p * pressure_correction))
            mass_scale = RHO * max(LID_VELOCITY, 1.0e-14) * max(self.dx, self.dy)
            continuity_residual = np.max(np.abs(mass_imbalance)) / mass_scale
            pressure_scale = max(RHO * LID_VELOCITY**2, 1.0e-14)
            residual = max(
                continuity_residual,
                velocity_change,
                pressure_change / pressure_scale,
            )
            history.append(
                [iteration, residual, continuity_residual, velocity_change, pressure_change]
            )

            if iteration == 1 or iteration % self.report_every == 0:
                print(
                    f"  CFD iteration {iteration:5d}: residual={residual:.3e}, "
                    f"continuity={continuity_residual:.3e}"
                )
            if residual < self.tolerance:
                break

        history = np.asarray(history, dtype=np.float64)
        converged = bool(len(history) and history[-1, 1] < self.tolerance)
        elapsed = time.perf_counter() - start
        return self._result(history, converged, elapsed)

    def _result(self, history, converged, elapsed):
        """Convert the staggered solution to benchmark output fields."""
        x = CAVITY_X[0] + (np.arange(self.nx) + 0.5) * self.dx
        y = CAVITY_Y[0] + (np.arange(self.ny) + 0.5) * self.dy
        u_center, v_center = self._cell_center_velocity()
        vertical_u = np.array(
            [np.interp(0.5, x, row) for row in u_center], dtype=np.float64
        )
        horizontal_v = np.array(
            [np.interp(0.5, y, v_center[:, i]) for i in range(self.nx)],
            dtype=np.float64,
        )
        mass_imbalance = (
            (self.u[:, 1:] - self.u[:, :-1]) * self.dy
            + (self.v[1:, :] - self.v[:-1, :]) * self.dx
        )

        return {
            "x": x,
            "y": y,
            "u": u_center,
            "v": v_center,
            "p": self.p.copy(),
            "vertical_y": y,
            "vertical_u": vertical_u,
            "horizontal_x": x,
            "horizontal_v": horizontal_v,
            "continuity_residual": mass_imbalance,
            "solver_history": history,
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "reynolds": RHO * LID_VELOCITY * L_CHAR / MU,
            "converged": int(converged),
            "iterations": int(history[-1, 0]) if len(history) else 0,
            "runtime_s": float(elapsed),
            "final_residual": float(history[-1, 1]) if len(history) else np.inf,
            "final_continuity_residual": float(history[-1, 2]) if len(history) else np.inf,
        }


def run_reference(
    nx,
    ny,
    output_path,
    max_iterations=CFD_MAX_ITERATIONS,
    tolerance=CFD_TOLERANCE,
    report_every=100,
    initial_result=None,
):
    """Run and save one CFD reference solution."""
    print("=" * 80)
    print(f"Finite-volume SIMPLE cavity reference ({nx} x {ny} cells, Re=10)")
    print("=" * 80)
    solver = StaggeredSimpleSolver(
        nx,
        ny,
        max_iterations=max_iterations,
        tolerance=tolerance,
        report_every=report_every,
    )
    if initial_result is not None:
        solver.initialize_from_cell_centered_result(initial_result)
    result = solver.solve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)
    print(
        f"Saved CFD reference to {output_path} ({result['iterations']} iterations, "
        f"residual={result['final_residual']:.3e}, converged={bool(result['converged'])})"
    )
    return result


def save_grid_convergence(coarse_path, refined_path, output_path):
    """Compare the refined CFD fields against the coarse-grid solution."""
    with np.load(coarse_path) as coarse, np.load(refined_path) as refined:
        points = np.stack(
            np.meshgrid(coarse["y"], coarse["x"], indexing="ij"), axis=-1
        ).reshape(-1, 2)
        metrics = {
            "coarse_nx": int(coarse["nx"]),
            "coarse_ny": int(coarse["ny"]),
            "refined_nx": int(refined["nx"]),
            "refined_ny": int(refined["ny"]),
        }
        for field in ("u", "v", "p"):
            interpolator = RegularGridInterpolator(
                (refined["y"], refined["x"]),
                refined[field],
                bounds_error=False,
                fill_value=None,
            )
            refined_on_coarse = interpolator(points).reshape(coarse[field].shape)
            difference = coarse[field] - refined_on_coarse
            denominator = max(float(np.linalg.norm(refined_on_coarse)), 1.0e-14)
            metrics[f"l2_relative_{field}"] = float(
                np.linalg.norm(difference) / denominator
            )

        for field in ("vertical_u", "horizontal_v"):
            coordinate = "vertical_y" if field == "vertical_u" else "horizontal_x"
            refined_profile = np.interp(
                coarse[coordinate], refined[coordinate], refined[field]
            )
            metrics[f"rmse_{field}"] = float(
                np.sqrt(np.mean((coarse[field] - refined_profile) ** 2))
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **metrics)
    print(f"Saved CFD grid-convergence metrics to {output_path}")
    return metrics


def _parse_grid(value):
    parts = value.lower().replace("x", " ").split()
    if len(parts) == 1:
        nx = ny = int(parts[0])
    elif len(parts) == 2:
        nx, ny = (int(part) for part in parts)
    else:
        raise argparse.ArgumentTypeError("Grid must be N or NxM.")
    if nx < 3 or ny < 3:
        raise argparse.ArgumentTypeError("Grid dimensions must be at least 3.")
    return nx, ny


def main():
    parser = argparse.ArgumentParser(
        description="Generate a finite-volume SIMPLE reference for cavity flow."
    )
    parser.add_argument(
        "--grid",
        type=_parse_grid,
        default=CFD_DEFAULT_GRID,
        help="Cell grid as N or NxM (default: 101x101).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NPZ path (default: results/cfd_reference.npz).",
    )
    parser.add_argument("--max_iterations", type=int, default=CFD_MAX_ITERATIONS)
    parser.add_argument("--tolerance", type=float, default=CFD_TOLERANCE)
    parser.add_argument("--report_every", type=int, default=100)
    parser.add_argument(
        "--refine",
        action="store_true",
        help="Use the 201x201 refinement grid and its standard output filename.",
    )
    parser.add_argument(
        "--grid_convergence",
        action="store_true",
        help="Run both reference grids and save their convergence comparison.",
    )
    args = parser.parse_args()

    if args.grid_convergence:
        coarse_path = RESULTS_DIR / CFD_REFERENCE_FILENAME
        refined_path = RESULTS_DIR / CFD_REFINED_FILENAME
        coarse = run_reference(
            CFD_DEFAULT_GRID[0],
            CFD_DEFAULT_GRID[1],
            coarse_path,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            report_every=args.report_every,
        )
        refined = run_reference(
            CFD_REFINED_GRID[0],
            CFD_REFINED_GRID[1],
            refined_path,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            report_every=args.report_every,
            initial_result=coarse,
        )
        if not bool(coarse["converged"]) or not bool(refined["converged"]):
            raise SystemExit("One or more CFD reference grids did not converge.")
        save_grid_convergence(
            coarse_path,
            refined_path,
            RESULTS_DIR / CFD_GRID_CONVERGENCE_FILENAME,
        )
        return

    grid = CFD_REFINED_GRID if args.refine else args.grid
    if args.refine and args.grid != CFD_DEFAULT_GRID:
        grid = args.grid
    output = args.output
    if output is None:
        filename = CFD_REFINED_FILENAME if args.refine else CFD_REFERENCE_FILENAME
        output = RESULTS_DIR / filename
    result = run_reference(
        grid[0],
        grid[1],
        output,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        report_every=args.report_every,
    )
    if not bool(result["converged"]):
        raise SystemExit("CFD reference did not reach the requested tolerance.")


if __name__ == "__main__":
    main()
