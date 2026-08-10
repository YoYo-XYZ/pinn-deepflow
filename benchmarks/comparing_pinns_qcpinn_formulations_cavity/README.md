# PINN/QCPINN × UVP/PSIP Cavity Benchmark

This benchmark evaluates four combinations on the steady two-dimensional
lid-driven cavity at **Re = 10**:

1. PINN with direct `(u,v,p)` outputs (UVP)
2. QCPINN with direct `(u,v,p)` outputs (UVP)
3. PINN with stream-function `(psi,p)` outputs (PSIP)
4. QCPINN with stream-function `(psi,p)` outputs (PSIP)

The UVP and PSIP formulations use the same cavity, sampling, optimizer, seed,
and DeepFlow FEM reference. PSIP derives velocity as `u=psi_y` and `v=-psi_x`, so its
continuity equation is satisfied analytically.

## Configuration

- Geometry: unit square `[0,1] x [0,1]`.
- Physics: steady incompressible Navier-Stokes, `U=1`, `L=1`, `rho=1` and
  `mu=0.1`, giving `Re=10`.
- Sampling: 50 points on each wall, one pressure-reference point, and a
  `50 x 50` interior grid.
- Training: seed `69`, no Adam warm-up, 100 L-BFGS epochs, and no resampling.
- PINN: `PINN(width=48, length=4)`.
- QCPINN: `pre=[32]`, `post=[32]`, `nqubits=4`, and
  `q_layer_iterations=10`.

The resulting trainable parameter counts are 7,347 (PINN-UVP), 607
(QCPINN-UVP), 7,298 (PINN-PSIP), and 574 (QCPINN-PSIP). These intentionally
preserve the existing QCPINN-vs-PINN benchmark capacities rather than
equalizing them.

## Running

From the repository root:

```bash
# Smoke test all four domains, models, residuals, and evaluations
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/smoke_test.py

# Full fresh FEM reference, four setup runs, plots, and report
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/run_benchmark.py --all

# Reduced pipeline check
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/run_benchmark.py \
    --all --num_runs 1 --epochs_adam 0 --epochs_lbfgs 1

# Recreate plots/report from existing combined results
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/run_benchmark.py \
    --compare
```

PennyLane is required for the two QCPINN setups. The full one-seed run is
expected to take several CPU hours, with QCPINN-PSIP being the slowest setup.

## Outputs

Results are written to `results/`:

- Four `*_results.npz` files containing metadata, aggregate metrics, the
  representative fields, residual fields, histories, and centerline profiles.
- Fresh 101x101 DeepFlow FEM reference.
- Five consolidated comparison figures: solution fields, PDE residuals, training
  losses, centerline profiles, and FEM reference/model errors. The solution
  figure includes `u`, `v`, velocity magnitude, and `p`; the FEM figure includes
  the reference fields and all four model-error rows.
- `benchmark_report.md` with the four-cell summary and factorized comparisons.

Raw UVP and PSIP PDE totals are not directly apples-to-apples because UVP has
three residual equations and PSIP has two. The report includes PDE loss per
residual, equation-wise residuals, solution fields, and FEM errors for the
cross-formulation comparison.

The reference stage requires the optional NGSolve/Netgen dependency used by
DeepFlow's `domain.solve_fem()` backend. The reference is saved as
`cfd_reference.npz`.
