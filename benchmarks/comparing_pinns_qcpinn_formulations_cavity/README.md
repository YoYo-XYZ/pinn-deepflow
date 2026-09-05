# PINN/QCPINN × UVP/PSIP Cavity Benchmark

This benchmark evaluates four variants of steady two-dimensional lid-driven
cavity flow at **Re = 10**:

1. PINN with direct `(u,v,p)` outputs (UVP)
2. QCPINN with direct `(u,v,p)` outputs (UVP)
3. PINN with streamfunction `(psi,p)` outputs (PSIP)
4. QCPINN with streamfunction `(psi,p)` outputs (PSIP)

The variants share the DeepFlow domain builder, training path, area
evaluation, centerline evaluation, persistence, plotting, and reporting. The
cavity-specific code is limited to its square geometry, boundary conditions,
and vertical/horizontal centerline definitions.

## Configuration

- Geometry: unit square `[0,1] x [0,1]`.
- Physics: steady incompressible Navier-Stokes, `U=1`, `L=1`, `rho=1`, and
  `mu=0.1`, giving `Re=10`.
- Sampling: 50 points on each wall, one pressure-reference point, and a
  `50 x 50` interior grid.
- Training: seed `69`, no Adam warm-up, and 100 L-BFGS epochs.
- PINN: `width=48`, `length=4`.
- QCPINN: `pre=[32]`, `post=[32]`, `nqubits=4`, and
  `q_layer_iterations=10`.

## Running

From the repository root:

```bash
# Fast shared-harness smoke path; includes QCPINN when PennyLane is installed
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/smoke_test.py --all-setups

# Fresh FEM/reference smoke when the optional backend is installed
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/smoke_test.py --pinn-reference

# Full reference, setup runs, and comparison
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/run_benchmark.py --all

# Re-run setup cells with the small configuration
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/run_benchmark.py \
    --all --smoke
```

The individual setup scripts also accept `--smoke`, `--num_runs`,
`--epochs_adam`, `--epochs_lbfgs`, and `--output-dir`.

`compare.py` solves a fresh FEM reference by default for a normal comparison.
Use `--no-reference` to skip it, or pass `--offline-reference PATH` to
explicitly use a cached archive. Cached data is never selected implicitly.

## Outputs

Results are written to `results/` as native DeepFlow model pickles,
visualizer field/loss plots, and `REPORT.md`. Reports include variant metadata,
residual metrics, timings, and centerline reference metrics when a FEM or
explicit offline reference is selected.

PennyLane is required for the two QCPINN variants. The FEM reference requires
the optional NGSolve/Netgen backend used by `domain.solve_fem()`.
