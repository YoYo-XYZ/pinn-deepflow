# PINN/QCPINN x UVP/PSIP Cylinder Benchmark

This benchmark compares four DeepFlow model/formulation variants on steady incompressible flow around the cylinder from `examples/cylinder_flow_steady/`. All variants use the shared benchmark domain, PDE, training, evaluation, and reporting helpers:

1. PINN with direct `(u,v,p)` outputs.
2. QCPINN with direct `(u,v,p)` outputs.
3. PINN with stream-function `(psi,p)` outputs.
4. QCPINN with stream-function `(psi,p)` outputs.

The channel is `[0,1.1] x [0,0.41]`, the cylinder is centered at `(0.2,0.2)` with radius `0.05`, and the inlet is the example’s parabolic profile. The reduced-Reynolds-number case uses `U=1`, `L=1`, `rho=1`, `mu=0.1`, giving `Re=10`. Solid walls and the cylinder are no-slip; the outlet has `p=0`.

## Configuration

- Uniform sampling: 50 points on each of six boundaries and a masked `50 x 50` interior grid.
- Training: seed 69, no Adam warm-up, 100 L-BFGS epochs.
- PINN: `width=48`, `length=4`.
- QCPINN: `pre=[32]`, `post=[32]`, `nqubits=4`, `q_layer_iterations=10`.
- FEM reference: fresh `domain.solve_fem` solution queried at the model and profile evaluation coordinates.

## Verification and running

From the repository root:

```bash
python benchmarks/comparing_pinns_qcpinn_cylinder/smoke_test.py --all-setups
python benchmarks/comparing_pinns_qcpinn_cylinder/smoke_test.py --pinn-reference
python benchmarks/comparing_pinns_qcpinn_cylinder/run_benchmark.py --all
```

The smoke path runs the standard variants everywhere and includes QCPINN when PennyLane is installed. The runner also supports `--pinn-uvp`, `--pinn-psip`, `--qcpinn-uvp`, `--qcpinn-psip`, `--reference`, `--compare`, `--num_runs`, `--epochs_adam`, and `--epochs_lbfgs`.

`compare.py` solves a fresh FEM reference by default. Use `--no-reference` to compare model outputs without a reference, or pass `--offline-reference PATH` to explicitly use a cached archive when the FEM backend is unavailable. Cached data is never selected implicitly.

## Outputs

`results/` receives native DeepFlow model pickles, visualizer field/loss plots, and `REPORT.md`. Reports include parameter counts, residual metrics, timings, outlet/wake profile errors when a FEM reference is selected, and the explicit reference mode.
