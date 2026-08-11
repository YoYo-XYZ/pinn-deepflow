# PINN/QCPINN x UVP/PSIP Cylinder Benchmark

This benchmark compares four DeepFlow cells on steady incompressible flow around the cylinder from `examples/cylinder_flow_steady/`:

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
- FEM reference: fresh NGSolve solution, with masked obstacle-aware field and error comparisons.

## Verification and running

From the repository root:

```bash
python benchmarks/comparing_pinns_qcpinn_cylinder/smoke_test.py --pinn-reference
python benchmarks/comparing_pinns_qcpinn_cylinder/smoke_test.py --all-setups
python benchmarks/comparing_pinns_qcpinn_cylinder/run_benchmark.py --all
```

The runner also supports `--pinn-uvp`, `--pinn-psip`, `--qcpinn-uvp`, `--qcpinn-psip`, `--reference`, `--compare`, `--num_runs`, `--epochs_adam`, and `--epochs_lbfgs`.

## Outputs

`results/` contains four setup NPZ files, `cfd_reference.npz`, solution/residual/loss/profile/FEM-error figures, and `benchmark_report.md`. The FEM error norms ignore the circular obstacle and non-finite interpolation cells. The report includes parameter counts, residual metrics, timings, outlet/wake profile errors, and fresh-reference diagnostics.
