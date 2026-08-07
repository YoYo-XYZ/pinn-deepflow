# Direct `(u,v,p)` vs Stream-Function `(psi,p)` Cavity Benchmark

This benchmark compares two classical DeepFlow PINNs on the same steady
lid-driven cavity problem used by
`benchmarks/comparing_pinns_qcpinn_cavity`:

- **UVP**: direct model outputs `u`, `v`, and `p`, using `NavierStokes`.
- **PSIP**: model outputs `psi` and `p`, using `StreamFunctionNavierStokes`;
  velocity is derived as `u=psi_y` and `v=-psi_x`.

## Problem and protocol

- Geometry: unit square `[0,1] x [0,1]`.
- Physics: steady incompressible Navier-Stokes with `U=1`, `L=1`,
  `rho=1`, `mu=0.1`, hence `Re=10`.
- UVP boundary conditions: no-slip on the left, bottom, and right walls;
  top lid `u=1, v=0`; pressure anchor `p=0` at the lower-left corner.
- PSIP boundary conditions: `psi_x=0, psi_y=0` on the stationary walls;
  `psi_x=0, psi_y=1` on the top lid; the same pressure anchor.
- Sampling: 50 points per wall, one pressure point, and a `50 x 50`
  interior grid.
- Training: no Adam warm-up, followed by 100 L-BFGS epochs, with three
  independent seeds `[69, 70, 71]` by default.
- Loss: raw `df.calc_loss_simple`, matching the existing cavity benchmark.

The raw PDE totals are not directly apples-to-apples: UVP has continuity plus
two momentum residuals, while PSIP has only the two momentum residuals because
continuity is satisfied analytically. The report therefore includes both raw
losses and equation-wise residual statistics, with shared CFD field errors as
the main solution-quality comparison.

The UVP model has 7,347 trainable parameters. The PSIP model uses hidden layers
`[48, 48, 48, 49]` and has 7,349 parameters, matching capacity to within two
parameters.

## Running

From the repository root:

```bash
# Full three-seed benchmark and comparison report
python benchmarks/comparing_pinn_formulations_cavity/run_benchmark.py --all

# Reduced pipeline validation
python benchmarks/comparing_pinn_formulations_cavity/run_benchmark.py \
    --all --num_runs 1 --epochs_adam 0 --epochs_lbfgs 1

# Validate geometry, derivative BCs, PDE residuals, and evaluation
python benchmarks/comparing_pinn_formulations_cavity/smoke_test.py

# Re-generate plots/report from existing model result files
python benchmarks/comparing_pinn_formulations_cavity/run_benchmark.py --compare
```

The comparison reuses
`benchmarks/comparing_pinns_qcpinn_cavity/results/cfd_reference.npz`. If that
file is unavailable, training metrics and plots are still generated, but CFD
accuracy metrics are skipped with a warning.

## Outputs

Results are written to `results/`:

- `uvp_results.npz` and `psip_results.npz` contain aggregate metrics and the
  median-loss run's fields.
- `benchmark_report.md` summarizes losses, residuals, timings, and CFD errors.
- `compare_loss_curves.png`, common field plots, residual plots, centerline
  profiles, and CFD error plots provide visual comparisons.
