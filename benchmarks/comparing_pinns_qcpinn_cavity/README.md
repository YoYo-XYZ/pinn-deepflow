# QCPINN vs PINN -- Re=10 Lid-Driven Cavity Benchmark

This benchmark compares a parameter-matched classical PINN and QCPINN on the
steady 2D lid-driven cavity problem at **Re = 10**.

The problem follows the unit-square cavity setup discussed in [A QPINN
Framework with Quantum Trainable Embeddings for the Lid-Driven Cavity
Problem](https://arxiv.org/pdf/2605.13892), but uses DeepFlow's existing direct
`(u, v, p)` Navier--Stokes API. The paper uses a stream-function formulation
with `(psi, p)` outputs, so this benchmark is a compatible adaptation rather
than an exact reproduction.

## How to run

From the repository root:

```bash
# Full benchmark: three independent runs of each model, CFD reference, then comparison
python benchmarks/comparing_pinns_qcpinn_cavity/run_benchmark.py --all

# Single-run smoke test of the full orchestrator
python benchmarks/comparing_pinns_qcpinn_cavity/run_benchmark.py --all --num_runs 1

# Reduced PINN run for checking the training loop
python benchmarks/comparing_pinns_qcpinn_cavity/benchmark_pinn.py \
    --num_runs 1 --epochs_adam 0 --epochs_lbfgs 100

# Reduced QCPINN run, if PennyLane is installed
python benchmarks/comparing_pinns_qcpinn_cavity/benchmark_qcpinn.py \
    --num_runs 1 --epochs_adam 0 --epochs_lbfgs 100

# Re-generate plots and the report from existing results
python benchmarks/comparing_pinns_qcpinn_cavity/run_benchmark.py --compare

# Small independent PINN sampling ablation (baseline, corner removal,
# and doubled boundary resolution)
python benchmarks/comparing_pinns_qcpinn_cavity/run_sampling_ablation.py

# Generate the 101x101 CFD reference and 201x201 grid-refinement check
python benchmarks/comparing_pinns_qcpinn_cavity/run_benchmark.py --reference

# Run the finite-volume solver directly on one grid
python benchmarks/comparing_pinns_qcpinn_cavity/reference_cfd.py \
    --grid 101 --output benchmarks/comparing_pinns_qcpinn_cavity/results/cfd_reference.npz
```

Prerequisites are `deepflow` from this repository and PennyLane for the
QCPINN side (`pip install pennylane`).

## Problem setup

- **Geometry**: square cavity `[0, 1] x [0, 1]`.
- **PDE**: steady incompressible 2D Navier--Stokes with `U=1`, `L=1`,
  `rho=1`, `mu=0.1`, giving `Re=10`.
- **Boundary conditions**: left, bottom, and right walls have `u=v=0`; the
  top lid has `u=1, v=0`.
- **Pressure reference**: `p=0` at the lower-left corner.
- **Sampling**: uniform grid-like sampling with 50 points per wall and a
  `[50, 50]` interior grid, matching the paper as closely as DeepFlow's
  separate area/boundary geometries allow.
- **Training**: 100 L-BFGS epochs, with no Adam warm-up, resampling, or early
  stopping. The paper reports 100 L-BFGS-style training epochs.

The independent CFD reference uses a staggered-grid finite-volume SIMPLE
solver with central-difference convection and diffusion. It runs on a
101x101 cell grid for routine comparisons and a 201x201 grid for refinement
verification. The CFD pressure is gauge-shifted so that the lower-left cell
matches the benchmark pressure reference.

## What it measures

Each run records parameter count, final total/BC/PDE losses, maximum and mean
absolute continuity and momentum residuals, optimizer timings, loss histories,
predicted `u`, `v`, `p` fields, and two standard cavity diagnostics:

- `u(y)` along the vertical centerline `x=0.5`.
- `v(x)` along the horizontal centerline `y=0.5`.

The representative field and profile plots use the median-loss run, while
scalar results are aggregated as mean +/- standard deviation across runs.

## Architecture

| Model | Configuration | Parameters |
|-------|---------------|------------|
| PINN | `PINN(width=32, length=4)` with direct `(u,v,p)` outputs | 3,363 |
| QCPINN | `QCPINN(pre=[50], post=[50], nqubits=4, q_layer_iterations=10)` | 877 |

The paper's reported 6,594-parameter PINN uses separate pressure and
stream-function models. The direct `(u,v,p)` DeepFlow adaptation therefore has
a different parameter count even with the same width and depth. Likewise,
`q_layer_iterations=10` matches the paper's variational depth but does not add
the paper's trainable quantum embedding circuit.

## Outputs

Results are written to `results/`:

- `pinn_results.npz` and `qcpinn_results.npz` -- aggregated metrics and
  representative fields/profiles.
- `compare_loss_curves.png` -- total, BC, and PDE loss curves.
- `compare_u_field.png`, `compare_v_field.png`, and `compare_p_field.png` --
  predicted fields rendered as `viridis` filled-contour maps.
- `compare_continuity_residual.png` -- absolute continuity residual fields,
  also rendered with `viridis` filled contours.
- `compare_centerline_profiles.png` -- vertical `u` and horizontal `v`
  centerline profiles.
- `cfd_reference_fields.png` -- finite-volume CFD `u`, `v`, and `p` fields.
- `compare_pinn_cfd_errors.png` -- PINN-minus-CFD field errors.
- `compare_qcpinn_cfd_errors.png` -- QCPINN-minus-CFD field errors when
  QCPINN results are available.
- `cfd_reference.npz`, `cfd_reference_201.npz`, and
  `cfd_grid_convergence.npz` -- CFD fields and grid-refinement metrics.
- `benchmark_report.md` -- generated configuration, summary metrics, and
  figure list.
- `pinn_sampling_*.npz` and `sampling_ablation_report.md` -- independent
  PINN-only sampling ablation results against the CFD reference.

The benchmark reports both training/PDE residuals and solution errors against
the independent CFD reference. The CFD result is a numerical reference, not
an analytical solution; its discretization error is estimated by comparing
the 101x101 and 201x201 solutions.
