# FP32 vs FP64 Channel-Flow Precision Benchmark

This folder contains a simple, reproducible benchmark that compares single
precision (`torch.float32`) and double precision (`torch.float64`) for 2D steady
channel flow using DeepFlow.

## Motivation

The Burgers benchmark in `benchmarks/comparing_precision` may be too small or
smooth to expose FP64 benefits. This benchmark keeps the same comparison scheme
but uses steady incompressible Navier-Stokes with coupled `u`, `v`, and `p`
outputs and second-order residuals.

## How to run

From the repository root:

```bash
python benchmarks/comparing_precision_channel_flow/benchmark_precision.py
```

For more robust statistics, average over multiple independent runs:

```bash
python benchmarks/comparing_precision_channel_flow/benchmark_precision.py --num_runs 5
```

You can also change the number of epochs:

```bash
python benchmarks/comparing_precision_channel_flow/benchmark_precision.py --epochs 500 --num_runs 3
```

## What it measures

For each precision, the script:

1. Sets `df.dtype = torch.float32` or `df.dtype = torch.float64`.
2. Builds the same 2D channel-flow problem (geometry, PDE, BCs, sampling).
3. Creates one FP32 baseline per seed, including sampled coordinates and initial model weights.
4. Casts that baseline to FP32 and FP64, then trains each with `torch.optim.LBFGS` for the configured epochs.
5. Repeats for `--num_runs` paired seeds and reports mean ± std.
6. Evaluates the selected best model on one canonical uniform `[500, 100]` grid and records the
   PDE residual fields.

The comparison table reports:

- Final total / BC / PDE loss
- Max and mean absolute PDE residual
- Max and mean absolute continuity, x-momentum, and y-momentum residuals
- Training time
- Percentage delta (`(FP64 − FP32) / FP32 × 100`)

All per-precision metrics are aggregated over paired seeds. Loss histories and
fields shown in the figures come from the same representative seed/run index for
both dtypes.

These results measure training losses and PDE residual behavior on collocation
and evaluation grids. They do not measure independent solution accuracy against
a reference solution.

## Outputs

All outputs are written to `benchmarks/comparing_precision_channel_flow/results/`:

- `fp32_results.npz` – raw FP32 metrics, fields, residuals, and loss histories
- `fp64_results.npz` – raw FP64 metrics, fields, residuals, and loss histories
- `loss_curves.png` – total / BC / PDE loss curves side by side
- `velocity_magnitude_comparison.png` – predicted speed fields and difference
- `pressure_comparison.png` – predicted pressure fields and difference
- `residual_difference.png` – FP64 minus FP32 residual-difference fields

## Interpreting results

A **negative Delta** for a loss or residual metric means FP64 is lower/better.
A **positive Delta** for training time means FP64 is slower.
