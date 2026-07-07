# FP32 vs FP64 Precision Benchmark

This folder contains a simple, reproducible benchmark that compares single
precision (`torch.float32`) and double precision (`torch.float64`) for the
1D Burgers equation using DeepFlow.

## Motivation

Recent literature reports that FP64 can dramatically improve PINN training for
certain problems. This benchmark lets you measure whether the same effect
appears for the standard DeepFlow Burgers setup.

## How to run

From the repository root:

```bash
python benchmarks/comparing_precision/benchmark_precision.py
```

For more robust statistics, average over multiple independent runs:

```bash
python benchmarks/comparing_precision/benchmark_precision.py --num_runs 5
```

You can also change the number of epochs:

```bash
python benchmarks/comparing_precision/benchmark_precision.py --epochs 500 --num_runs 3
```

## What it measures

For each precision, the script:

1. Sets `df.dtype = torch.float32` or `df.dtype = torch.float64`.
2. Builds the same 1D Burgers problem (geometry, PDE, BCs, sampling).
3. Trains a `df.PINN(width=16, depth=4)` with `torch.optim.LBFGS` for the configured epochs.
4. Repeats for `--num_runs` seeds and reports mean ± std.
5. Evaluates the trained model on a uniform `[500, 250]` grid and records the
   PDE residual field.

The comparison table reports:

- Final total / BC / PDE loss
- Max and mean absolute PDE residual
- Training time
- Percentage delta (`(FP64 − FP32) / FP32 × 100`)

All per-precision metrics are aggregated over `--num_runs` independent seeds.
Loss histories and fields shown in the figures come from the median-loss run.

## Outputs

All outputs are written to `benchmarks/comparing_precision/results/`:

- `fp32_results.npz` – raw FP32 metrics and loss histories
- `fp64_results.npz` – raw FP64 metrics and loss histories
- `loss_curves.png` – total / BC / PDE loss curves side by side
- `u_field_comparison.png` – predicted `u` field and `FP64 − FP32` difference

## Interpreting results

A **negative Delta** for a loss or residual metric means FP64 is lower/better.
A **positive Delta** for training time means FP64 is slower.

On the default 200-epoch LBFGS Burgers setup, results are mixed:

- **Total / BC / PDE loss** are usually comparable, with FP32 occasionally
  reaching a lower final loss but showing higher run-to-run variance.
- **PDE residual** (especially `max |PDE residual|`) is consistently lower with
  FP64, because the higher precision stabilizes derivative calculations and the
  strong-Wolfe line search.
- **Training time** with LBFGS is often similar or even slightly faster in FP64,
  since the optimizer's internal line-search iterations can converge more
  reliably at higher precision.

For harder / stiffer problems, the FP64 advantage is expected to grow.
