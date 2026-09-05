# FP32 vs FP64 channel-flow benchmark

This suite compares the same DeepFlow PINN in single precision and double
precision on steady incompressible Navier-Stokes flow through a rectangular
channel.

Run it from the repository root:

```bash
python benchmarks/comparing_precision_channel_flow/benchmark_precision.py
```

Use the CPU-friendly smoke path while changing the benchmark:

```bash
python benchmarks/comparing_precision_channel_flow/benchmark_precision.py --smoke
python benchmarks/comparing_precision_channel_flow/smoke_test.py
```

The suite uses the shared channel domain builder, including its boundary-point
rule. The shared precision harness builds one FP32 sampled/model baseline per
seed, casts a copy for each precision, trains both variants, evaluates them on
the same grid, and reports metrics from the evaluator and loss history. Use
`--num_runs 5` for paired seeds or `--epochs 500` to run L-BFGS for more
epochs.

Outputs are native DeepFlow model files, evaluator plots, and `REPORT.md` in
`results/`. The checked-in numeric archives from the former benchmark are not
part of the new workflow.
