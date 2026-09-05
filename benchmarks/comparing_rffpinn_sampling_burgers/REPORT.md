# Burgers RFFPINN sampling benchmark

This suite runs the same RFFPINN model, Burgers domain builder, training path,
evaluator, and reporting helpers for three sampling schemes:

- `lhs`: fixed Latin hypercube samples.
- `uniform`: fixed uniform samples, including the interior grid.
- `r3`: LHS initialization followed by fixed-budget R3 resampling during Adam.

The benchmark writes native model files (`rffpinn_lhs.pkl`,
`rffpinn_uniform.pkl`, and `rffpinn_r3.pkl`), visualizer-generated field and
loss plots, and this Markdown report under `results/`. If the optional FEM
backend is available, reference errors are calculated from evaluator coordinates
and the public reference-solution evaluator.

Run the CPU-friendly smoke path with:

```powershell
python benchmarks\comparing_rffpinn_sampling_burgers\benchmark.py --smoke
```

Run the full benchmark with:

```powershell
python benchmarks\comparing_rffpinn_sampling_burgers\benchmark.py
```

Regenerate plots from the saved native models without retraining:

```powershell
python benchmarks\comparing_rffpinn_sampling_burgers\plot_results.py
```
