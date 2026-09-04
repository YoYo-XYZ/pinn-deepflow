# Burgers PINN vs RFFPINN benchmark

This suite compares the standard DeepFlow `PINN` with `RFFPINN` on the shared
one-dimensional Burgers domain. Both variants use the same geometry, PDE,
sampling budget, optimizer schedule, seed, evaluation grid, and report path.
The only model-specific settings are the RFF embedding dimension (`256`) and
frequency scale (`5.0`).

The full run uses LHS sampling followed by fixed-budget R3 resampling during
Adam, then L-BFGS. If the optional FEM backend is available, the report also
includes errors obtained by querying the FEM reference at the model evaluator
coordinates. Without that backend, the report still contains evaluator
residuals and training-history metrics.

Run the CPU-friendly smoke path with:

```powershell
python benchmarks\comparing_rff_pinn_burgers\benchmark.py --smoke
```

Run the full benchmark with:

```powershell
python benchmarks\comparing_rff_pinn_burgers\benchmark.py
```

The benchmark writes native model files (`pinn.pkl` and `rffpinn.pkl`), field
and loss plots produced by the shared visualizer helper, and this Markdown
report under `results/`. To regenerate those plots from the native models:

```powershell
python benchmarks\comparing_rff_pinn_burgers\plot_results.py
```
