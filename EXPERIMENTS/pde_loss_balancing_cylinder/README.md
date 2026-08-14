# Per-equation PDE gradient balancing: cylinder flow

This isolated experiment compares DeepFlow's `calc_loss_simple` with dynamic
gradient-norm balancing and fixed `(1, 0.05, 0.05)` equation weights for the
continuity, x-momentum, and y-momentum losses. It does not modify `src/deepflow`.

The problem matches the repository cylinder example geometry and uses the
requested parameters `U=1`, `mu=0.2`, `rho=1`, `L=1` (Reynolds number 5).

Run the default paired benchmark from the repository root. It trains for 10,000
Adam epochs, updates balancing weights every 500 completed optimizer epochs,
and generates/caches an NGSolve FEM reference:

```powershell
python EXPERIMENTS/pde_loss_balancing_cylinder/benchmark.py
```

For a longer and more robust run:

```powershell
python EXPERIMENTS/pde_loss_balancing_cylinder/benchmark.py `
  --epochs 10000 --balance-every 500 `
  --boundary-points 1000 --interior-points 4000 `
  --evaluation-points 20000 --seeds 69 70 71
```

`--scope full` measures gradients over every trainable parameter. Use
`--scope last_layer` for a cheaper proxy. `--balance-every 500` is the default.
The standalone training loop binds updates to completed optimizer epochs, not
loss-function call counts. At each update, the instantaneous inverse-gradient
weights are blended with the previous weights using `alpha=0.9`:

```text
weights_new = 0.9 * weights_old + 0.1 * weights_instantaneous
```

Weights are clipped, detached, normalized to mean one, and frozen between
updates.

The report compares raw, unweighted PDE residual MSE on a fresh paired
collocation sample and compares `u`, `v`, `p`, and velocity magnitude against
the same FEM solution coordinates. The dynamically weighted objective is not
used as an accuracy metric. The FEM cache includes the physical parameters,
mesh size, and grid shape and is regenerated when those settings change.

`results/pde_weight_history.png` shows the piecewise-constant weight assigned to
each PDE equation. It includes the initial equal weights and every 500-epoch
update.

Completed methods are cached with their configuration. For example, adding or
rerunning only the fixed method does not retrain the other two:

```powershell
python EXPERIMENTS/pde_loss_balancing_cylinder/benchmark.py --methods fixed
```

Pass `--force` only when the selected method should be retrained.
