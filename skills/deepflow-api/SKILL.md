---
name: deepflow-api
description: Build complete, runnable DeepFlow PINN and PDE examples using the framework's public API and creator-style geometry-first workflow. Use whenever the user mentions DeepFlow, deepflow as df, PINNs with this repository, PDE simulation examples, geometry-attached physics, DeepFlow training, sampling, evaluation, or debugging DeepFlow code.
---

# DeepFlow public API expert

Use this skill to build concise, executable DeepFlow examples that follow the
framework's geometry-first, CFD-style workflow. Emulate the documented design
idioms of the framework; do not claim to be its author.

## Source of truth

When the DeepFlow repository is available, inspect the current public
implementation in `src/deepflow/`, then cross-check `docs/`, `examples/`, and
`tests/`. Prefer current signatures and regression tests over stale notebook
text. If the installed package differs from the repository, report the
version difference and follow the runtime package that will execute the code.

Use public functions, classes, attributes, and methods only. Do not invent
arguments, PDE names, model fields, or plotting methods. If a requested
capability is not present, say so and offer the smallest public-API workaround.

## Default workflow

Translate each problem into this sequence:

1. Choose the spatial domain and explicitly name important geometry objects.
2. Create `Area` and `Bound` objects with `df.geometry` factories or custom
   geometry.
3. Build a `ProblemDomain` with `df.domain(...)`.
4. Attach boundary conditions, initial conditions, and PDEs with
   `define_bc`, `define_ic`, and `define_pde`.
5. Configure time on every geometry that participates in a transient problem.
6. Sample training points with one of the domain sampling methods.
7. Create a `PINN` or `RFFPINN` whose input and output names match the PDE.
8. Train with Adam, optionally resample with R3, then refine the best Adam
   model with L-BFGS.
9. Evaluate on a separately sampled geometry and inspect fields and residuals.
10. Save the best model or export reference/evaluation data when requested.

Keep examples short and notebook-friendly, but include all imports, constants,
setup, training, evaluation, and persistence needed to run them. Use named
arguments for physical parameters and model settings so the code remains
readable and avoids constructor-order mistakes.

## Public API building blocks

### Geometry and domain

Use the following public geometry factories as appropriate:

```python
area = df.geometry.rectangle([x_min, x_max], [y_min, y_max])
obstacle = df.geometry.circle(x, y, radius)
segment = df.geometry.line_horizontal(y, [x_min, x_max])
segment = df.geometry.line_vertical(x, [y_min, y_max])
segment = df.geometry.line([x1, y1], [x2, y2])
area = df.geometry.polygon([x1, y1], [x2, y2], [x3, y3])
```

`Area` supports union with `|` or `+`, subtraction with `-`, and explicit
boundary attachment with `+ bound`. Use `area.show()` or
`domain.show_setup()` before assigning index-based boundary conditions. This
prevents an incorrect assumption about which boundary is inlet, outlet, wall,
or initial-condition geometry.

`df.domain(area)` automatically includes an area's boundaries. When the PDE
area should have only selected boundaries, pass `area.area_list` together with
the standalone `Bound` objects instead. Avoid adding the same geometry twice
unless the repeated reference is intentional.

Attach physics after the domain is assembled:

```python
domain.bound_list[0].define_bc({"u": 1.0, "v": 0.0})
domain.bound_list[1].define_bc({"u": 0.0, "v": 0.0})
domain.area_list[0].define_pde(
    df.pde.NavierStokes(mu=mu, rho=rho, U=U, L=L)
)
```

Boundary and initial conditions accept constants or function conditions in the
form `("x", callable)` or `["x", callable]`:

```python
domain.bound_list[0].define_bc({"u": ["y", lambda y: 4 * y * (1 - y)]})
domain.area_list[1].define_ic({"u": 0.0})
```

Use `df.parabolic(...)` when its public helper matches the requested inlet
profile. Use `df.hard_constraint(...)` only when a hard constraint is actually
appropriate and the same output field has compatible constants on all relevant
bounds.

### PDE/model variable mapping

Choose model keys from the PDE contract rather than from visual field names:

| PDE | Inputs | Outputs | Notes |
| --- | --- | --- | --- |
| `NavierStokes` | `['x', 'y']` or `['x', 'y', 't']` | `['u', 'v', 'p']` | Steady or transient 2D incompressible flow. |
| `StreamFunctionNavierStokes` | `['x', 'y']` | `['psi', 'p']` | Steady only; derives `u = psi_y` and `v = -psi_x`. |
| `BurgersEquation1D` | `['x', 'y']` | `['u']` | Steady 2D `(x, y)` form, with `y` acting as the evolution coordinate. |
| `HeatEquation` | `['x', 'y', 't']` | `['u']` | Requires time and an initial condition for the usual transient setup. |
| `WaveEquation` | `['x', 'y', 't']` | `['u']` | Requires time and second time derivatives. |
| `CustomPDE` | The coordinates plus declared outputs | Declared outputs | The callable must return a tuple of residual tensors. |

Construct models with explicit keys whenever the PDE is not the default
`u, v, p` configuration:

```python
model = df.PINN(
    input_vars=["x", "y", "t"],
    output_vars=["u", "v", "p"],
    width=40,
    length=4,
)
```

The model receives a dictionary keyed by `input_vars` and returns a dictionary
keyed by `output_vars`. Never pass a raw concatenated tensor to the standard
DeepFlow model.

### Time and sampling

For transient problems, set time on every participating geometry before
sampling. Initial-condition geometries use the start of the configured range:

```python
for geometry in domain:
    geometry.define_time(
        range_t=(0.0, 1.0),
        sampling_scheme="random",
        expo_scaling=False,
    )

domain.sampling_lhs(
    bound_sampling_res=[500, 500, 500, 500],
    area_sampling_res=[2000, 2000],
)
```

Available domain samplers are `sampling_uniform`, `sampling_random`, and
`sampling_lhs`. Residual-based workflows start with one of those samplers and
then call `sampling_R3` or `sampling_RAR` from a training callback.
`sampling_R3` maintains a roughly fixed point budget; `sampling_R3_` is the
legacy accumulating variant. Keep each resolution list aligned with the
domain's bound and area order.

Call `df.manual_seed(seed)` before stochastic geometry sampling and model
construction when reproducibility matters. `df.dtype` can be set to
`torch.float32` or `torch.float64` before constructing geometry and models;
`df.device` or `df.get_device()` controls the DeepFlow device.

### Training

Use `df.calc_loss_simple(domain)` for the standard sum of PDE, BC, and IC
losses. Use `df.calc_loss_weighted(...)` only when the problem justifies
explicit weighting. Both return a callable for the training methods.

The training methods return `(trained_model, best_model)` and internally copy
the starting model. Preserve and use the best model:

```python
loss_fn = df.calc_loss_simple(domain)

_, adam_best = model.train_adam(
    learning_rate=1e-3,
    epochs=2000,
    calc_loss=loss_fn,
    print_every=200,
)

_, best_model = adam_best.train_lbfgs(
    epochs=300,
    calc_loss=loss_fn,
    print_every=50,
)
```

Use `do_between_epochs` for R3 resampling or other small, explicit callbacks.
Use `threshold_loss` for an intentional stopping criterion. Do not promise
that a particular epoch count guarantees a particular physical accuracy;
report that convergence depends on scaling, sampling, architecture, and
hardware.

`RFFPINN` is a public alternative for problems with difficult spectral
content. Its `embed_dim` must be a positive even integer. Use it only when the
user requests the architecture or the problem benefits from Fourier features.

### Evaluation and visualization

For one geometry, evaluate after sampling it:

```python
prediction = domain.area_list[0].evaluate(best_model)
prediction.sampling_area([200, 100])
prediction.plot_color("u", cmap="jet")
prediction.plot_loss_curve()
```

`domain.evaluate(best_model)` returns a `GroupEvaluator`, not one merged data
dictionary. Use `group.area_list` and `group.bound_list` to select a child;
each child exposes its underlying geometry through `evaluator.geometry`. All
unique included geometries must have coordinates before group evaluation. For
transient visualization, call the group's `define_time(...)` after sampling
and before plotting or animating.

Evaluators expose coordinate, model-field, residual, and loss-history data via
`evaluator.data_dict` and `evaluator[key]`. Common public plots are `plot`,
`plot_color`, `plot_contour`, `plot_streamline`, `plot_distribution`,
`plot_loss_curve`, and `plot_animate`. `GroupEvaluator` retains geometry-specific
plot delegation through `geometry=...`; direct child access through
`group.area_list[i]` or `group.bound_list[i]` is also supported when different
geometries contain different fields.

For optional FEM comparison, `domain.solve_fem(...)` requires the `cfd` extra
and NGSolve. It returns a `ReferenceGroupEvaluator` (a `GroupEvaluator` whose
children re-query the FEM fields via `reference.area_list`/
`reference.bound_list`, `sampling_area(...)`, and `plot_color("u_ref")`).
`reference.metadata` carries solver diagnostics, and the underlying
`ReferenceSolution` is exposed as `reference.reference_solution` for arbitrary point queries
(`reference.reference_solution.evaluate(x, y, t=None, fields=None)`) and
`export_npz(...)`. FEM queries must lie inside the PDE area, and transient
queries need aligned time coordinates.

## Failure prevention checklist

Before presenting code, verify:

- Every PDE-required coordinate and output exists in the model key lists.
- Every boundary/initial-condition key is meaningful for the model output.
- Time is configured on all relevant geometries before sampling.
- A sampler has run before training or evaluation.
- `domain.sampling_*` resolution lists match the intended geometry order.
- R3/RAR callbacks run only after an initial sample and use valid budgets.
- The best returned model is used for evaluation and saving.
- `StreamFunctionNavierStokes` is never given a `t` input.
- Optional FEM use is clearly separated from the PINN workflow.
- The code uses current public APIs and contains no undefined names.

If the user supplies broken code, identify the first contract violation,
provide the smallest corrected runnable version, and explain why the change
fixes it. Prefer a focused fix over an unrelated refactor.

## Response format

For a new example, respond in this order:

1. State the physical and API assumptions in a few lines.
2. Provide the complete Python script or notebook-ready cells.
3. Explain the training/evaluation outputs and expected runtime qualitatively.
4. Add one focused troubleshooting note for the most likely failure mode.

Keep the result practical: a small smoke-test configuration is preferable to
silently emitting a multi-hour training run. Offer larger production sampling
and epoch values separately when appropriate.
