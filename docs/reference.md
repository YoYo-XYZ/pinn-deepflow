# API Reference

This page lists the main classes and functions available in `deepflow`.

## Domain

The domain module handles the creation and management of the computational domain, including geometry, boundary conditions, and sampling.

### `domain`

```python
def domain(*geometries) -> ProblemDomain
```

Creates a `ProblemDomain` instance from a list of `Area` or `Bound` objects.

**Arguments:**
- `*geometries`: Can be single `Area`/`Bound` objects or lists of them.

### `ProblemDomain`

```python
class ProblemDomain(bound_list, area_list):
```
The main class managing the physics problem.

**Methods:**
- `sampling_uniform(bound_sampling_res, area_sampling_res)`: Samples points uniformly.
- `sampling_random(bound_sampling_res, area_sampling_res)`: Samples points randomly.
- `sampling_lhs(bound_sampling_res, area_sampling_res)`: Samples points using Latin Hypercube Sampling.
- `sampling_R3(bound_sampling_res, area_sampling_res)`: Samples points using R3 refinement.
- `evaluate(model)`: Returns a structured `GroupEvaluator` for all unique sampled geometries.
- `solve_fem(...)`: Solves the attached PDE with the optional NGSolve backend
  and returns a `ReferenceSolution` for direct point queries.
- `show_setup()`: Plots the domain geometry and boundary conditions.
- `show_coordinates(display_physics=False)`: Plots the sampled collocation points.

The primary FEM workflow is:

```python
reference = domain.solve_fem(
    mesh_size=0.05,
    boundary_resolution=64,
    max_iterations=200,
)
values = reference.evaluate([0.25, 0.5], [0.5, 0.5])
```

The FEM solve does not require DeepFlow point samples. The returned
`ReferenceSolution` evaluates the FEM fields at any requested coordinates and
exposes solver metadata through `reference.metadata`.

### `calc_loss_simple`

```python
def calc_loss_simple(domain: ProblemDomain) -> callable
```
Returns a loss function that calculates the weighted sum of boundary and PDE losses for the given domain.

## Geometry

The `deepflow.geometry` module provides helper functions to create 1D and 2D geometries.

### `rectangle`
```python
def rectangle(range_x: List[float], range_y: List[float]) -> Area
```
Creates a rectangular area.

### `circle`
```python
def circle(x: float, y: float, r: float) -> Area
```
Creates a circular area.

### `Bound`
Represents a boundary (e.g., line segment).

### `Area`
Represents a 2D area. Supports subtraction (e.g., `rect - circle`).

## PDE

The `deepflow.pde` module contains standard Partial Differential Equations.

### `NavierStokes`
```python
class NavierStokes(U, L, mu, rho)
```
2D Incompressible Navier-Stokes equations.

### `StreamFunctionNavierStokes`
```python
class StreamFunctionNavierStokes(mu, rho, U=1.0, L=1.0)
```
Steady 2D incompressible Navier-Stokes equations with model outputs `psi` and
`p`. The velocity is derived as `u = psi_y` and `v = -psi_x`, so continuity is
satisfied identically. The model can be constructed as follows:

```python
pinn = df.PINN(input_vars=["x", "y"], output_vars=["psi", "p"])
pde = df.StreamFunctionNavierStokes(mu=0.01, rho=1.0)
```

This formulation returns the two momentum residuals and currently supports
steady problems only. Derived `u` and `v` fields are available during PDE
evaluation for plotting streamlines.

### `BurgersEquation1D`
```python
class BurgersEquation1D(nu)
```
Steady 2D Burgers equation in the `(x, y)` domain,
`u_y + u * u_x = nu * u_xx`.

### `HeatEquation`
```python
class HeatEquation(alpha)
```
2D Heat equation.

## Neural Network

### `PINN`

```python
class PINN(width, length, input_vars, output_vars, activation=nn.Tanh())
```
Physics-Informed Neural Network model.

**Methods:**
- `train_adam(calc_loss, learning_rate, epochs, ...)`: Train using Adam optimizer.
- `train_lbfgs(calc_loss, epochs, ...)`: Train using L-BFGS optimizer.
- `save_as_pickle(path)`: Save model.
- `load_from_pickle(path)`: Load model.

### `RFFPINN`

```python
class RFFPINN(
    input_vars=None,
    output_vars=None,
    width=32,
    length=4,
    embed_dim=256,
    alpha=5.0,
    activation=nn.Tanh(),
    weight_init="kaiming",
)
```

Physics-Informed Neural Network with a fixed Joint Random Fourier Feature
embedding. For a stacked coordinate vector `x`, the model samples
`B ~ Normal(0, alpha²)` with shape
`(len(input_vars), embed_dim // 2)` and replaces the raw coordinates with
`[cos(x @ B), sin(x @ B)]` before the first dense layer. `embed_dim` must be a
positive even integer. The frequency matrix is fixed during training and is
reproducible through `df.manual_seed(...)`.

```python
df.manual_seed(69)
model = df.RFFPINN(
    width=50,
    length=5,
    input_vars=["x", "y", "t"],
    output_vars=["u", "v", "p"],
    embed_dim=256,
    alpha=5.0,
)
```

## Evaluation

### `Evaluator` (Visualizer)

Returned by `domain.area_list[i].evaluate(model)`.

**Methods:**
- `sampling_area(res_list)`: Sample points for visualization.
- `plot(key)`: Plot a variable.
- `plot_color(key)`: Plot a variable as a color map.
- `plot_streamline(u, v)`: Plot streamlines.
- `plot_loss_curve()`: Plot loss history.
- `plot_animate(...)`: Create animation (for transient problems).

### `GroupEvaluator`

Returned by `domain.evaluate(model)`.

`GroupEvaluator` keeps one `Evaluator` per unique geometry instead of merging
fields from different physics types into one data dictionary.

```python
results = domain.evaluate(model)

area_prediction = results.area_evaluators[0]
bound_prediction = results.get_evaluator(domain.bound_list[0])
```

**Attributes and methods:**
- `area_evaluators`: Evaluators aligned with the unique entries in `domain.area_list`.
- `bound_evaluators`: Evaluators aligned with the unique entries in `domain.bound_list`.
- `get_evaluator(geometry)`: Select an evaluator using the original geometry object.
- `postprocess()` / `refresh()`: Recompute all child results after model or sample changes.
- `sampling_area(...)` and `sampling_line(...)`: Broadcast sampling to compatible child geometries.
- `define_time(...)`: Broadcast time-coordinate configuration to all child geometries.
- `plot_color("u")` / `plot_scatter("u")`: Plot all child geometries that contain `u` in one combined scatter plot.
- `plot_color("u", geometry=area)`: Delegate visualization to one selected child evaluator.

All included geometries must have sampled coordinates before
`domain.evaluate(model)` is called.

Aggregate color plots skip child evaluators that do not contain the requested
field or coordinate keys. The temporary concatenated data is used only for the
plot and is not stored on `GroupEvaluator`.

`ReferenceSolver` and `ReferenceSolution` are available for direct point
queries and `export_npz()`:

```python
values = reference.evaluate(x, y, t=t)
reference.export_npz("reference.npz", x=x, y=y, t=t)
```

