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
- `show_setup()`: Plots the domain geometry and boundary conditions.
- `show_coordinates(display_physics=False)`: Plots the sampled collocation points.

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
1D Burgers' equation.

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

