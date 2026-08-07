# NGSolve reference solutions

DeepFlow can optionally solve its built-in PDEs with an unstructured
NGSolve/Netgen finite-element backend.  The dependency is not imported by a
normal `import deepflow`.

Install the optional extra in a supported Python 3.10+ environment:

```bash
pip install "deepflow[cfd]"
```

The backend accepts one connected two-dimensional PDE `Area`.  Rectangles,
circles, polygons, lines, parameterized curves, and subtraction-defined holes
are supported when their boundaries are represented by `Bound` objects.  An
arbitrary `contains_fn` without explicit boundary curves cannot be meshed.

Reference boundary conditions use the existing DeepFlow value syntax:

```python
bound.define_bc({"u": 0})
bound.define_bc({"u": ("y", lambda y: 4 * y * (1 - y))})
```

These are interpreted as Dirichlet values.  Hard constraints, derivative
conditions such as `u_x`, and typed Neumann/traction/periodic conditions are
reported as unsupported in v1.  Unspecified fields use the natural FEM
condition.  A pressure field without a pressure Dirichlet condition is
normalized to zero mean after each Navier–Stokes solve; the normalization mode
is recorded in `reference.metadata["pressure_gauge"]`.

## Usage

```python
import numpy as np
import deepflow as df

area = df.geometry.rectangle([0, 1], [0, 1])
domain = df.domain(area)
for bound in domain.bound_list:
    bound.define_bc({"u": 0, "v": 0})
area.define_pde(df.NavierStokes(mu=1.0, rho=1.0))

solver = df.ReferenceSolver(mesh_size=0.05, boundary_resolution=128)
reference = solver.solve(domain)
values = reference.evaluate(np.array([0.25, 0.5]), np.array([0.5, 0.5]))
reference.export_npz("reference.npz", x=[0.25, 0.5], y=[0.5, 0.5])
```

Heat, wave, transient Navier–Stokes, and 1-D Burgers problems require a time
interval and initial data.  Set `time_step` to control the stored snapshots;
the default stores 100 uniform steps.  Transient `evaluate` calls linearly
interpolate between snapshots.  Burgers follows DeepFlow's existing convention
that `x` is space and `y` is time, so `reference.evaluate(x, y)` is a valid
transient query for that PDE.  Other transient PDEs use `t=...`.

The reference object retains the FEM fields in memory, so repeated queries do
not solve the PDE again.  `metadata` contains mesh statistics, iteration
counts, residual diagnostics, time values, and convergence status.
