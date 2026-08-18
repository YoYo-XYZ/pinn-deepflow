# Examples

Six worked examples ship with DeepFlow, each as a Jupyter notebook in
`examples/` with a hand-authored documentation page. Every page states which
DeepFlow version it was verified against, reports a reference-run result, and
links back to its source notebook.

## Steady problems

### Burgers' Equation

Challenging because of the sharp shock front that demands adaptive resampling
(R3) to concentrate collocation points where the residual lives.

[Solve it →](burgers_eq.md) ·
[`examples/burgers_eq/burgers_eq.ipynb`](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/examples/burgers_eq/burgers_eq.ipynb)

### Lid-driven Cavity Flow

The classic benchmarking case: a square cavity with a sliding lid, used to
validate incompressible solvers at `Re = 100`.

[Solve it →](cavity_flow_steady.md) ·
[`examples/cavity_flow_steady/cavity_flow_steady.ipynb`](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/examples/cavity_flow_steady/cavity_flow_steady.ipynb)

### Flow around a Cylinder

Steady flow past a circular cylinder in a channel — boolean geometry
(rectangle minus circle) with a parabolic inflow and a steady symmetric wake.

[Solve it →](cylinder_flow_steady.md) ·
[`examples/cylinder_flow_steady/cylinder_flow_steady.ipynb`](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/examples/cylinder_flow_steady/cylinder_flow_steady.ipynb)

### Backward-facing Step

The recirculation benchmark: flow separates at a step corner and reattaches
downstream, forming a recirculation zone whose length you can check against
the literature.

[Solve it →](BFS_flow_steady.md) ·
[`examples/BFS_flow_steady/BFS_flow_steady.ipynb`](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/examples/BFS_flow_steady/BFS_flow_steady.ipynb)

## Transient problems

### Transient Channel Flow

Start-up flow in a channel: the fluid accelerates from rest to a steady
Poiseuille profile, visualized as an animation. The introduction to
time-dependent problems (`t` as a network input).

[Solve it →](channel_flow_transient.md) ·
[`examples/channel_flow_transient/channel_flow_transient.ipynb`](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/examples/channel_flow_transient/channel_flow_transient.ipynb)

### 2D Heat Equation

Transient diffusion on a square with a hot top wall — a compact, fast example
that introduces time ranges, initial conditions, and animated solutions.

[Solve it →](heat_eq.md) ·
[`examples/heat_eq/heat_eq.ipynb`](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/examples/heat_eq/heat_eq.ipynb)

---

Every example ends with an optional FEM comparison that runs when the
[`deepflow[cfd]`](install.md) extra (NGSolve backend) is installed.
