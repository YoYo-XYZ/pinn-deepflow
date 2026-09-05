# Solving the Backward-Facing Step Flow (Steady)

This example solves steady flow over a backward-facing step — a standard
benchmark for separated flows. After the step, the flow detaches and forms a
recirculation zone whose length is a classic validation metric.

!!! note "At a glance"
    - **Physics**: steady incompressible Navier–Stokes equations,
      nondimensionalized with `(U, L) = (0.0001, 1)` and `mu = 0.001`,
      `rho = 1000` — Reynolds number `Re = ρUL/μ = 100`. Parabolic inflow
      on the upper inlet (`y ∈ [0.6, 1]`), no-slip walls (including the step
      faces), `p = 0` at the outlet.
    - **What to expect**: the boundary layer separates at the step corner and
      reattaches further downstream, enclosing a recirculation zone. The
      reattachment length is the number to verify against the literature.
    - **Setup cost**: FNN 50 × 5 (outputs `u, v, p`); Adam 2,000 epochs
      (lr 0.001) followed by L-BFGS 350 epochs (threshold 1e-4). Reference
      run: a few minutes on GPU; L-BFGS reached `total_loss ≈ 2.5e-3` when
      its epoch budget ran out.
    - **Verified against**: DeepFlow v0.1.3 (candidate source commit `9f751fe`).

## 1. Define the geometry

A six-vertex polygon describes the channel with the step:

```python
import deepflow as df

df.manual_seed(69)  # for reproducibility

rectangle = df.geometry.polygon(
    [0, 0.4], [0, 1], [5, 1], [5, 0], [1, 0], [1, 0.4]
)

domain = df.domain(rectangle)
domain.show_setup()
```

![Domain setup](static/examples/bfs/domain_setup.png)

## 2. Define the physics

```python
domain.bound_list[0].define_bc({'u': 0, 'v': 0})                          # Step face
domain.bound_list[1].define_bc({'u': df.parabolic_func('y', 0.6, 1, 0.7), 'v': 0})  # Inlet
domain.bound_list[2].define_bc({'u': 0, 'v': 0})                          # Top wall
domain.bound_list[3].define_bc({'p': 0})                                  # Outlet
domain.bound_list[4].define_bc({'u': 0, 'v': 0})                          # Bottom wall
domain.bound_list[5].define_bc({'u': 0, 'v': 0})                          # Step riser
domain.area_list[0].define_pde(df.NavierStokes(U=0.0001, L=1, mu=0.001, rho=1000))
domain.show_setup()
```

![Physics setup](static/examples/bfs/physics_setup.png)

## 3. Sample training data

```python
domain.sampling_lhs(bound_sampling_res=[1000, 1000, 1000, 1000, 1000, 1000], area_sampling_res=[2000])
domain.show_coordinates(display_physics=False)
```

![Sampled coordinates](static/examples/bfs/sampling_coordinates.png)

## 4. Train the model

```python
def do_in_adam(epoch, model):
    if epoch % 100 == 0 and epoch > 0:
        domain.sampling_R3(bound_sampling_res=[1000, 1000, 1000, 1000, 1], area_sampling_res=[2000])
        print(domain)

model0 = df.PINN(width=50, length=5, input_vars=['x', 'y'], output_vars=['u', 'v', 'p'])

model1, model1_best = model0.train_adam(
    learning_rate=0.001,
    epochs=2000,
    calc_loss=df.calc_loss_simple(domain),
    threshold_loss=0.005,
    do_between_epochs=do_in_adam)

model2, model2_best = model1_best.train_lbfgs(
    calc_loss=df.calc_loss_simple(domain),
    epochs=350,
    threshold_loss=0.0001)
```

!!! note "Reference-run result"
    Adam (2,000 epochs) reduced `total_loss` from ≈ 55 to ≈ 0.5; L-BFGS
    (350 epochs) reached `total_loss ≈ 2.5e-3` — its epoch budget was
    exhausted before the 1e-4 threshold. Increase `epochs` in
    `train_lbfgs` or lower `learning_rate` in Adam to push further.

```python
domain.show_coordinates()
```

![R3-densified coordinates](static/examples/bfs/resampled_coordinates.png)

Save (and reload) the trained model:

```python
model2.save("model.pt")
model2 = df.load_model("model.pt")
```

## 5. Visualize the solution

```python
df.Visualizer.refwidth_default = 7

area_eval = domain.area_list[0].evaluate(model2)
area_eval.sampling_area([300, 60])
print(area_eval)

area_eval['v_mag'] = (area_eval['u']**2 + area_eval['v']**2)**0.5

_ = area_eval.plot_color('v_mag', s=2, cmap='jet').savefig("colorplot_v_mag.png")
_ = area_eval.plot_color('u', s=2, cmap='rainbow')
_ = area_eval.plot_color('v', s=2, cmap='rainbow')
_ = area_eval.plot_color('p', s=2, cmap='rainbow')
_ = area_eval.plot_streamline('u', 'v', cmap='jet')
_ = area_eval.plot_color('pde_residual', s=2, cmap='viridis')
```

![Velocity magnitude](static/examples/bfs/velocity_magnitude.png)
![u field](static/examples/bfs/u_field.png)
![v field](static/examples/bfs/v_field.png)
![Pressure field](static/examples/bfs/pressure_field.png)
![Streamlines](static/examples/bfs/streamlines.png)
![PDE residual](static/examples/bfs/pde_residual.png)

The streamlines reveal the recirculation zone behind the step — the hallmark
of this benchmark. Residuals concentrate at the step corner and along the
reattaching shear layer.

```python
_ = area_eval.plot_loss_curve(log_scale=True)
```

![Loss curve](static/examples/bfs/loss_curve.png)

### Export data

```python
import numpy as np

x_data = area_eval.data_dict['x']
y_data = area_eval.data_dict['y']
u_data = area_eval.data_dict['u']

array = np.column_stack((x_data, y_data, u_data))
np.savetxt('outlet_velocity.txt', array)
```

## 6. Optional: FEM reference comparison

```python
try:
    import ngsolve  # noqa: F401
except (ImportError, OSError) as exc:
    print(f'Skipping optional FEM comparison: {exc}')
else:
    import numpy as np

    fem_reference = domain.solve_fem(
        mesh_size=0.2,
        boundary_resolution=32,
        max_iterations=20,
    )
    fem_area = fem_reference.area_list[0]
    fem_area.sampling_area([100, 40])
    fem_values = fem_area['u_ref']
    pinn_area = domain.area_list[0].evaluate(model2_best)
    u_error = float(np.mean(np.abs(pinn_area.data_dict['u'] - fem_values)))
    if not np.isfinite(u_error):
        raise RuntimeError('FEM comparison produced a non-finite error')
    print(f'FEM mean absolute u error: {u_error:.6e}')
```

## Run it yourself

Source notebook:
[`examples/BFS_flow_steady/BFS_flow_steady.ipynb`](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/examples/BFS_flow_steady/BFS_flow_steady.ipynb)

```bash
jupyter notebook examples/BFS_flow_steady/BFS_flow_steady.ipynb
```

This page is hand-authored; the notebook is the source of truth (see
[REGENERATE.md](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/REGENERATE.md)).
