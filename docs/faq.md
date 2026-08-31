# FAQ

## When should I use FP64?

Recent PINN research shows that double precision improves convergence and
accuracy, at the cost of speed and memory. Switch the entire pipeline —
model weights, sampled coordinates, and PDE residuals — before building
anything:

```python
import torch
import deepflow as df

df.dtype = torch.float64

# ... define geometry, PDE, sample, build model, and train as usual
```

Switch back with `df.dtype = torch.float32`. Only `float32` and `float64`
are supported. Reach for FP64 when a model converges to a plateau that is
still above the target loss, or when residuals at sharp gradients look noisy.

## CPU or GPU?

DeepFlow auto-selects `'cuda'` when available (`df.device` reports the
choice). Force the CPU with:

```python
df.device = 'cpu'
```

CPU works for all examples but expect order-of-magnitude slower training —
the reference runs on the example pages used a GPU.

## Which sampling method should I use?

| Method | When |
|---|---|
| `sampling_uniform` / `sampling_random` | Quick first experiments on simple domains |
| `sampling_lhs` | Default choice for a good static initial point set |
| `sampling_RAR` | Add points where the residual is largest (top-k of random candidates) |
| `sampling_R3` | **Recommended for training**: re-sample collocation points proportionally to the residual, replacing the old set (see [arXiv:2207.02338](https://arxiv.org/abs/2207.02338)) |
| `sampling_R3_` | Like R3 but the residual itself is used as the sampling density (see [arXiv:2301.04744](https://arxiv.org/abs/2301.04744)) |
| `sampling_accumulate` | Keep the old points and add new ones |

A static point set is a fixed resource; for problems with moving features
(shocks, fronts, wakes) use R3 inside a training callback:

```python
def do_in_adam(epoch, model):
    if epoch % 500 == 0 and epoch > 0:
        domain.sampling_R3([2000, 1000, 1000], [4000])
```

## My loss is stuck. What should I try first?

1. **Check the training schedule** — the working pattern is Adam first (lr
   ~1e-3–1e-2) to get into the basin, then L-BFGS to polish. L-BFGS alone
   from a bad start often stalls.
2. **Resample** — static points can over- or under-cover sharp features;
   switch on R3 with a moderate frequency.
3. **Lower the learning rate** — if Adam oscillates (loss jumps up after
   long flat stretches), try `learning_rate=1e-4`–`1e-3`.
4. **Switch to FP64** — the fastest fix for residual plateaus.
5. **Give L-BFGS a threshold and budget** — `threshold_loss` stops early
   when converged; a too-small `epochs` leaves it mid-run.
6. **Inspect the loss terms** — print the per-term losses (BC vs PDE vs IC).
   A stuck BC term usually means a mis-specified boundary condition or an
   impossible-to-satisfy constraint, not an optimization problem.

## What is the difference between hard and soft boundary conditions?

Conditions are soft by default. To request a constant hard condition, wrap its
value with `df.hard_constraint(...)`:

```python
domain.bound_list[0].define_bc({"u": df.hard_constraint(0.0)})
```

When using a domain loss function, DeepFlow configures supported hard
constraints automatically. Straight graph-like boundaries and area-based
initial conditions are supported; area initial conditions require a `t` model
input and a time range. Arbitrary parametric curves currently remain soft.

Hard constraints are represented in the model output, so their residual is
zero when active. If a hard constraint is used directly without a domain loss,
it remains in the residual rather than being silently discarded. Derivative,
function-valued, and unsupported geometry conditions use soft treatment.

## Can I define a custom PDE?

Yes. Subclass `df.pde.PDE`, implement `compute_residuals(inputs_dict)`, and
store residuals in `self.var` / return them as tensors:

```python
import torch
import deepflow as df

class MyPDE(df.pde.PDE):
    def compute_residuals(self, inputs_dict):
        # inputs_dict has coordinates and predicted fields
        ...
        return (residual,)

domain.area_list[0].define_pde(MyPDE())
```

See `src/deepflow/pde.py` for the built-in implementations to model yours on.
Custom PDEs with hard BCs need the boundary condition syntax to know which
field constrains which coordinate.

## How do I use the FEM reference backend?

```bash
pip install "deepflow[cfd]"
```

The NGSolve/Netgen backend solves DeepFlow's built-in PDEs on unstructured
meshes and returns a `ReferenceGroupEvaluator` with per-geometry fields
(`u_ref`, `v_ref`, ...). See the FEM comparison section at the end of every
[example page](examples.md).

## Why does `define_time` warn about `expo_scaling`?

`expo_scaling` is an experimental flag for exponential time scaling; it
defaults to `False` and warns until you set it explicitly:

```python
g.define_time(range_t=[0, 1], sampling_scheme='random', expo_scaling=False)
```

## How do I force reproducibility?

```python
df.manual_seed(69)
```

This seeds the global RNGs (including sampling). Training on a GPU may still
introduce nondeterminism from non-deterministic kernels.
