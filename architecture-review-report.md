# Architecture Review Report: DeepFlow PINN Framework

## 1. Scope and Goals

**Scope:** A comprehensive architecture review of the DeepFlow physics-informed neural network (PINN) framework, focusing on the core source code under `src/deepflow/`.

**In scope:**
- `src/deepflow/__init__.py`
- `src/deepflow/domain.py`
- `src/deepflow/evaluation.py`
- `src/deepflow/geometry.py`
- `src/deepflow/nn.py`
- `src/deepflow/pde.py`
- `src/deepflow/physicsinformed.py`
- `src/deepflow/utility.py`
- `src/deepflow/visualization.py`

**Out of scope:**
- `src/deepflow/qnn.py` (excluded per request)
- Examples, notebooks, benchmarks, documentation, and tests

**Focus areas:**
- Code structure and modularity
- Internal dependencies and data flow
- Training and autograd correctness
- Geometry and sampling robustness
- Performance and scalability risks
- Maintainability and API stability

**Constraints:** Read-only review; no source code changes unless recommendations are requested separately.

---

## 2. Structural Overview

DeepFlow is a compact PyTorch-based PINN library built around a geometry-first, CFD-solver-style workflow. The code is organized into roughly four layers:

| Layer | Files | Responsibility |
|-------|-------|--------------|
| **Utilities / primitives** | `utility.py` | Random-seed management, device selection, autograd helpers (`calc_grad`, `calc_grads`), tensor conversion. |
| **Geometry & spatial DSL** | `geometry.py`, `domain.py` | Construct 2D domains from `Bound`/`Area`/`CustomData` objects, sample points, apply boolean operations, and visualize setups. |
| **Physics & model coupling** | `physicsinformed.py`, `pde.py`, `nn.py` | Attach boundary/initial conditions and PDEs to geometries, define network architectures, compute residuals. |
| **Training & evaluation** | `domain.py`, `nn.py`, `evaluation.py` | Optimizers (Adam, L-BFGS), batched loss assembly, post-processing, animation, and result extraction. |
| **Visualization** | `visualization.py`, `evaluation.py` | Plotting wrappers built on `ultraplot` with a nominal `matplotlib` fallback. |

### Public API Surface

The public API is exposed almost entirely through wildcard imports in `__init__.py`:

- **Geometry factories:** `Area`, `Bound`, `CustomData`, `rectangle`, `circle`, `polygon`, `line`, `curve`, `point`, `custom_data`
- **Domain orchestrator:** `domain()`
- **Network classes:** `PINN`, `FNN`, `NN`
- **PDE classes:** `NavierStokes`, `HeatEquation`, `WaveEquation`, `BurgersEquation1D`, `CustomPDE`, `PDE`
- **Utilities:** `device`, `get_device`, `manual_seed`, `latin_hypercube_sampling`
- **Visualization:** `Visualizer`, `Evaluator`
- **Condition helpers:** `function`/`func`, `parabolic_func`/`parabolic`, `HardConstraint`/`hard_constraint`

### Main Complexity Centers

1. **Geometry boolean masking and auto-orientation** (`geometry.py`, `domain.py`) — converting boundaries into point-in-domain masks via `reject_above` and ray-casting heuristics.
2. **Batched loss with higher-order derivatives** (`domain.py:325–393`) — grouping geometries by `physics_type`, concatenating inputs, slicing outputs, and preserving leaf tensors for autograd.
3. **Physics/model coupling** (`physicsinformed.py`) — a single `PhysicsAttach` base class manages `Bound`, `Area`, and `CustomData`, mixing coordinate sampling, PDE residual computation, and adaptive sampling state.

### Lines of Code (approximate)

| File | Approx. Lines |
|------|---------------|
| `geometry.py` | 500 |
| `physicsinformed.py` | 400 |
| `nn.py` | 350 |
| `domain.py` | 350 |
| `visualization.py` | 170 |
| `evaluation.py` | 120 |
| `pde.py` | 210 |
| `utility.py` | 110 |
| `__init__.py` | 30 |

---

## 3. Dependency and Data Flow Analysis

### External Dependencies

- **PyTorch** — core tensor operations, autograd, neural network modules, optimizers.
- **NumPy** — numerical conversions and array handling in evaluation/visualization.
- **SciPy** — `scipy.stats.qmc.LatinHypercube` for LHS, `scipy.interpolate.griddata` for contour/streamline interpolation.
- **Matplotlib** — plotting (required, but `ultraplot` is preferred).
- **UltraPlot** — enhanced plotting API; `matplotlib` is nominally a fallback.
- **SymPy** — LaTeX formatting of condition functions in `show_setup`.

### Internal Dependency Graph

```text
utility.py
    ↑
    ├── geometry.py ──► physicsinformed.py ──► evaluation.py ──► visualization.py
    │       ↑              ↑
    │       └──────────────┘
    │                      │
    ├── nn.py ◄────────────┘
    │       ↑
    ├── pde.py ◄─────────────┘
    │
    └── domain.py ◄──────────┘

__init__.py imports all of the above via wildcard imports.
```

### Data Flow (Training Loop)

1. **Geometry construction:** User calls `rectangle(...)`, `circle(...)`, etc., producing `Area`/`Bound` objects that inherit from `PhysicsAttach`.
2. **Domain assembly:** `domain(*geometries)` aggregates `Bound`/`Area` instances into a `ProblemDomain`.
3. **Sampling:** `ProblemDomain.sampling_*()` calls `Bound.sampling_line()` / `Area.sampling_area()`, which populate `self.X`, `self.Y` tensors.
4. **Coordinate processing:** `process_coordinates()` moves `X`/`Y` to the target device, detaches them, and sets `requires_grad_()` to create fresh leaf tensors (`X_`, `Y_`, `T_`). This is essential for higher-order PDE derivatives after resampling.
5. **Loss computation:** `calc_loss_simple(domain)` returns a closure that calls `ProblemDomain._batched_loss(model)`:
   - Geometries are grouped by `physics_type` (`BC`, `IC`, `PDE`).
   - Inputs are concatenated per group; one forward pass is performed per group.
   - Outputs are sliced back per geometry; `model_inputs` is set to the original leaf tensors (not the concatenated slice) so that `torch.autograd.grad` can compute second derivatives.
   - `_compute_residual_field()` computes residuals per geometry, and losses are summed by physics type.
6. **Training:** `NN.train_adam()` or `NN.train_lbfgs()` uses the closure, records history, and tracks the best model.
7. **Evaluation:** `geometry.evaluate(model)` returns an `Evaluator`, which runs a final forward pass and converts outputs/residuals to NumPy for visualization.

### Key Design Decisions

- **Single inheritance base (`PhysicsAttach`)** for `Bound`, `Area`, and `CustomData`. This reduces duplication but places condition handling, time handling, residual computation, and adaptive sampling into one class.
- **Batched forward pass** reduces forward passes from *N geometries* to *N physics types*.
- **Leaf-tensor preservation** in `process_coordinates` and `_batched_loss` is correct and necessary for higher-order PDE derivatives.

---

## 4. Performance and Scalability Assessment

### Strengths

- **Batched forward passes:** `_batched_loss` groups geometries by physics type and performs one forward pass per group, which is significantly more efficient than per-geometry forward passes for problems with many boundaries.
- **Autograd-aware design:** Fresh leaf tensors are created via `detach().requires_grad_()` to avoid stale `grad_fn` references after R3 resampling, which would otherwise break higher-order derivatives.

### Risks and Observations

- **Unbounded `sampling_line(10000)` in `_postprocess`:** `Bound._postprocess()` and `CustomData._postprocess()` sample 10,000 points solely to compute bounding boxes and centers. This is expensive and creates unnecessary tensors during geometry construction.
- **Geometry masking is O(N_points × N_bounds):** `Area.sampling_area()` evaluates every boundary mask for every candidate point. For complex domains with many boundaries, this scales quadratically in the number of boundaries and could become a bottleneck.
- **No batching across physics types:** While geometries are batched within a physics type, there is still one forward pass per physics type. For very large 3D or time-dependent problems, even these batches may exceed GPU memory.
- **No memory management for large LHS grids:** `latin_hypercube_sampling` uses `scipy.stats.qmc`, which materializes the full sample array before converting to a PyTorch tensor. This is fine for 2D problems but could be memory-intensive for higher-dimensional sampling.
- **Pickle serialization:** `save_as_pickle` / `load_from_pickle` use `pickle`, which is not portable across Python versions and can be unsafe with untrusted files. No alternative format (e.g., TorchScript, state dicts) is provided.
- **Adaptive sampling (RAR/R3):** Containers accumulate residual-based point tensors. `apply_residual_based_points` repeatedly concatenates, so training set sizes can grow unbounded across adaptive steps. No cap or deduplication is applied.
- **Performance claims are unmeasured:** The README notes GPU acceleration, but no profiling or benchmarking code is present in the reviewed source files. This review qualifies performance observations only from static code inspection.

---

## 5. Findings

| # | Area | Severity | Observation | Risk | Recommendation | Effort |
|---|------|----------|-------------|------|----------------|--------|
| 1 | `nn.py` L-BFGS | **Critical** | `train_lbfgs` has a latent `NameError`. When `total_loss` is NaN, the closure prints a warning and returns `0.0` without populating `loss_dict_container`. The code then checks `threshold_loss and total_loss_num < threshold_loss` outside the `else` branch, but `total_loss_num` is only defined inside the `else` branch. If a NaN occurs, `total_loss_num` is unbound and raises `NameError`. | Training crashes on NaN loss instead of handling it gracefully. | Move the `total_loss_num` extraction and threshold check into the `else` branch (or initialize `total_loss_num = float('inf')` before the branch). Also break the epoch loop on NaN rather than reinitializing the optimizer silently. | Small |
| 2 | `domain.py` R3 sampling | **Critical** | `sampling_R3` computes `res - len(self.bound_list[i].X_residual_container[0])` and passes it to `sampling_line`. If `X_residual_container` is empty or already contains many points, the result can be negative, causing `torch.linspace` / uniform sampling to fail or raise an error. | Adaptive sampling can crash depending on prior residual state. | Validate `n_points` before calling `sampling_line`; ensure it is non-negative. If the residual container is empty, default to `res`. | Small |
| 3 | `physicsinformed.py` RAR threshold | **Critical** | `get_residual_based_points_threshold` with `maintain_points=True` masks `mask[:(self.residual_field.shape[0] - self._amounts_before_add)] = False`. This assumes residual points are appended at the end of `self.X`, but the indexing logic is brittle and may suppress newly added points or fail when `_amounts_before_add` is larger than the residual field size. | Adaptive sampling may exclude valid high-residual points or include stale ones, silently degrading training. | Replace the offset-based mask with an explicit set/distance-based deduplication check, or maintain a persistent "added" mask keyed by point identity. | Medium |
| 4 | `geometry.py` boolean masking | **High** | The `checkbound()` method uses ray-casting from the area center and assigns `reject_above` based on sorted boundary intersections rounded to two decimals. It is fragile for non-convex shapes, intersecting edges, and boundaries not aligned with the casting axis. `mask_area` only handles `x` and `y` and assumes a 2D spatial domain. | Complex or non-convex geometries may produce incorrect interior masks, leading to invalid training data. | Document the convexity assumption; add explicit validation and test cases; consider a robust point-in-polygon test (e.g., shapely or winding number) for production use. | Medium |
| 5 | `visualization.py` plotting | **High** | The code imports `ultraplot as plt` and uses `plt.subplot(refwidth=...)`, `ax.format(...)`, and `fig.colorbar(...)` calls. The `except ImportError` fallback to `matplotlib.pyplot` is broken because these APIs are `ultraplot`-specific and do not exist in standard Matplotlib. | Users without `ultraplot` will hit `AttributeError` on every plot call. | Make the Matplotlib fallback a real implementation path (e.g., branch on `plt.__name__` or use a compatibility wrapper) or drop the fallback and make `ultraplot` a hard dependency. | Medium |
| 6 | `nn.py` hard constraints | **High** | `apply_hard_constraints` only considers `x` and `y` (`coords = {0: inputs_dict.get("x"), 1: inputs_dict.get("y")}`). It ignores time (`t`) and any other inputs, so hard constraints cannot be applied correctly to transient boundaries. | Transient problems with hard BCs will silently enforce a spatial-only constraint, producing wrong solutions. | Extend `coords` to include all input keys present in `inputs_dict` and update `HardConstraint.define_zero_func` to accept the full input dictionary. | Medium |
| 7 | `domain.py` batched loss | **High** | `_batched_loss` assumes every geometry in the same `physics_type` group has identical `inputs_tensor_dict` keys. If one BC has only `x`/`y` and another has `x`/`y`/`t`, the concatenation fails with a `KeyError`. | Mixed-dimension problems crash during training. | Validate key consistency at domain construction or during grouping; fall back to per-geometry forward passes when keys differ. | Medium |
| 8 | `utility.py` global state | **High** | `_GLOBAL_SEED` and `_RNG` are module-level mutable globals. `device` is also a global string initialized at import time. The helpers are not thread-safe, and `device` becomes stale if CUDA becomes available after import. | Reproducibility breaks in multi-threaded or notebook reload scenarios; GPU selection may be inconsistent. | Encapsulate RNG state in a class or context manager; use `torch.device` dynamically and re-evaluate CUDA availability at runtime. | Medium |
| 9 | `__init__.py` wildcard imports | **Medium** | The public API is defined via `from .module import *`. Adding new names to any module silently changes the package's exported API and can cause name collisions. | API stability is hard to maintain; unintended names leak into the public surface. | Replace wildcard imports with explicit `__all__` lists in each module or explicit imports in `__init__.py`. | Small |
| 10 | `physicsinformed.py` type confusion | **Medium** | `physics_type` is annotated as `Optional[list[str]]` but assigned strings (`"BC"`, `"IC"`, `"PDE"`). `range_t` can be `tuple`, `int`, `float`, or `None`, leading to many `isinstance` branches. | Type hints are misleading; the code is harder to reason about and static analysis fails. | Introduce an enum or literal union for `PhysicsType`; narrow `range_t` to `Optional[Tuple[float, float]]` and handle scalar time separately. | Small |
| 11 | `physicsinformed.py` derivative conditions | **Medium** | Derivative condition keys (e.g., `u_x`) are split on the first underscore: `var_name, grad_var = key.split('_')`. This breaks for higher-order or mixed derivatives like `u_x_y` or `u_xx`. | Multi-derivative boundary conditions cannot be expressed. | Parse derivative keys with a dedicated grammar or tuple-based representation (e.g., `('u', 'x', 'y')`) instead of string splitting. | Small |
| 12 | `physicsinformed.py` exponential scaling | **Medium** | `self.t = (1 + self.t)**(self.t/T1) - 1` is unusual and can overflow for large `t`. The physical motivation is not documented. | Time scaling may produce `inf`/`nan` values for large time ranges. | Document the formula or replace it with a bounded, well-understood time warping function (e.g., sigmoid or linear mapping). | Small |
| 13 | `nn.py` serialization | **Medium** | Model checkpointing uses `pickle`. Pickle is unsafe for untrusted files and breaks across Python versions. | Users cannot safely share models, and loading may fail after upgrades. | Add `save_state_dict` / `load_state_dict` wrappers using `torch.save`/`torch.load` and keep pickle as a deprecated fallback. | Small |
| 14 | `pde.py` residual naming | **Medium** | `PDE.calc_residuals` docstring says "Mean Absolute Error (L1)" but returns `mean(abs(sum(residuals)))`, which is the mean of the absolute *sum* of residuals per point, not the MAE of individual residual components. | Metric naming is misleading; users may misinterpret reported residuals. | Rename the method or change the aggregation to match the documented semantics; document the intended behavior. | Small |
| 15 | `pde.py` Burgers convention | **Medium** | `BurgersEquation1D` uses `y` as the time variable (`u_y + u*u_x`) despite the docstring saying `u_t`. | Users may be confused when setting up a 1D Burgers problem in space-time. | Rename the equation to `BurgersEquationXT` or switch to an explicit `t` input and add a clear migration note. | Small |
| 16 | `pde.py` dead code | **Low** | `NavierStokes.nondimensionalize_inputs` is defined but never called by `compute_residuals` or any other reviewed module. | Carries maintenance burden without providing value. | Remove or wire it into the residual computation if non-dimensionalization is intended. | Tiny |
| 17 | `geometry.py` vertical line hack | **Low** | `line_vertical` uses `LARGE_SLOPE = 1e5` to approximate a vertical boundary. This is numerically unstable and not truly vertical. | Near-vertical boundaries can leak or invert masks. | Represent vertical lines explicitly with a constant-x function and an axis flag. | Small |
| 18 | `geometry.py` area sampling | **Low** | In `Area.sampling_area`, if `n_points_square` is a single `int`, it sets `nx = ny = n_points_square` and `n_total = n_points_square`, so the total number of points is `n_points_square`, not `n_points_square^2`. | The area resolution is much lower than expected when users pass an integer. | Treat a single int as the count per dimension (`n_total = n_points_square^2`) or document the current behavior. | Small |
| 19 | `domain.py` sampling option | **Low** | `self.sampling_option = self.sampling_option + ' + R3'` is repeated in `sampling_RAR`, `sampling_R3`, `sampling_R3_`, and `sampling_accumulate`. If called multiple times, the string becomes malformed (e.g., `"uniform + R3 + R3 + R3"`). | Diagnostic strings are unreliable and may confuse users. | Use a list of applied techniques and join them when formatting. | Tiny |
| 20 | `domain.py` `__getitem__` stub | **Low** | `ProblemDomain.__getitem__` returns `None` unconditionally. | Users may expect dictionary-style access to geometries. | Implement indexing or remove the method to avoid silent failures. | Tiny |
| 21 | `physicsinformed.py` `calc_loss` | **Low** | `calc_loss` accepts `model` but calls `calc_residual_field(model)`, which itself calls `process_model(model)`. This can result in a redundant forward pass when the caller has already cached `model_inputs`/`model_outputs`. | Minor performance overhead and potential graph inconsistency. | Prefer `_compute_residual_field()` when outputs are already cached; document which method to call in each context. | Tiny |
| 22 | `utility.py` unused helper | **Low** | `to_require_grad` is defined but not used in any reviewed file. | Minor dead code. | Remove or use it consistently in `process_coordinates`. | Tiny |
| 23 | `evaluation.py` state mutation | **Low** | `Evaluator.postprocess` mutates `geometry.scheme = "uniform"` and `geometry.expo_scaling = False`, overriding user settings. | Post-processing can silently change sampling behavior for later calls. | Preserve the original scheme or clone the geometry before mutating. | Small |
| 24 | `evaluation.py` loss history mixing | **Low** | `data_dict.update(self.model.loss_history)` mixes 1D scalar arrays with per-point fields. Downstream plotting may misinterpret shapes. | Loss curves may be plotted incorrectly if they share a figure with field data. | Separate training history into its own namespace or accessor. | Small |

---

## 6. Prioritized Recommendations

### P1 — Fix Critical Bugs Before Release

1. **Fix L-BFGS NaN handling** (`nn.py`). Ensure `total_loss_num` is defined before use and that NaN loss terminates the epoch loop cleanly rather than raising `NameError`.
2. **Fix R3 negative point count** (`domain.py`). Guard against negative `n_points` before calling `sampling_line`/`sampling_area`.
3. **Fix RAR threshold masking** (`physicsinformed.py`). Replace the brittle offset mask with an explicit deduplication or index-based tracking scheme.

### P2 — Stabilize Core Geometry and Visualization

4. **Make geometry masking robust** (`geometry.py`). Document the convexity assumption, add validation, and consider replacing the custom ray-casting with a well-tested point-in-polygon algorithm for complex shapes.
5. **Fix the Matplotlib fallback** (`visualization.py`). Either make the fallback a real implementation or drop it and make `ultraplot` a hard dependency.
6. **Extend hard constraints to time-dependent inputs** (`nn.py`). Pass the full `inputs_dict` to the constraint function so transient hard BCs work correctly.

### P3 — Improve API Clarity and Maintainability

7. **Replace wildcard imports** (`__init__.py`). Define explicit `__all__` lists or explicit re-exports to make the public API stable.
8. **Introduce an enum for `physics_type`** (`physicsinformed.py`, `domain.py`, `pde.py`). Use `Literal` or an enum instead of raw strings.
9. **Fix derivative-condition parsing** (`physicsinformed.py`). Support higher-order and mixed derivatives with a structured representation.
10. **Replace pickle checkpointing** (`nn.py`). Provide `state_dict` based save/load as the primary serialization path.

### P4 — Performance and Scalability Hardening

11. **Avoid 10,000-point sampling just for bounds** (`geometry.py`). Compute bounding boxes analytically from function ranges or use a much smaller sample.
12. **Cap adaptive sampling growth** (`domain.py`, `physicsinformed.py`). Add a maximum size or deduplication to prevent unbounded training sets.
13. **Make device selection dynamic** (`utility.py`). Re-evaluate CUDA availability at runtime and return a `torch.device`.

### P5 — Documentation and Cleanup

14. **Document or remove exponential time scaling** (`physicsinformed.py`).
15. **Fix misleading residual method names** (`pde.py`).
16. **Remove or wire dead code** (`pde.py:nondimensionalize_inputs`, `utility.py:to_require_grad`, `domain.py:__getitem__`).

---

## 7. Appendix

### A.1 Files Reviewed

- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\__init__.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\domain.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\evaluation.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\geometry.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\nn.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\pde.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\physicsinformed.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\utility.py`
- `c:\Users\thamm\OneDrive\Documents\1 - Projects\0 - STEM\2 - Numerical Physics\9 - PINNs\pinn-deepflow\src\deepflow\visualization.py`

### A.2 Strengths Summary

- Clear separation between geometry, PDE, network, and training concerns.
- Thoughtful batched forward pass optimization in `_batched_loss`.
- Correct leaf-tensor handling for higher-order autograd derivatives.
- Simple, notebook-friendly API with geometry DSL.

### A.3 Review Methodology

- Static code review of all files listed above.
- Cross-reference of public API usage via `__init__.py` and README quick-start.
- No runtime tests or profiling were performed; performance observations are qualified as unmeasured.
