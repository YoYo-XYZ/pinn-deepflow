# DeepFlow

DeepFlow is a physics-informed neural network framework with a geometry-first workflow. This context fixes the vocabulary for the benchmark rework so future suites land in the right place.

## Language

### Suite kinds

**Benchmark**:
A suite under `benchmarks/` that runs two or more variants on one domain and PDE through the shared harness with a smoke-mode check.
_Avoid_: Comparison (legacy `comparing_*` folder prefix; the suite itself is a Benchmark)

**Experiment**:
Exploratory work under `EXPERIMENTS/`, usually a notebook, that tries an idea with no shared-harness or smoke-path obligation.
_Avoid_: Benchmark, example

**Example**:
A runnable notebook under `examples/` with a hand-authored docs page demonstrating canonical DeepFlow use for one PDE.
_Avoid_: Tutorial, demo

### Ground truth and variants

**Reference solution**:
The FEM solve via `domain.solve_fem` plus reference evaluation; cached numeric archives are only an explicit offline fallback when the FEM backend is unavailable.
_Avoid_: Ground truth, cached data

**Formulation**:
The PDE variant under test (for example `NavierStokes` uvp versus `StreamFunctionNavierStokes` psip).
_Avoid_: Model, architecture

**Model variant**:
The neural-model construction under test (for example PINN versus RFFPINN versus QCPINN, or width, depth, init, and precision settings).
_Avoid_: Formulation

### Rework rules

**Bloat**:
Training, sampling, evaluation, or plotting logic cloned across suites instead of delegating to the shared harness; the deletion target of the rework.
_Avoid_: Clone (a clone is one copy; bloat is the problem)

**Canonical use**:
Public-surface-only DeepFlow calls (geometry, domain, PDE library, neural models, sampling, Adam and L-BFGS trainers with the simple-loss helper, evaluate, evaluator and reference-evaluator, visualizer, FEM solve, pickle-style persistence, manual seeding); anything else needs an allowlisted flex marker with a reason.
_Avoid_: Raw-torch construction, hand-rolled residuals, custom plots
