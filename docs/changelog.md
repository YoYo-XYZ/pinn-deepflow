# Changelog

All notable changes to DeepFlow are recorded here, grouped by release. The
format follows [Keep a Changelog](https://keepachangelog.com/); releases are
tagged on the `dev` branch.

## [0.1.3] - in development

### Added

- `ReferenceGroupEvaluator` — `solve_fem` now returns a directly queryable
  reference solution (previously a plain solver result); per-geometry
  evaluators re-query the FEM fields.
- `GroupEvaluator` access API and `domain.evaluate()` — evaluators gained
  `[]`-style field access (`area_eval['u']`, `fem_area['u_ref']`) alongside
  `data_dict`; lazy evaluator field expressions.
- `RFFPINN` — random Fourier feature network variant, with tests.
- `StreamFunctionNavierStokes` PDE (stream-function formulation), with tests.
- `max_grad_norm` gradient clipping in training methods.
- Configurable `weight_init` and Xavier initialization for FNN/PINN.
- Robust boolean geometry operations (`Area.__add__`, containment, robust
  union) with tests.
- Unit tests for PINN training with Adam and L-BFGS optimizers.
- End-to-end reproducibility test — seeding now covers all RNGs.

### Changed

- Evaluation API refined (`Refine GroupEvaluator access API`): cleaner
  field access and consistent contracts across steady and transient cases.
- FEM backend expanded: transient solves with time stepping, reference
  solutions handling float32 inputs, default mesh size set.
- Training loop refactored: batched forward passes per physics type,
  reduced peak memory in L-BFGS, consistent training status output.
- R3 resampling matches the original paper's algorithm; repeated random/LHS
  sampling advances a shared RNG.
- Example notebooks aligned to the current API and standardized on R3
  adaptive resampling; example docs pages cleaned and hand-authored (see
  [REGENERATE.md](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/REGENERATE.md)).
- Documentation: complete API reference and contract fixes; quickstart
  training examples fixed; docs site restructured (Examples gallery,
  Install, FAQ, Contributing, Changelog, Cite pages added).

### Fixed

- `Area.__add__` crash and `train_adam`/`train_lbfgs` return-consistency.
- Hard-constraint loss and forward-mutation bugs in geometry evaluation.
- `Area` union and subtraction edge cases.
- Non-finite loss now stops training cleanly.
- Windows case-sensitivity collisions removed.
- Missing `notebook` title/description swaps in example docs.

### Removed

- Obsolete benchmark scripts and experiment artifacts consolidated under
  `EXPERIMENTS/` and `benchmarks/`.
- Duplicate doc assets (`docs/img/` tree) — one canonical `docs/static/`
  tree plus the README's root `static/`.

## [0.1.2] - 2026-02-03

Initial tagged release.

### Added

- First complete set of example demos (Burgers, cavity, cylinder, heat
  equation, transient channel flow) and their documentation pages.
- Animation support for transient solutions (`plot_animate`).
- Heat equation example and `HeatEquation` PDE.
- Better-plot visualization layer (ultraplot-based) and promo assets.

### Fixed

- Various demo bugs, R3 sampling edge cases, hard-BC handling.
- Repository metadata, README badges, and requirements.

### Docs

- First MkDocs prototype and GitHub Pages deployment workflow.

## Older history

Untagged commits before `v0.1.2` — see `git log v0.1.2`.
