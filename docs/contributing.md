# Contributing

Thanks for considering a contribution to DeepFlow. This page explains how the
repository is organized and how to develop, test, and submit changes.

## Repository layout

```
src/deepflow/          Package source (src layout)
  geometry.py          Geometries and boolean operations
  domain.py            Domains, boundaries, sampling (incl. R3)
  pde.py               PDE implementations and the PDE base class
  nn.py                FNN/PINN/RFFPINN models and training (Adam/L-BFGS)
  physicsinformed.py   Physics attachment and evaluation glue
  evaluation.py        Evaluators (GroupEvaluator, ReferenceGroupEvaluator)
  visualization.py     Plotting (ultraplot-based)
  reference/           NGSolve FEM backend (optional)
examples/              Runnable notebooks; one folder per example
tests/                 Unit and regression tests (pytest)
benchmarks/            Comparative benchmark scripts
EXPERIMENTS/           Comparative experiments and reports
docs/                  MkDocs site source
static/                Images used by the GitHub README
```

The intended and correct usage of the API is demonstrated in `examples/` —
if you change the API, update the corresponding example and its docs page
(see [REGENERATE.md](https://github.com/YoYo-XYZ/pinn-deepflow/blob/dev/REGENERATE.md)).

## Development setup

The repository uses `uv` (a `uv.lock` is committed). With uv:

```bash
uv sync
```

or with plain pip:

```bash
python -m venv .venv
.\.venv\Scripts\activate        # Windows; see venv docs for other shells
pip install -e .
```

To run the FEM reference comparisons locally, additionally install the
optional extra:

```bash
pip install -e ".[cfd]"
```

## Running tests

```bash
pytest tests/
```

The test suite covers geometry, sampling, PDE residuals, training, the FEM
backend, and the documentation example workflows. Run it before pushing.

## Building the docs locally

```bash
pip install -r requirements-docs.txt
mkdocs serve
```

`mkdocs build --strict` is enforced by CI (`docs-check.yml`) on every pull
request that touches `docs/` — broken links and missing images fail the
build, so build locally before opening the PR.

## Pull request workflow

1. Open an issue first to discuss major changes — the maintainers prefer to
   agree on direction before large diffs.
2. Create a feature branch from `dev`; the docs deploy workflow publishes
   from `dev`.
3. Keep changes **surgical**: touch only what your change requires, and
   follow the style rules in `AGENTS.md` (simplicity first; no speculative
   features; match existing style).
4. Add or update tests for new behavior, and update examples/docs that show
   the affected API.
5. Run `pytest tests/` and `mkdocs build --strict`, then open the pull
   request against `dev`.

## Code style

- Minimum code that solves the problem — no speculative features or
  single-use abstractions.
- Match the existing style of the file you are editing.
- New public API must come with docstrings (Google style) — the API
  reference is generated from them via mkdocstrings.
- Do not commit secrets, generated artifacts, or stray notebooks outputs.

## Release process

Releases are tagged on `dev` (e.g. `v0.1.3`) and the changelog in
[`docs/changelog.md`](changelog.md) is updated by hand per release.
