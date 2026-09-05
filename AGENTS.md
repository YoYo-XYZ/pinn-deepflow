# Repository Guidelines

## Project Structure & Module Organization

DeepFlow is a Python package using a `src` layout. Core APIs and implementations live in `src/deepflow/` (`geometry.py`, `domain.py`, `pde.py`, `nn.py`, `physicsinformed.py`, and related utilities). Regression and unit tests are in `tests/`; runnable scientific examples are grouped under `examples/`, while comparative experiments and benchmark scripts are under `EXPERIMENTS/` and `benchmarks/`. User documentation and the MkDocs site are in `docs/`; images and other site assets are in `static/`.

The intended and correct usage of the DeepFlow API is demonstrated in `examples/`.

## Coding Style & Naming Conventions

### 1. Simplicity First

Minimum code that solves the problem. No speculative features, no single-use abstractions, no unrequested config. If 200 lines could be 50, rewrite to prioritize maintainability.

### 2. Surgical Changes

Touch only what you must. Don't improve adjacent code or refactor unrelated things. Match existing style. Remove only the dead code your changes create.

## Agent skills

### Issue tracker

Issues live in GitHub Issues (YoYo-XYZ/pinn-deepflow). See `docs/agents/issue-tracker.md`.

### Triage labels

Using default five canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout (root `CONTEXT.md` + `docs/adr/`). See `docs/agents/domain.md`.
