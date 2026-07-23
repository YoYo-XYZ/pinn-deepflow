# Repository Guidelines

## Project Structure & Module Organization

DeepFlow is a Python package using a `src` layout. Core APIs and implementations live in `src/deepflow/` (`geometry.py`, `domain.py`, `pde.py`, `nn.py`, `physicsinformed.py`, and related utilities). Regression and unit tests are in `tests/`; runnable scientific examples are grouped under `examples/`, while comparative experiments and benchmark scripts are under `EXPERIMENTS/` and `benchmarks/`. User documentation and the MkDocs site are in `docs/`; images and other site assets are in `static/`.

The intended and correct usage of the DeepFlow API is demonstrated in `examples/`.

## Coding Style & Naming Conventions

### 1. Simplicity First

Minimum code that solves the problem. No speculative features, no single-use abstractions, no unrequested config. If 200 lines could be 50, rewrite.

### 2. Surgical Changes

Touch only what you must. Don't improve adjacent code or refactor unrelated things. Match existing style. Remove only the dead code your changes create.
