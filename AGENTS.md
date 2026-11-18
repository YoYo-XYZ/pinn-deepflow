# DeepFlow

DeepFlow is a user-freindly Python library for solving PDEs with PyTorch physics-informed neural networks, focused on fluid dynamics. Its notebook-oriented workflow follows a CFD solver: define geometry and physics, sample points, train, then evaluate and plot.

## Coding Principles

1. Simplicity First
Minimum code that solves the problem. No speculative features, no single-use abstractions, no unrequested config. If 200 lines could be 50, rewrite to prioritize maintainability.

2. Surgical Changes
Touch only what you must. Don't improve adjacent code or refactor unrelated things. Match existing style. Remove only the dead code your changes create.

## Codebase Practice

1. Since the examples in `examples/` are written and maintained by the author, treat them as the source of truth for how to use Deepflow.

2. Do not write .md files unless explicitly asked

3. Do not add `#` comments in source code or internal docstrings; keep user-facing API documentation concise.