# DeepFlow

DeepFlow is a Python library for solving PDEs with PyTorch physics-informed neural networks, focused on fluid dynamics. Its notebook-oriented workflow follows a CFD solver: define geometry and physics, sample points, train, then evaluate and plot.

## Coding Principles

1. Simplicity First
Minimum code that solves the problem. No speculative features, no single-use abstractions, no unrequested config. If 200 lines could be 50, rewrite to prioritize maintainability.

2. Surgical Changes
Touch only what you must. Don't improve adjacent code or refactor unrelated things. Match existing style. Remove only the dead code your changes create.