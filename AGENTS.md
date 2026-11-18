# DeepFlow

DeepFlow is a user-freindly Python library for solving PDEs with PyTorch physics-informed neural networks, focused on fluid dynamics. Its notebook-oriented workflow follows a CFD solver: define geometry and physics, sample points, train, then evaluate and plot.

The library is still immature, so critically evaluate the codebase. Be skeptical of existing implementations and do not assume that the current code follows best practices.

## Coding Principles

1. Minimal and Intuitive Public API. Expose only what users strictly need, keeping internal helper functions and state private. A small surface area makes the library easier to learn and prevents accidental misuse.

2. Keep the implementations simple and avoiding speculative features. Adding unnecessary flexibility early on creates permanent maintenance overhead once users depend on it.

3. Simplicity First. Minimum code that solves the problem. No speculative features, no single-use abstractions, no unrequested config. If 200 lines could be 50, rewrite to prioritize maintainability.

4. Surgical Changes
Touch only what you must. Don't improve adjacent code or refactor unrelated things. Match existing style. Remove only the dead code your changes create.

## Codebase Practice
- `src/deepflow` is the source code, where you should be focused on.
- `examples/` are written and maintained by the author, so treat them as the source of truth for how to use Deepflow.
- `benchmark/` is for conducting the benchmark of Deepflow such as performance evaluation, not for testing code changes.
- `experiments/` is for author's personal interest on applying Deepflow to conduct his own research
- `tests/` is for conducting tests
- Test public APIs, not internals. Verify behavior through the functions and classes users actually call.
- Do not write .md files unless explicitly asked
- Do not add `#` comments in source code or internal docstrings; keep user-facing API documentation concise.