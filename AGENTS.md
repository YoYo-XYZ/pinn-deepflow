# DeepFlow

DeepFlow is a Python library for solving PDEs with PyTorch physics-informed neural networks, focused on fluid dynamics. Its notebook-oriented workflow follows a CFD solver: define geometry and physics, sample points, train, then evaluate and plot.

## Before contributing

- Read the relevant code in `src/deepflow/` and a matching notebook in `examples/`. The notebooks show how users combine the public `import deepflow as df` API.
- Core PINN functionality must work without the optional FEM and quantum backends. NGSolve/Netgen comes through the `cfd` extra; experimental quantum models require a separate PennyLane install.
- Treat numerical behavior as part of compatibility. Changes to equations, loss definitions, sampling, precision, or training defaults can change scientific results even when the API stays the same.

## Making changes

- Keep changes focused on the requested behavior. Extend the existing geometry-to-evaluation workflow before introducing another abstraction or dependency.
- Preserve notebook usage and saved-model compatibility unless the task requires a breaking change. Explain any change to numerical results or persistence formats.
- Preserve CPU/GPU support, FP32/FP64 support, and seeded reproducibility. Evaluation can compute PDE derivatives too, so it still needs autograd.
- Keep restricted model loading separate from trusted Python pickle loading. Do not weaken artifact validation to accommodate unsupported models.
- Use Google-style public docstrings for the generated API reference. When example usage changes, follow `REGENERATE.md` to keep notebooks, hand-authored pages, and figures aligned.

## Development and verification

Install from the repository root:

```sh
python -m pip install -e ".[test]"
```

`pyproject.toml` defines dependencies and extras. Use `.github/workflows/` for CI commands and supported environments.

- Run focused tests for the changed behavior, then `python -m pytest tests/ -q -m "not slow"`. Sampling or training changes also need the reproducibility regression. Verify numerical changes in both precisions with a small, seeded problem before attempting full example training.
- For FEM work, install `.[test,cfd]` and run the CFD checks listed in CI.
- For docs or public API changes, install `requirements-docs.txt` and run `mkdocs build --strict`. For packaging changes, build and validate distributions as CI does.
- Report what you verified and any checks you could not run. This checkout currently lacks test sources and Markdown documentation pages referenced by CI; an empty test run is not a pass.
