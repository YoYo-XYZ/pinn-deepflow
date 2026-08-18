# Installation

## Install from PyPI

```bash
pip install deepflow
```

## Optional: FEM reference backend (`deepflow[cfd]`)

The `[cfd]` extra installs NGSolve/Netgen, the finite-element backend that
solves DeepFlow's built-in PDEs on unstructured meshes. You only need it if
you want to compare PINN results against FEM reference solutions. It is
optional and loaded lazily — a plain `import deepflow` never touches it:

```bash
pip install "deepflow[cfd]"
```

The extra requires Python 3.10+.

## Experimental: quantum models

QPINN and QCPINN are experimental and are not part of the stable 0.1.3
support commitment. If you want to try them, install PennyLane separately:

```bash
pip install pennylane
```

## Install from source

```bash
git clone https://github.com/YoYo-XYZ/pinn-deepflow.git
cd pinn-deepflow
pip install -e .
```

An editable install keeps the package in sync with the `src/` checkout — the
right choice for development (see [Contributing](contributing.md)).

## GPU / CUDA

DeepFlow uses PyTorch as its backend and automatically selects CUDA when a
compatible GPU is available (`df.device` reports `'cuda'`). To use it, install
a PyTorch build with CUDA support first:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Then verify device selection in Python:

```python
import deepflow as df
print(df.device)  # 'cuda' or 'cpu'
```

## Requirements

- Python >= 3.10
- PyTorch >= 1.7.0
- NumPy >= 1.19.0
- Matplotlib >= 3.3.0
- SymPy >= 1.5.0
- SciPy >= 1.5.0
- ultraplot >= 1.0.0

## Troubleshooting

### Device selection

DeepFlow selects `'cuda'` when `torch.cuda.is_available()` is true, `'cpu'`
otherwise. To force the CPU:

```python
import deepflow as df
df.device = 'cpu'
```

### `expo_scaling has not yet defined` warning

`define_time` without an explicit `expo_scaling` argument warns and defaults
to `False`:

```python
g.define_time(range_t=[0, 1], sampling_scheme='random', expo_scaling=False)
```

### Import errors after upgrading

If `import deepflow` fails after an upgrade, the usual causes are a stale
editable install or a broken PyTorch build. Reinstall cleanly:

```bash
pip uninstall deepflow
pip install deepflow
```

### Missing plots or visualization failures

The plotting layer is built on ultraplot. If plots fail with
`ImportError: UltraPlot`, ensure `ultraplot >= 1.0.0` is installed and not
shadowed by a package named `UltraPlot` (the PyPI name is lowercase).
