# QCPINN vs PINN — 2D Steady Cylinder Flow Benchmark

Compares a **hybrid quantum-classical PINN (QCPINN)** against a **parameter-matched classical PINN** on the 2D steady incompressible flow-around-cylinder benchmark at **Re = 50** (the same problem as `examples/cylinder_flow_steady/`).

---

## Motivation

QCPINN replaces the central hidden layers of a feedforward network with a **parameterized quantum circuit** (simulated via [PennyLane](https://pennylane.ai/)). The classical pre-layers compress inputs to `nqubits` dimensions, the quantum circuit processes them through AngleEmbedding + cascade ansatz + PauliZ measurement, and classical post-layers expand back to the output dimension. Everything else (training, loss, physics, geometry) is inherited from the classical PINN.

This benchmark answers: **at matched parameter count (~770–800 trainable parameters), does the QCPINN match the accuracy and convergence behavior of a pure classical PINN on a realistic 2D Navier-Stokes problem?**

---

## How to run

From the repository root:

```bash
# Full benchmark: 3 independent runs of each model, then comparison
python benchmarks/comparing_pinns_qcpinn_cylinder/run_benchmark.py --all

# Single-run smoke test (fast)
python benchmarks/comparing_pinns_qcpinn_cylinder/run_benchmark.py --all --num_runs 1

# Custom epochs (quick smoke test of training loop)
python benchmarks/comparing_pinns_qcpinn_cylinder/benchmark_pinn.py --num_runs 1 --epochs_adam 100 --epochs_lbfgs 50
python benchmarks/comparing_pinns_qcpinn_cylinder/benchmark_qcpinn.py --num_runs 1 --epochs_adam 100 --epochs_lbfgs 50

# Comparison only (re-generate plots and report from existing NPZs)
python benchmarks/comparing_pinns_qcpinn_cylinder/run_benchmark.py --compare
```

**Prerequisites**: `deepflow` (this repo, `pip install -e .` from repo root) and `pennylane` (`pip install pennylane`) for the QCPINN side. The classical PINN runs without PennyLane.

---

## What it measures

1. **Parameter count** — total trainable parameters per model (for the parameter-matched design).
2. **Training** — Adam (lr=0.004, 2000 epochs, threshold=0.01) → L-BFGS (500 epochs, threshold=0.0001).
3. **Resampling** — `"randomr"`: full LHS resampling of all collocation + boundary points every 100 L-BFGS epochs (non-adaptive periodic baseline).
4. **Metrics per run** — final total / BC / PDE loss, max & mean absolute continuity / x-momentum / y-momentum residuals, full loss curves, wall-clock time (Adam / L-BFGS / total).
5. **Aggregation** — mean ± std (ddof=1) across runs; **median-loss run** used for representative field plots and loss-curve plots.
6. **Visual comparison** — side-by-side u, v, p fields, continuity residual field, outlet velocity profile, loss curves.

---

## Outputs

All outputs land in `results/`:

| File | Contents |
|------|----------|
| `pinn_results.npz`   | Aggregated PINN results — mean/std metrics + median-run fields/losses/outlet profile |
| `qcpinn_results.npz` | Same structure for QCPINN |
| `compare_loss_curves.png`       | Total / BC / PDE loss curves (semilogy), both models overlaid |
| `compare_u_field.png`           | u velocity field, side-by-side |
| `compare_v_field.png`           | v velocity field, side-by-side |
| `compare_p_field.png`           | Pressure field, side-by-side |
| `compare_continuity_residual.png` | |continuity residual| field, side-by-side |
| `compare_outlet_velocity.png`   | Outlet u(y) profile at x=1.1, overlaid |
| `benchmark_report.md`           | Markdown summary table + figure list |

---

## Architecture details

### Parameter matching

| Model | Layers | Parameters |
|-------|--------|------------|
| **PINN**   | `PINN(width=18, length=3)` — 3 hidden layers × 18 neurons, Tanh | 795 |
| **QCPINN** | `QCPINN(pre=[50], post=[50], nqubits=4, q_layer_iterations=1)` | 769 |

QCPINN breakdown:
- **Pre**: `Linear(2→50) = 150` + `Linear(50→4) = 204` → **354**
- **Quantum**: weights `(1, 3, 4) = 12` (1 cascade layer × 3 rotation rows × 4 qubits)
- **Post**: `Linear(4→50) = 250` + `Linear(50→3) = 153` → **403**
- **Total**: **769** (within 3.4% of the classical PINN)

Quantum circuit (`src/deepflow/qnn.py::_qcpinn_circuit`):
1. `AngleEmbedding(inputs, rotation="Y")` on `nqubits=4` qubits
2. Per cascade layer: `RX(θ)` + `RY(φ)` on each qubit, then ring of `CRX(ψ)` entangling gates
3. Measure `⟨PauliZ⟩` on each qubit → `nqubits` classical outputs

### Problem setup

- **Geometry**: channel `[0, 1.1] × [0, 0.41]` with circular cylinder at `(0.2, 0.2)`, radius `0.05`
- **PDE**: 2D steady incompressible Navier-Stokes, Re = 50
- **BCs**: parabolic inlet `u(y) = 4·y·(0.41−y)/0.41²`, `v=0`; no-slip walls + cylinder; outlet `p=0`
- **Sampling**: LHS initial — 1000 points per boundary × 6 boundaries = 6000; 4000 interior
- **Loss**: `df.calc_loss_simple(domain)` (unweighted BC + PDE sum)

---

## Expected runtime

**The full configuration (2000 Adam + 500 L-BFGS epochs) is fast for PINN but extremely slow for QCPINN on CPU.** PennyLane's `default.qubit` is a pure-Python classical simulator — every forward pass through the quantum circuit is much more expensive than a `Linear` layer of comparable width. Empirically observed rates (CPU, FP32, 6000 boundary + 4000 interior points):

| Model | Adam (per epoch) | L-BFGS (per epoch) | Full run (3×) |
|-------|------------------|--------------------|--------------|
| **PINN**   | ~0.03 s | ~0.9 s | ~20 min |
| **QCPINN** | ~30–60 s | ~5–10 min | many hours |

**Practical recommendations:**

1. **First run** — validate the pipeline with a smoke test (PINN full, QCPINN reduced):
   ```bash
   python benchmark_pinn.py --num_runs 1            # ~5 min
   python benchmark_qcpinn.py --num_runs 1 --epochs_adam 50 --epochs_lbfgs 10  # ~1–2 hours
   python compare.py
   ```
2. **For faster QCPINN** — use a C++-accelerated PennyLane backend (e.g. `lightning.qubit`):
   ```python
   # In src/deepflow/qnn.py, change:
   qml_device = qml.device("default.qubit", wires=self.nqubits)
   # to:
   qml_device = qml.device("lightning.qubit", wires=self.nqubits)
   ```
   Expect 5–20× speedup depending on hardware.
3. **For meaningful convergence** — the full configuration (2000 Adam + 500 L-BFGS) is needed; the QCPINN converges much more slowly than the PINN at matched parameter count (empirically ~3–5× more epochs needed for similar loss). Plan for overnight runs.
4. **For the `--all` orchestrator** — use reduced QCPINN epochs if you want a same-day result:
   ```bash
   python benchmark_pinn.py --num_runs 3
   python benchmark_qcpinn.py --num_runs 1 --epochs_adam 200 --epochs_lbfgs 20
   python compare.py
   ```

---

## Project layout

```
benchmarks/comparing_pinns_qcpinn_cylinder/
├── common_config.py        # shared geometry, PDE, sampling, training, architecture
├── benchmark_pinn.py       # classical PINN training (3 runs by default)
├── benchmark_qcpinn.py     # QCPINN training (3 runs by default)
├── compare.py              # load NPZs → console table → plots → Markdown report
├── run_benchmark.py        # orchestrator (--pinn, --qcpinn, --compare, --all)
├── smoke_test.py           # minimal QCPINN pipeline test (20 Adam epochs, tiny sampling)
├── README.md               # this file
└── results/                # all outputs (NPZs, PNGs, report.md, run logs)
```

`smoke_test.py` is a fast (~2 min) standalone script that builds the full domain, creates a QCPINN, runs 20 Adam epochs, and evaluates — useful for verifying PennyLane integration without committing to a full run.
