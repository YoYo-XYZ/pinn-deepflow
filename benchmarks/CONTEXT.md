# Benchmarks

`benchmarks/` holds suites that compare model variants, formulations, sampling schemes, or library implementations on a common domain and PDE. Use the [root glossary](../CONTEXT.md) for terminology. Experiments belong in `EXPERIMENTS/`; canonical-use Examples belong in `examples/`. The `comparing_*` directory names are legacy names for Benchmarks.

## Suite map

| Suite | What varies | Documentation |
| --- | --- | --- |
| `comparing_pinns_qcpinn_formulations_cavity` | PINN/QCPINN model variants and UVP/PSIP formulations on lid-driven cavity flow | [README](comparing_pinns_qcpinn_formulations_cavity/README.md) |
| `comparing_pinns_qcpinn_cylinder` | The same four variants on flow around a cylinder | [README](comparing_pinns_qcpinn_cylinder/README.md) |
| `comparing_precision`, `comparing_precision_channel_flow` | FP32/FP64 on Burgers and channel flow | [Burgers README](comparing_precision/README.md), [channel README](comparing_precision_channel_flow/README.md) |
| `comparing_rff_pinn_burgers` | PINN/RFFPINN model variants | [Suite guide](comparing_rff_pinn_burgers/REPORT.md) |
| `comparing_rffpinn_sampling_burgers` | Fixed LHS, fixed uniform, and LHS followed by R3 sampling for RFFPINN | [Suite guide](comparing_rffpinn_sampling_burgers/REPORT.md) |
| `comparing_init_method` | Kaiming-uniform/Glorot-normal initialization on Burgers and channel flow | Entrypoints: [Burgers](comparing_init_method/burgers_equation/compare_init.py), [channel](comparing_init_method/chennel_flow/compare_init.py). No separate suite guide. Keep the existing `chennel_flow` spelling in commands. |
| `comparing_deepxde` | DeepFlow/DeepXDE channel-flow implementations | [Historical report](comparing_deepxde/results/benchmark_report.md), [runner](comparing_deepxde/run_benchmark.py) |
| `comparing_legacy_deepflow` | Saved Burgers models labeled as old/new library versions | [Historical report](comparing_legacy_deepflow/results/benchmark_report.md), [training entrypoint](comparing_legacy_deepflow/benchmark_burgers.py), [comparison entrypoint](comparing_legacy_deepflow/compare_versions.py) |

Historical reports describe earlier runs, not current output contracts. The RFF suite guides live outside `results/`; new runs write their generated reports inside `results/`.

## Shared workflow

[ADR 0001](../docs/adr/0001-shared-harness-break-and-migrate-allowlist.md) sets the convention: suites declare what varies and delegate common work to [`shared_harness`](shared_harness/__init__.py).

- [`config.py`](shared_harness/config.py) defines `BenchmarkConfig` for network size, optimizer schedule, seeds, boundary/interior budgets, evaluation grid, initial sampler, and R3 interval. Physics parameters stay with domain builders. Suites define their own normal and smoke configurations.
- [`domains.py`](shared_harness/domains.py) constructs Burgers, channel, cavity, and cylinder domains through geometry, boundary-condition, PDE, and sampling APIs. Burgers uses `y` as time. Channel boundary counts default to a shared perimeter-weighting rule; explicit counts override it.
- [`reporting.py`](shared_harness/reporting.py) owns `train_one`: seed model construction, use `df.calc_loss_simple`, train Adam followed by L-BFGS, and return the selected model with timings and final losses. Zero epochs skip a phase. R3 runs only during Adam when enabled.
- The same module evaluates the first domain area on a fresh grid and supports line evaluation. Metrics come from evaluator `data_dict`, loss-history tails, and Reference solution queries at matching coordinates. Persistence uses `save_as_pickle`/`load_from_pickle`; plots use evaluator visualizers; reports list configuration, metrics, and artifacts.
- [`flow.py`](shared_harness/flow.py) orchestrates cavity/cylinder variants, profiles, repeated seeds, and saved-model comparisons. [`precision.py`](shared_harness/precision.py) copies one FP32 model and sampled/evaluation coordinates per seed, casts them for each precision, and synchronizes CUDA for overall training timing.
- Reference solves remain suite-owned calls to `domain.solve_fem`. [`reference.py`](shared_harness/reference.py) supplies explicit offline cache loading and FEM-value export. Cavity/cylinder comparisons solve a fresh FEM reference by default; `--no-reference` skips it and `--offline-reference PATH` explicitly selects a numeric archive. Running their `reference.py` alone does not create a cache unless `--export-cache PATH` is supplied.

## Run from the repository root

Use Python 3.10 or newer. Install required dependencies with:

```powershell
python -m pip install -e .
```

[pyproject.toml](../pyproject.toml) lists PyTorch, NumPy, Matplotlib, SymPy, UltraPlot, SciPy, and cloudpickle. Optional dependencies:

- `python -m pip install -e ".[cfd]"` installs NGSolve for FEM Reference solutions, which also use Netgen.
- `python -m pip install pennylane` enables QCPINN variants.
- `python -m pip install deepxde` enables the external competitor; its entrypoint selects the PyTorch backend.
- `python -m pip install -e ".[test]"` installs pytest for repository tests.

Representative normal runs, which may train for substantial time:

```powershell
python benchmarks/comparing_rff_pinn_burgers/benchmark.py --no-reference
python benchmarks/comparing_precision_channel_flow/benchmark_precision.py --num_runs 3
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/run_benchmark.py --all
python benchmarks/comparing_pinns_qcpinn_cylinder/run_benchmark.py --all
python benchmarks/comparing_deepxde/run_benchmark.py --all
```

The cavity/cylinder `--all` runs require FEM and PennyLane; the DeepXDE run requires DeepXDE. RFF normal runs attempt FEM unless `--no-reference` is supplied, and skip missing-backend errors. Precision runs do not solve a reference.

For changes, use the small assertion-based smoke scripts. These write to temporary directories and remove their outputs:

```powershell
python benchmarks/comparing_rff_pinn_burgers/smoke_test.py
python benchmarks/comparing_rffpinn_sampling_burgers/smoke_test.py
python benchmarks/comparing_precision/smoke_test.py
python benchmarks/comparing_precision_channel_flow/smoke_test.py
python benchmarks/comparing_init_method/burgers_equation/smoke_test.py
python benchmarks/comparing_init_method/chennel_flow/smoke_test.py
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/smoke_test.py --all-setups
python benchmarks/comparing_pinns_qcpinn_cylinder/smoke_test.py --all-setups
python benchmarks/comparing_deepxde/smoke_test.py
python benchmarks/comparing_legacy_deepflow/smoke_test.py
```

Cavity/cylinder `--all-setups` includes QCPINN when available. Use `--pinn-reference` instead to exercise FEM and reference metrics; it skips unavailable-backend errors. Smoke success verifies execution and artifacts, not converged accuracy or competitive performance.

To retain small-run outputs, use a training entrypoint with `--smoke`, for example:

```powershell
python benchmarks/comparing_rff_pinn_burgers/benchmark.py --smoke --output-dir benchmarks/comparing_rff_pinn_burgers/results/smoke
python benchmarks/comparing_pinns_qcpinn_formulations_cavity/run_benchmark.py --all --smoke
```

Flags differ by entrypoint. The cylinder orchestrator and DeepXDE orchestrator have no `--smoke`; use their smoke scripts or individual DeepFlow training entrypoints. Small configurations are CPU-friendly, but DeepFlow selects CUDA when available rather than forcing CPU.

## Results and artifacts

Normal outputs default to each suite's `results/`, including each initialization subfolder. Saved-model comparisons for cavity, cylinder, DeepXDE, and legacy versions default to `results/comparison/`. Many individual entrypoints accept `--output-dir`; orchestrators use fixed suite paths. Repeated runs can overwrite the same filenames.

| Artifact | Meaning |
| --- | --- |
| `*.pkl` | Native DeepFlow model, reloadable through the shared helper for evaluation and plotting. Repeated-seed flow, precision, and legacy runs save a representative model, not every run. |
| `*_field.png`, `*_loss_curve.png` | Shared visualizer output. The helper plots the first available solution field and a loss curve when history exists, not every field/profile. |
| `REPORT.md` | Current configuration, variant-prefixed metrics, timings, and artifact filenames. `final_*_loss` is recomputed on the selected model; `history_last_*` is the history tail. Residual metrics measure PDE consistency; reference-error metrics require a Reference solution. |
| Numeric `*.npz` archives | Most checked-in archives are legacy outputs. Current readers use native DeepFlow models, except the DeepXDE competitor archive and explicitly selected offline reference caches. |

Repeated-seed reports include `*_mean` and sample `*_std`, with zero standard deviation for one run. Flow and legacy choose the median-ranked final-loss run; precision chooses one paired seed by the median-ranked mean loss across precisions. RFF and initialization suites currently run only `config.seed`, despite the shared config supporting a seed list.

## Add or modify a benchmark

1. Declare the domain/PDE, formulations or model variants, and the controlled difference. Reuse domain builders and `BenchmarkConfig`; extend shared code only when common behavior needs to change. Use `FlowBenchmarkHarness` or `run_precision_suite` where applicable, and the reporting helpers elsewhere.
2. Keep comparisons fair. Hold physics, budgets, evaluation coordinates, optimizer schedule, and paired seeds constant except for the variable under test. Seed before stochastic domain construction as well as model construction; `train_one` alone seeds too late for domain sampling. Precision comparisons should reuse the copied baseline. Report model-size differences and reference mode.
3. Use canonical DeepFlow calls. The ADR permits reasoned `FLEX` markers for the external competitor, quantum-backend construction, precision casts/synchronization, and demonstrated visualizer gaps. This is the intended allowlist; current code also marks parameter-count metadata and offline-cache interpolation. Existing markers do not grant a general exception for new raw code.
4. Provide a small smoke configuration and temporary-output smoke script. Verify selected variants, evaluated fields, meaningful metrics, persistence, plots, and report creation. Exercise the changed optimizer, sampling, comparison-reader, or reference path; an Adam-only smoke cannot verify L-BFGS. Run the affected smoke and relevant repository tests before relying on a full run.
5. Link suite documentation and update readers together with output changes. The ADR deliberately breaks compatibility with old outputs; do not silently fall back to checked-in archives.

## Current limits

- UVP and PSIP have three and two PDE residual equations respectively. Flow training reports include `pde_loss_per_equation`; raw summed losses alone are not equivalent formulation comparisons. Pressure errors in saved-model flow comparisons are mean-centered to remove the pressure gauge.
- DeepXDE remains an external raw implementation with its own sampling and loss reduction. Shared budgets do not guarantee identical points or losses. Its smoke uses a synthetic competitor archive and does not train or validate DeepXDE.
- Legacy `benchmark_burgers.py --version` labels an output; it always imports this checkout's `src/deepflow`. It does not check out or select an old revision. `compare_versions.py` needs two compatible native model files. Its smoke compares the same newly trained file under both labels, testing the reader rather than a version difference.
- Shared area evaluation uses `domain.area_list[0]`. A new multi-area benchmark needs an explicit evaluation plan. Default shared plots also omit suite-specific profile and full-field comparison panels.
