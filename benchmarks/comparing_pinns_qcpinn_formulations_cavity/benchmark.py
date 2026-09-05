"""Cavity-specific adapter for the shared flow benchmark harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
for path in (PROJECT_ROOT, PROJECT_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import deepflow as df  # noqa: E402
from benchmarks.shared_harness import (  # noqa: E402
    BenchmarkConfig,
    FLOW_FORMULATIONS,
    FLOW_VARIANTS,
    FlowBenchmarkHarness,
    build_cavity_domain,
    build_flow_pde,
)


CAVITY_X = (0.0, 1.0)
CAVITY_Y = (0.0, 1.0)
U_INF = 1.0
L_CHAR = 1.0
MU = 0.1
RHO = 1.0
REYNOLDS = RHO * U_INF * L_CHAR / MU
PROFILE_POINTS = 50
PROFILE_EPSILON = 1.0e-6
FEM_MESH_SIZE = 0.05
FEM_BOUNDARY_RESOLUTION = 128
FEM_TOLERANCE = 1.0e-5
FEM_MAX_ITERATIONS = 200
RESULTS_DIR = SCRIPT_DIR / "results"
REPORT_NAME = "REPORT.md"

DEFAULT_CONFIG = BenchmarkConfig(
    width=48,
    depth=4,
    learning_rate=0.004,
    epochs_adam=0,
    epochs_lbfgs=100,
    seed=69,
    boundary_points=[50, 50, 50, 50, 1],
    interior_points=[[50, 50]],
    eval_grid=[50, 50],
    sampling="uniform",
)

SMOKE_CONFIG = BenchmarkConfig(
    width=8,
    depth=2,
    learning_rate=0.004,
    epochs_adam=2,
    epochs_lbfgs=0,
    seed=69,
    boundary_points=[4, 4, 4, 4, 1],
    interior_points=[[4, 4]],
    eval_grid=[5, 5],
    sampling="uniform",
)

VARIANTS = FLOW_VARIANTS
FORMULATIONS = FLOW_FORMULATIONS


def build_pde(formulation: str):
    """Build the shared cavity flow PDE for one formulation."""
    return build_flow_pde(
        formulation,
        U=U_INF,
        L=L_CHAR,
        mu=MU,
        rho=RHO,
    )


def build_domain(
    formulation: str,
    config: BenchmarkConfig = DEFAULT_CONFIG,
):
    """Build the sampled cavity domain through the shared domain builder."""
    df.manual_seed(config.seed)
    return build_cavity_domain(
        formulation=formulation,
        boundary_points=list(config.boundary_points),
        interior_points=list(config.interior_points),
        sampling=config.sampling,
        u_inf=U_INF,
        L=L_CHAR,
        mu=MU,
        rho=RHO,
    )


def _profile_geometries():
    """Return the cavity-specific vertical and horizontal centerlines."""
    x_min, x_max = CAVITY_X
    y_min, y_max = CAVITY_Y
    epsilon = PROFILE_EPSILON
    return {
        "vertical": df.geometry.line_vertical(
            x=(x_min + x_max) / 2.0,
            range_y=[y_min + epsilon, y_max - epsilon],
        ),
        "horizontal": df.geometry.line_horizontal(
            y=(y_min + y_max) / 2.0,
            range_x=[x_min + epsilon, x_max - epsilon],
        ),
    }


def _report_metadata():
    return {
        "reynolds": REYNOLDS,
        "cavity_x": CAVITY_X,
        "cavity_y": CAVITY_Y,
    }


HARNESS = FlowBenchmarkHarness(
    problem="cavity",
    results_dir=RESULTS_DIR,
    report_name=REPORT_NAME,
    default_config=DEFAULT_CONFIG,
    smoke_config=SMOKE_CONFIG,
    domain_builder=build_domain,
    pde_builder=build_pde,
    profile_geometries=_profile_geometries,
    profile_fields={"vertical": "u", "horizontal": "v"},
    report_metadata=_report_metadata,
)


def available_variants():
    """Return the cavity variants supported by the installed backends."""
    return HARNESS.available_variants()


def build_pinn_model(formulation: str, config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build a standard PINN for the requested cavity formulation."""
    return HARNESS.build_pinn_model(formulation, config)


def build_qcpinn_model(formulation: str, config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build a QCPINN for the requested cavity formulation."""
    return HARNESS.build_qcpinn_model(formulation, config)


def build_model(variant: str, config: BenchmarkConfig = DEFAULT_CONFIG):
    """Build one named cavity model variant."""
    return HARNESS.build_model(variant, config)


def evaluate_profiles(model, formulation: str, points: int = PROFILE_POINTS):
    """Evaluate cavity-specific centerline profiles for a trained model."""
    return HARNESS.evaluate_profiles(model, formulation, points)


def run_once(
    variant: str,
    config: BenchmarkConfig,
    reference_solution=None,
    model_factory=None,
):
    """Train and evaluate one cavity variant through the shared harness."""
    return HARNESS.run_once(
        variant,
        config,
        reference_solution=reference_solution,
        model_factory=model_factory,
    )


def run_variant(
    variant: str,
    config: BenchmarkConfig,
    output_dir: Path,
    reference_solution=None,
    model_factory=None,
):
    """Run, persist, plot, and report one cavity variant."""
    return HARNESS.run_variant(
        variant,
        config,
        output_dir,
        reference_solution=reference_solution,
        model_factory=model_factory,
    )


def run_suite(
    config: BenchmarkConfig = DEFAULT_CONFIG,
    output_dir: Path = RESULTS_DIR,
    variants=None,
    reference_solution=None,
):
    """Run selected cavity variants and write the benchmark report."""
    return HARNESS.run_suite(
        config,
        output_dir,
        variants=variants,
        reference_solution=reference_solution,
    )


def run_variant_cli(variant: str, model_factory=None, argv=None):
    """Run one cavity variant using shared command-line options."""
    return HARNESS.run_variant_cli(variant, model_factory=model_factory, argv=argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=VARIANTS, default="PINN-UVP")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()
    run_suite(
        SMOKE_CONFIG if args.smoke else DEFAULT_CONFIG,
        args.output_dir,
        variants=(args.variant,),
    )
