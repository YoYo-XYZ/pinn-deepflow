#!/usr/bin/env python3
"""Benchmark a direct ``(u, v, p)`` PINN on lid-driven cavity flow."""

from benchmark_common import df, run_benchmark
from common_config import PINN_LENGTH, PINN_WIDTH, UVP_RESULTS_FILE


def build_model():
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        width=PINN_WIDTH,
        length=PINN_LENGTH,
    )


if __name__ == "__main__":
    run_benchmark(
        label="UVP",
        formulation="uvp",
        description="Benchmark: direct (u,v,p) PINN for Re=10 cavity flow.",
        network_description=f"PINN(width={PINN_WIDTH}, length={PINN_LENGTH}, outputs=[u,v,p])",
        pde_residual_count=3,
        model_factory=build_model,
        results_file=UVP_RESULTS_FILE,
    )
