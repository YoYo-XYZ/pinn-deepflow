#!/usr/bin/env python3
"""Benchmark the classical PINN on 2D steady lid-driven cavity flow."""

from benchmark_common import df, run_benchmark
from common_config import PINN_LENGTH, PINN_RESULTS_FILE, PINN_WIDTH


def build_model():
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        width=PINN_WIDTH,
        length=PINN_LENGTH,
    )


if __name__ == "__main__":
    run_benchmark(
        label="PINN",
        description="Benchmark: classical PINN for 2D lid-driven cavity flow (Re=10).",
        network_description=f"PINN(width={PINN_WIDTH}, length={PINN_LENGTH})",
        model_factory=build_model,
        results_file=PINN_RESULTS_FILE,
    )
