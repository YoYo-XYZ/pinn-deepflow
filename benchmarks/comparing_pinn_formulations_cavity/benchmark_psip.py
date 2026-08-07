#!/usr/bin/env python3
"""Benchmark a stream-function ``(psi, p)`` PINN on lid-driven cavity flow."""

from benchmark_common import df, run_benchmark
from common_config import PSIP_RESULTS_FILE, STREAM_HIDDEN_LAYERS


def build_model():
    return df.FNN(
        input_vars=["x", "y"],
        output_vars=["psi", "p"],
        hidden_layer=STREAM_HIDDEN_LAYERS,
    )


if __name__ == "__main__":
    run_benchmark(
        label="PSIP",
        formulation="psip",
        description="Benchmark: stream-function (psi,p) PINN for Re=10 cavity flow.",
        network_description=(
            f"FNN(hidden_layer={STREAM_HIDDEN_LAYERS}, outputs=[psi,p])"
        ),
        pde_residual_count=2,
        model_factory=build_model,
        results_file=PSIP_RESULTS_FILE,
    )
