#!/usr/bin/env python3
"""Benchmark a QCPINN with stream-function (psi,p) outputs."""

import sys

try:
    import pennylane  # noqa: F401
except ImportError:
    print(
        "[ERROR] PennyLane is required for QCPINN. Install it with: "
        "pip install pennylane"
    )
    sys.exit(1)

from benchmark_common import df, run_benchmark  # noqa: E402
from common_config import (  # noqa: E402
    QC_ITERATIONS,
    QC_NQUBITS,
    QC_POST,
    QC_PRE,
    QCPINN_PSIP_RESULTS_FILE,
)


def build_model():
    return df.QCPINN(
        input_vars=["x", "y"],
        output_vars=["psi", "p"],
        hidden_layer_pre=QC_PRE,
        hidden_layer_post=QC_POST,
        nqubits=QC_NQUBITS,
        q_layer_iterations=QC_ITERATIONS,
    )


if __name__ == "__main__":
    network = (
        f"QCPINN(pre={QC_PRE}, post={QC_POST}, nqubits={QC_NQUBITS}, "
        f"q_layer_iterations={QC_ITERATIONS}, outputs=[psi,p])"
    )
    run_benchmark(
        label="QCPINN-PSIP",
        model_name="QCPINN",
        formulation="psip",
        description="Benchmark: QCPINN with stream-function (psi,p) formulation.",
        network_description=network,
        model_factory=build_model,
        results_file=QCPINN_PSIP_RESULTS_FILE,
    )
