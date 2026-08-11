"""Benchmark a stream-function PINN on steady cylinder flow."""

from benchmark_common import df, run_benchmark
from common_config import PINN_LENGTH, PINN_PSIP_RESULTS_FILE, PINN_WIDTH


def build_model():
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["psi", "p"],
        width=PINN_WIDTH,
        length=PINN_LENGTH,
    )


if __name__ == "__main__":
    run_benchmark(
        label="PINN-PSIP",
        model_name="PINN",
        formulation="psip",
        description="PINN stream-function (psi,p) cylinder-flow benchmark.",
        network_description=f"PINN(width={PINN_WIDTH}, length={PINN_LENGTH}, outputs=[psi,p])",
        model_factory=build_model,
        results_file=PINN_PSIP_RESULTS_FILE,
    )
