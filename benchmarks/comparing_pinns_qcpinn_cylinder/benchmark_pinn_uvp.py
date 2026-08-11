"""Benchmark a direct-output PINN on steady cylinder flow."""

from benchmark_common import df, run_benchmark
from common_config import PINN_LENGTH, PINN_UVP_RESULTS_FILE, PINN_WIDTH


def build_model():
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=["u", "v", "p"],
        width=PINN_WIDTH,
        length=PINN_LENGTH,
    )


if __name__ == "__main__":
    run_benchmark(
        label="PINN-UVP",
        model_name="PINN",
        formulation="uvp",
        description="PINN direct (u,v,p) cylinder-flow benchmark.",
        network_description=f"PINN(width={PINN_WIDTH}, length={PINN_LENGTH}, outputs=[u,v,p])",
        model_factory=build_model,
        results_file=PINN_UVP_RESULTS_FILE,
    )
