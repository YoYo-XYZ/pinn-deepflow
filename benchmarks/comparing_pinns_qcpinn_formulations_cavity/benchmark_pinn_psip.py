"""Thin model-construction wrapper for the PINN-PSIP cavity variant."""

try:  # Package execution.
    from .benchmark import DEFAULT_CONFIG, build_pinn_model, run_variant_cli
except ImportError:  # Direct script execution.
    from benchmark import DEFAULT_CONFIG, build_pinn_model, run_variant_cli


def build_model(config=DEFAULT_CONFIG):
    """Construct only the standard streamfunction-pressure model."""
    return build_pinn_model("psip", config)


if __name__ == "__main__":
    run_variant_cli("PINN-PSIP", build_model)
