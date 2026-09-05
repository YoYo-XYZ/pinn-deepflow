"""Thin model-construction wrapper for the PINN-UVP benchmark cell."""

try:  # Package execution.
    from .benchmark import DEFAULT_CONFIG, build_pinn_model, run_variant_cli
except ImportError:  # Direct script execution.
    from benchmark import DEFAULT_CONFIG, build_pinn_model, run_variant_cli


def build_model(config=DEFAULT_CONFIG):
    """Construct only the model-specific part of the benchmark."""
    return build_pinn_model("uvp", config)


if __name__ == "__main__":
    run_variant_cli("PINN-UVP", build_model)
