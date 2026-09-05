"""Thin model-construction wrapper for the QCPINN-PSIP benchmark cell."""

try:  # Package execution.
    from .benchmark import DEFAULT_CONFIG, build_qcpinn_model, run_variant_cli
except ImportError:  # Direct script execution.
    from benchmark import DEFAULT_CONFIG, build_qcpinn_model, run_variant_cli


def build_model(config=DEFAULT_CONFIG):
    """Construct only the optional quantum model variant."""
    return build_qcpinn_model("psip", config)


if __name__ == "__main__":
    run_variant_cli("QCPINN-PSIP", build_model)
