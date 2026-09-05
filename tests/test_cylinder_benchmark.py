"""Public smoke seam for the cylinder PINN/QCPINN benchmark."""

import numpy as np

from benchmarks.comparing_pinns_qcpinn_cylinder.benchmark import (
    SMOKE_CONFIG,
    run_suite,
)
from benchmarks.comparing_pinns_qcpinn_cylinder.compare import run_comparison


def test_cylinder_standard_variants_run_shared_smoke_path(tmp_path):
    result = run_suite(
        config=SMOKE_CONFIG,
        output_dir=tmp_path,
        variants=("PINN-UVP", "PINN-PSIP"),
    )

    assert set(result["variants"]) == {"PINN-UVP", "PINN-PSIP"}
    assert result["report"].exists()
    comparison = run_comparison(
        model_paths={
            variant: values["model_path"]
            for variant, values in result["variants"].items()
        },
        config=SMOKE_CONFIG,
        output_dir=tmp_path / "comparison",
        variants=("PINN-UVP", "PINN-PSIP"),
    )
    assert comparison["report"].exists()
    for values in result["variants"].values():
        assert values["evaluator"].data_dict["u"].size > 0
        assert values["profiles"]["outlet"].data_dict["u"].size > 0
        assert np.isfinite(values["metrics"]["final_total_loss"])
        assert all(path.exists() for path in values["artifacts"])
