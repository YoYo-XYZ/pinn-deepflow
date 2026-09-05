"""Public smoke seam for the shared-harness cavity benchmark."""

import numpy as np

from benchmarks.comparing_pinns_qcpinn_formulations_cavity.benchmark import (
    SMOKE_CONFIG,
    run_suite,
)
from benchmarks.comparing_pinns_qcpinn_formulations_cavity.compare import (
    run_comparison,
)
from benchmarks.comparing_pinns_qcpinn_formulations_cavity.reference import (
    load_cached_reference,
)


def test_cavity_standard_variants_run_shared_smoke_path(tmp_path):
    variants = ("PINN-UVP", "PINN-PSIP")
    result = run_suite(
        config=SMOKE_CONFIG,
        output_dir=tmp_path,
        variants=variants,
    )

    assert set(result["variants"]) == set(variants)
    assert result["report"].exists()
    comparison = run_comparison(
        model_paths={
            variant: values["model_path"]
            for variant, values in result["variants"].items()
        },
        config=SMOKE_CONFIG,
        output_dir=tmp_path / "comparison",
        variants=variants,
        reference_solution=None,
    )
    assert comparison["report"].exists()
    for values in result["variants"].values():
        assert values["evaluator"].data_dict["u"].size > 0
        assert values["profiles"]["vertical"].data_dict["u"].size > 0
        assert values["profiles"]["horizontal"].data_dict["v"].size > 0
        assert np.isfinite(values["metrics"]["final_total_loss"])
        assert all(path.exists() for path in values["artifacts"])


def test_cavity_reference_cache_is_explicit_offline_fallback(tmp_path):
    axis = np.linspace(0.0, 1.0, 3)
    grid_x, grid_y = np.meshgrid(axis, axis, indexing="xy")
    cache_path = tmp_path / "reference.npz"
    np.savez(
        cache_path,
        x=axis,
        y=axis,
        u=grid_x + grid_y,
        v=grid_x - grid_y,
        p=grid_y,
    )

    reference = load_cached_reference(cache_path)
    values = reference.evaluate(
        np.array([0.25]),
        np.array([0.75]),
        fields=("u", "v", "p"),
    )

    assert reference.metadata["backend"] == "explicit_offline_cache"
    assert all(np.isfinite(values[field]).all() for field in ("u", "v", "p"))
