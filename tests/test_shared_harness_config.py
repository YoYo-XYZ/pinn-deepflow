"""Boundary smoke for the shared configuration concept."""

import pytest

from benchmarks.shared_harness.config import BenchmarkConfig


def test_config_covers_cloned_knobs():
    config = BenchmarkConfig(
        width=16,
        depth=4,
        learning_rate=0.004,
        epochs_adam=2,
        epochs_lbfgs=1,
        seed=69,
        boundary_points=[8, 4, 4],
        interior_points=[16],
        eval_grid=[8, 8],
        sampling="lhs",
    )
    payload = config.to_dict()
    assert payload["width"] == 16
    assert payload["depth"] == 4
    assert payload["learning_rate"] == 0.004
    assert payload["epochs_adam"] == 2
    assert payload["seed"] == 69
    assert payload["seeds"] == [69]
    assert payload["boundary_points"] == [8, 4, 4]
    assert payload["interior_points"] == [16]


def test_config_normalizes_scalar_interior_count():
    config = BenchmarkConfig(interior_points=16)

    assert config.to_dict()["interior_points"] == [16]


def test_config_supports_multiple_seeds():
    config = BenchmarkConfig(seeds=[70, 71])

    assert config.seed == 70
    assert config.to_dict()["seeds"] == [70, 71]


def test_config_rejects_bad_knobs():
    with pytest.raises(ValueError):
        BenchmarkConfig(width=0)
    with pytest.raises(ValueError):
        BenchmarkConfig(learning_rate=0.0)
    with pytest.raises(ValueError):
        BenchmarkConfig(epochs_adam=-1)
    with pytest.raises(ValueError):
        BenchmarkConfig(boundary_points=[])
    with pytest.raises(ValueError):
        BenchmarkConfig(sampling="r3")
    with pytest.raises(ValueError):
        BenchmarkConfig(seeds=[])
