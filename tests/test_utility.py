import random
import subprocess
import sys

import numpy as np
import pytest
import torch

import deepflow as df
from deepflow.utility import calc_grad, calc_grads, to_require_grad, torch_to_numpy

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_controls_sampling_and_new_models(dtype):
    df.dtype = dtype
    assert df.dtype == df.get_dtype() == torch.get_default_dtype() == dtype
    area = df.geometry.rectangle([0, 1], [0, 1])
    x, y = area.sampling_area(3)
    assert x.dtype == y.dtype == dtype
    model = df.PINN(width=4, length=1)
    assert all(p.dtype == dtype for p in model.parameters())
    assert df.latin_hypercube_sampling(4, 2, [0, 0], [1, 1]).dtype == dtype
    assert df.get_device() == "cpu"


@pytest.mark.parametrize("dtype", [torch.float16, torch.int64, "float64", None])
def test_invalid_dtype_preserves_configuration(dtype):
    with pytest.raises(ValueError, match="dtype must be"):
        df.set_dtype(dtype)
    assert df.dtype == torch.get_default_dtype() == torch.float64


def test_manual_seed_replays_all_random_sources_and_advances_between_calls():
    def draw():
        return (
            random.random(),
            np.random.rand(3),
            torch.rand(3),
            df.latin_hypercube_sampling(8, 2, [-2, 3], [4, 8]),
        )

    df.manual_seed(42, deterministic=True)
    first, second = draw(), draw()
    df.manual_seed(42, deterministic=True)
    for expected, actual in zip(first, draw()):
        np.testing.assert_array_equal(expected, actual)
    assert not torch.equal(first[-1], second[-1])
    assert torch.are_deterministic_algorithms_enabled()


def test_lhs_places_one_sample_in_each_stratum_per_axis():
    lower, upper = torch.tensor([-2.0, 3.0]), torch.tensor([4.0, 8.0])
    points = df.latin_hypercube_sampling(16, 2, lower, upper, seed=7)
    normalized = (points - lower) / (upper - lower)
    assert points.shape == (16, 2)
    assert ((normalized >= 0) & (normalized < 1)).all()
    strata = torch.floor(normalized * 16).sort(dim=0).values
    torch.testing.assert_close(strata, torch.arange(16.0)[:, None].expand(-1, 2))
    df.manual_seed(999)
    torch.testing.assert_close(
        points, df.latin_hypercube_sampling(16, 2, lower, upper, seed=7)
    )


def test_autograd_helpers_preserve_higher_derivatives_and_unused_inputs():
    x, y = to_require_grad(torch.tensor([0.2, 0.7]), torch.tensor([0.3, 0.8]))
    value = x**3
    dx, dy = calc_grads(value, [x, y])
    torch.testing.assert_close(dx, 3 * x**2)
    torch.testing.assert_close(dy, torch.zeros_like(y))
    torch.testing.assert_close(calc_grad(dx, x), 6 * x)
    torch.testing.assert_close(calc_grad(value, y), torch.zeros_like(y))


def test_tensor_conversion_detaches_and_clones_without_changing_source():
    source = torch.tensor([1.0, 2.0], requires_grad=True) * 2
    copy = to_require_grad(source)
    assert copy.is_leaf and copy.requires_grad
    with torch.no_grad():
        copy[0] = 99
    torch.testing.assert_close(source, torch.tensor([2.0, 4.0]))
    np.testing.assert_array_equal(torch_to_numpy(source), [2, 4])
    arrays = torch_to_numpy(source, copy)
    assert isinstance(arrays, tuple) and all(isinstance(a, np.ndarray) for a in arrays)


def test_core_workflow_imports_without_optional_quantum_dependency():
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockPennyLane(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] == 'pennylane':
            raise ModuleNotFoundError('blocked optional dependency', name='pennylane')

sys.meta_path.insert(0, BlockPennyLane())
import torch
import deepflow as df
model = df.PINN(width=3, length=1)
outputs = model({'x': torch.tensor([0.2]), 'y': torch.tensor([0.4])})
assert set(outputs) == {'u', 'v', 'p'}
assert all(torch.isfinite(value).all() for value in outputs.values())
assert 'pennylane' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )
