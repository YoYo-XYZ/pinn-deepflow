#!/usr/bin/env python3
"""
Unit tests for the configurable weight initialization in deepflow.
"""

import math
import os
import sys

import torch
import torch.nn as nn

# Resolve project source from the test directory.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import deepflow as df


def _first_linear(model):
    """Return the first nn.Linear module found in model.net."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            return m
    raise RuntimeError("No nn.Linear module found in model")


def test_default_is_kaiming():
    """Default PINN uses the same Kaiming uniform init as PyTorch nn.Linear."""
    torch.manual_seed(123)
    model_default = df.PINN(width=32, length=2, input_vars=["x", "y"], output_vars=["u"])
    lin = _first_linear(model_default)

    # Kaiming uniform with a=sqrt(5) => bound = 1/sqrt(fan_in)
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(lin.weight)
    bound = 1.0 / math.sqrt(fan_in)

    assert lin.weight.min() >= -bound, "Kaiming weight below lower bound"
    assert lin.weight.max() <= bound, "Kaiming weight above upper bound"
    assert lin.bias is not None, "PINN should have biases"
    assert lin.bias.min() >= -bound, "Kaiming bias below lower bound"
    assert lin.bias.max() <= bound, "Kaiming bias above upper bound"


def test_glorot_zero_bias():
    """Glorot/Xavier initialization zeros the biases."""
    model = df.PINN(
        width=32, length=2,
        input_vars=["x", "y"], output_vars=["u"],
        weight_init="glorot",
    )
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            assert torch.allclose(m.bias, torch.zeros_like(m.bias)), \
                "Glorot init should zero all biases"


def test_xavier_alias():
    """'xavier' is an accepted alias for 'glorot'."""
    model = df.PINN(
        width=8, length=1,
        input_vars=["x"], output_vars=["u"],
        weight_init="xavier",
    )
    # Just verify construction succeeds; bias-zero is tested above.
    assert isinstance(model, df.PINN)


def test_he_alias():
    """'he' is an accepted alias for 'kaiming'."""
    model = df.PINN(
        width=8, length=1,
        input_vars=["x"], output_vars=["u"],
        weight_init="he",
    )
    assert isinstance(model, df.PINN)


def test_none_leaves_default_pytorch_init():
    """weight_init=None leaves PyTorch's default initialization untouched."""
    torch.manual_seed(42)
    model_no_init = df.PINN(
        width=8, length=1,
        input_vars=["x"], output_vars=["u"],
        weight_init=None,
    )
    lin = _first_linear(model_no_init)
    # PyTorch default uses Kaiming uniform with a=sqrt(5); just verify
    # weights are not all zeros and finite.
    assert torch.isfinite(lin.weight).all()
    assert not torch.allclose(lin.weight, torch.zeros_like(lin.weight))


def test_custom_callable_initializer():
    """A callable weight_init is applied to the model."""
    def set_constant(model):
        for m in model.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.123)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.456)

    model = df.PINN(
        width=8, length=1,
        input_vars=["x"], output_vars=["u"],
        weight_init=set_constant,
    )
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert torch.allclose(m.weight, torch.full_like(m.weight, 0.123))
            assert torch.allclose(m.bias, torch.full_like(m.bias, 0.456))


def test_invalid_weight_init_raises():
    """An unknown string weight_init raises ValueError."""
    try:
        df.PINN(
            width=8, length=1,
            input_vars=["x"], output_vars=["u"],
            weight_init="unknown",
        )
    except ValueError as exc:
        assert "Unknown weight_init" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unknown weight_init")


def test_fnn_supports_weight_init():
    """FNN also accepts and applies weight_init."""
    model = df.FNN(
        input_vars=["x", "y"], output_vars=["u"],
        hidden_layer=[16, 16],
        weight_init="glorot",
    )
    for m in model.modules():
        if isinstance(m, nn.Linear) and m.bias is not None:
            assert torch.allclose(m.bias, torch.zeros_like(m.bias))


def test_reproducibility_with_explicit_default():
    """Explicit default weight_init='kaiming' yields bit-exact weights as default."""
    torch.manual_seed(0)
    m1 = df.PINN(width=8, length=1, input_vars=["x"], output_vars=["u"])

    torch.manual_seed(0)
    m2 = df.PINN(width=8, length=1, input_vars=["x"], output_vars=["u"], weight_init="kaiming")

    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2), "Explicit default should match implicit default"


if __name__ == "__main__":
    test_default_is_kaiming()
    test_glorot_zero_bias()
    test_xavier_alias()
    test_he_alias()
    test_none_leaves_default_pytorch_init()
    test_custom_callable_initializer()
    test_invalid_weight_init_raises()
    test_fnn_supports_weight_init()
    test_reproducibility_with_explicit_default()
    print("All weight_init tests passed.")
