#!/usr/bin/env python3
"""Regression tests for device assignment and evaluation sampling settings."""

import os
import sys

import torch

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import deepflow as df
import deepflow.utility as _df_util


def test_device_assignment_updates_internal_device():
    original_device = _df_util.device
    try:
        df.device = "cpu"
        assert df.device == "cpu"
        assert df.get_device() == "cpu"
        assert _df_util.device == "cpu"
    finally:
        _df_util.device = original_device


def test_evaluator_define_time_preserves_sampling_and_exponential_scaling():
    original_device = _df_util.device
    try:
        _df_util.device = "cpu"
        geometry = df.custom_data(
            {
                "x": torch.linspace(0.0, 1.0, 4),
                "y": torch.linspace(0.0, 1.0, 4),
            }
        )
        model = df.PINN(
            input_vars=["x", "y"],
            output_vars=["u"],
            width=4,
            length=1,
        )

        evaluation = geometry.evaluate(model)
        evaluation.define_time(
            [0.0, 1.0],
            sampling_scheme="random",
            expo_scaling=True,
        )

        assert geometry.scheme == "random"
        assert geometry.expo_scaling is True
        assert geometry.T_.shape == geometry.X_.shape
    finally:
        _df_util.device = original_device
