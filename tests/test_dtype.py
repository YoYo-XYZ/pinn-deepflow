#!/usr/bin/env python3
"""
Tests for configurable floating-point dtype (FP32/FP64) in deepflow.
"""

import os
import sys

import torch

# Resolve project source from the test directory.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import deepflow as df
import deepflow.utility as _df_util


def _reset_dtype():
    """Restore the default FP32 state after each test."""
    df.set_dtype(torch.float32)


def _ensure_cpu():
    """Force CPU device for deterministic, lightweight tests."""
    _df_util.device = "cpu"


def test_default_dtype_is_float32():
    """Fresh import uses FP32 by default."""
    assert df.get_dtype() == torch.float32
    assert torch.get_default_dtype() == torch.float32


def test_set_dtype_float64():
    """Setting df.dtype updates both deepflow state and PyTorch default."""
    try:
        df.dtype = torch.float64
        assert df.get_dtype() == torch.float64
        assert torch.get_default_dtype() == torch.float64
    finally:
        _reset_dtype()


def test_set_dtype_function():
    """Explicit set_dtype/get_dtype API works and round-trips."""
    try:
        df.set_dtype(torch.float64)
        assert df.get_dtype() == torch.float64
        df.set_dtype(torch.float32)
        assert df.get_dtype() == torch.float32
    finally:
        _reset_dtype()


def test_invalid_dtype_raises():
    """Only FP32 and FP64 are accepted."""
    original = df.get_dtype()
    try:
        for bad in (torch.float16, torch.bfloat16, torch.int32, "float64", None):
            try:
                df.set_dtype(bad)
            except (ValueError, TypeError):
                pass
            else:
                raise AssertionError(f"Expected ValueError/TypeError for dtype={bad}")
    finally:
        df.set_dtype(original)


def test_pinn_parameters_use_global_dtype():
    """PINN weights/biases reflect the global dtype setting."""
    try:
        for target_dtype in (torch.float32, torch.float64):
            df.set_dtype(target_dtype)
            model = df.PINN(
                input_vars=["x", "y"],
                output_vars=["u"],
                width=8,
                length=2,
            )
            params = list(model.parameters())
            assert len(params) > 0
            assert all(p.dtype == target_dtype for p in params)
    finally:
        _reset_dtype()


def test_fnn_parameters_use_global_dtype():
    """FNN weights/biases reflect the global dtype setting."""
    try:
        df.set_dtype(torch.float64)
        model = df.FNN(
            input_vars=["x", "y"],
            output_vars=["u", "v"],
            hidden_layer=[10, 10],
        )
        assert all(p.dtype == torch.float64 for p in model.parameters())
    finally:
        _reset_dtype()


def test_geometry_sampling_dtype():
    """Geometry sampling respects the configured dtype."""
    try:
        for target_dtype in (torch.float32, torch.float64):
            df.set_dtype(target_dtype)
            area = df.geometry.rectangle([0.0, 1.0], [0.0, 1.0])
            area.sampling_area([10, 10], scheme="uniform")
            assert area.X.dtype == target_dtype
            assert area.Y.dtype == target_dtype

            line = df.geometry.line_horizontal(y=0.0, range_x=[0.0, 1.0])
            line.sampling_line(10, scheme="uniform")
            assert line.X.dtype == target_dtype
            assert line.Y.dtype == target_dtype
    finally:
        _reset_dtype()


def test_lhs_sampling_dtype():
    """Latin Hypercube sampling respects the configured dtype."""
    try:
        for target_dtype in (torch.float32, torch.float64):
            df.set_dtype(target_dtype)
            area = df.geometry.rectangle([0.0, 1.0], [0.0, 1.0])
            area.sampling_area(20, scheme="lhs")
            assert area.X.dtype == target_dtype
            assert area.Y.dtype == target_dtype
    finally:
        _reset_dtype()


def test_time_sampling_dtype():
    """Time-coordinate sampling respects the configured dtype."""
    try:
        for target_dtype in (torch.float32, torch.float64):
            df.set_dtype(target_dtype)
            area = df.geometry.rectangle([0.0, 1.0], [0.0, 1.0])
            area.sampling_area([5, 5], scheme="uniform")
            area.define_time((0.0, 1.0), sampling_scheme="uniform")
            area.process_coordinates()
            assert area.T_.dtype == target_dtype
    finally:
        _reset_dtype()


def test_time_sampling_dtype_scalar_range():
    """Scalar time range also respects the configured dtype (regression test)."""
    try:
        for target_dtype in (torch.float32, torch.float64):
            df.set_dtype(target_dtype)
            area = df.geometry.rectangle([0.0, 1.0], [0.0, 1.0])
            area.sampling_area([5, 5], scheme="uniform")
            area.define_time(0.5, sampling_scheme="uniform")
            area.process_coordinates()
            assert area.T_.dtype == target_dtype
            assert area.T_.shape[0] == area.X_.shape[0]
    finally:
        _reset_dtype()


def test_model_forward_preserves_dtype():
    """A full forward pass on sampled coordinates preserves the configured dtype."""
    _ensure_cpu()
    try:
        df.set_dtype(torch.float64)
        area = df.geometry.rectangle([0.0, 1.0], [0.0, 1.0])
        area.sampling_area([5, 5], scheme="uniform")
        area.define_pde(df.pde.HeatEquation(alpha=0.1))
        area.process_coordinates()

        model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=8, length=2)
        model.to(_df_util.get_device())
        outputs = area.process_model(model)
        assert outputs["u"].dtype == torch.float64
        assert area.X_.dtype == torch.float64
        assert area.Y_.dtype == torch.float64
    finally:
        _reset_dtype()


def test_quantum_models_respect_dtype():
    """QCPINN parameters use the configured dtype when PennyLane is available."""
    try:
        import pennylane as qml  # noqa: F401
    except ImportError:
        # Skip if PennyLane is not installed in the test environment.
        return

    try:
        df.set_dtype(torch.float64)
        qc_model = df.QCPINN(
            input_vars=["x", "y"],
            output_vars=["u"],
            nqubits=2,
            q_layer_type="cascade",
            q_layer_iterations=1,
            hidden_layer_pre=[4],
            hidden_layer_post=[4],
        )
        assert all(p.dtype == torch.float64 for p in qc_model.parameters())
    finally:
        _reset_dtype()


if __name__ == "__main__":
    test_default_dtype_is_float32()
    test_set_dtype_float64()
    test_set_dtype_function()
    test_invalid_dtype_raises()
    test_pinn_parameters_use_global_dtype()
    test_fnn_parameters_use_global_dtype()
    test_geometry_sampling_dtype()
    test_lhs_sampling_dtype()
    test_time_sampling_dtype()
    test_time_sampling_dtype_scalar_range()
    test_model_forward_preserves_dtype()
    test_quantum_models_respect_dtype()
    print("All dtype tests passed.")
