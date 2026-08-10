"""Focused tests for QCPINN ansatz selection and circuit structure."""

import os
import sys

import pytest
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

qml = pytest.importorskip("pennylane")
import deepflow as df


def _quantum_layer(model):
    return next(module for module in model.modules() if isinstance(module, qml.qnn.TorchLayer))


def _operations(q_layer, nqubits, iterations):
    q_layer.qnode(torch.zeros(nqubits), q_layer.weights)
    return q_layer.qnode._tape.operations[1:]


def test_cascade_is_the_default_and_matches_published_structure():
    model = df.QCPINN(input_vars=["x", "y"], output_vars=["u"], nqubits=4, q_layer_iterations=2)
    q_layer = _quantum_layer(model)

    assert model.q_layer_type == "cascade"
    assert tuple(q_layer.weights.shape) == (2, 3, 4)
    operations = _operations(q_layer, 4, 2)
    assert q_layer.qnode._tape.operations[0].hyperparameters["rotation"] is qml.RX

    expected = (
        [("RX", (i,)) for i in range(4)]
        + [("RZ", (i,)) for i in range(4)]
        + [("CRX", (3, 0)), ("CRX", (0, 1)), ("CRX", (1, 2)), ("CRX", (2, 3))]
    ) * 2
    actual = [(operation.name, tuple(operation.wires)) for operation in operations]
    assert actual == expected


def test_pictured_hea_uses_three_rotations_and_linear_cnot_entanglers():
    model = df.QCPINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        nqubits=4,
        q_layer_type="hea",
        q_layer_iterations=2,
    )
    q_layer = _quantum_layer(model)

    assert tuple(q_layer.weights.shape) == (2, 3, 4)
    expected_layer = (
        [(gate, (i,)) for i in range(4) for gate in ("RX", "RY", "RZ")]
        + [("CNOT", (0, 1)), ("CNOT", (1, 2)), ("CNOT", (2, 3))]
    )
    operations = _operations(q_layer, 4, 2)
    assert [(operation.name, tuple(operation.wires)) for operation in operations] == expected_layer * 2
    assert all(len(operation.parameters) == 0 for operation in operations if operation.name == "CNOT")


def test_q_layer_type_is_case_insensitive():
    model = df.QCPINN(input_vars=["x"], output_vars=["u"], nqubits=2, q_layer_type="HEA")
    assert model.q_layer_type == "hea"


def test_invalid_q_layer_type_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported q_layer_type.*cascade.*hea"):
        df.QCPINN(input_vars=["x"], output_vars=["u"], nqubits=2, q_layer_type="layered")
