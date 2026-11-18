import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.fixture
def qnn():
    pytest.importorskip("pennylane")
    from deepflow import qnn

    return qnn


@pytest.mark.parametrize("layer_type", ["cascade", "hea"])
def test_qcpinn_supports_coordinate_derivatives_and_parameter_gradients(
    qnn, layer_type
):
    model = qnn.QCPINN(
        output_vars=["u"],
        nqubits=2,
        q_layer_type=layer_type,
        hidden_layer_pre=[3],
        hidden_layer_post=[3],
    )
    x = torch.tensor([0.2, 0.7], requires_grad=True)
    y = torch.tensor([0.3, 0.4], requires_grad=True)
    outputs = model({"x": x, "y": y})
    assert outputs["u"].shape == (2,)
    (derivative,) = torch.autograd.grad(outputs["u"].sum(), x, create_graph=True)
    (second,) = torch.autograd.grad(derivative.sum(), x, create_graph=True)
    (outputs["u"].square().mean() + second.square().mean()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


@pytest.mark.parametrize("feature_map", ["simple", "product", "chebyshev"])
@pytest.mark.parametrize("ansatz", ["simple", "hea", "cascade"])
def test_qpinn_feature_maps_and_ansatz_have_finite_outputs_and_gradients(
    qnn, feature_map, ansatz
):
    model = qnn.QPINN(
        output_vars=["u"], nqubits=2, q_depth=1, feature_map=feature_map, ansatz=ansatz
    )
    x = torch.tensor([0.2, 0.4], requires_grad=True)
    y = torch.tensor([0.1, 0.3], requires_grad=True)
    u = model({"x": x, "y": y})["u"]
    u.square().mean().backward()
    assert u.shape == (2,) and torch.isfinite(u).all()
    assert torch.isfinite(x.grad).all() and torch.isfinite(y.grad).all()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_qpinn_single_qubit_cascade_has_valid_gradients(qnn):
    model = qnn.QPINN(output_vars=["u"], nqubits=1, q_depth=1, ansatz="cascade")
    result = model({"x": torch.tensor([0.2]), "y": torch.tensor([0.4])})["u"]
    result.sum().backward()
    assert torch.isfinite(result).all()


def test_qcpinn_rejects_unknown_layer_type(qnn):
    with pytest.raises(ValueError, match="Unsupported q_layer_type"):
        qnn.QCPINN(q_layer_type="unknown")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"nqubits": 0},
        {"nqubits": True},
        {"nqubits": 1.5},
        {"q_depth": 0},
        {"q_depth": True},
        {"q_depth": 1.5},
        {"feature_map": "unknown"},
        {"ansatz": "unknown"},
        {"cost_func": "unknown"},
    ],
)
def test_qpinn_rejects_invalid_configuration(qnn, kwargs):
    with pytest.raises(ValueError):
        qnn.QPINN(**kwargs)


def test_seeded_quantum_model_and_state_dict_reproduce_predictions(qnn):
    inputs = {"x": torch.tensor([0.2, 0.7]), "y": torch.tensor([0.3, 0.4])}
    df.manual_seed(14)
    first = qnn.QPINN(nqubits=2, q_depth=1)
    df.manual_seed(14)
    second = qnn.QPINN(nqubits=2, q_depth=1)
    for key, value in first(inputs).items():
        torch.testing.assert_close(second(inputs)[key], value)
    df.manual_seed(17)
    restored = qnn.QPINN(nqubits=2, q_depth=1)
    restored.load_state_dict(first.state_dict())
    for key, value in first(inputs).items():
        torch.testing.assert_close(restored(inputs)[key], value)
