import math

import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


def zero_weights(model):
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)


@pytest.mark.parametrize(
    "model_type,options",
    [
        (df.FNN, {"hidden_layer": [4, 3]}),
        (df.PINN, {"width": 4, "length": 2}),
        (df.RFFPINN, {"width": 4, "length": 2, "embed_dim": 8}),
    ],
)
def test_networks_return_named_fields_with_coordinate_and_parameter_gradients(
    model_type, options
):
    model = model_type(
        input_vars=["x", "y", "t"], output_vars=["temperature"], **options
    )
    inputs = {k: torch.linspace(0.1, 0.9, 5).requires_grad_() for k in ["x", "y", "t"]}
    outputs = model(inputs)
    assert list(outputs) == ["temperature"]
    assert outputs["temperature"].shape == (5,)
    (gradient,) = torch.autograd.grad(
        outputs["temperature"].sum(), inputs["t"], create_graph=True
    )
    (second,) = torch.autograd.grad(gradient.sum(), inputs["t"], retain_graph=True)
    assert torch.isfinite(second).all()
    outputs["temperature"].square().mean().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )


def test_input_order_and_output_names_control_mapping():
    def initialize(model):
        for layer in model.modules():
            if isinstance(layer, torch.nn.Linear):
                with torch.no_grad():
                    layer.weight.copy_(torch.tensor([[2.0, 3.0], [-1.0, 4.0]]))
                    layer.bias.copy_(torch.tensor([0.5, -0.5]))

    model = df.FNN(
        input_vars=["y", "x"],
        output_vars=["v", "u"],
        hidden_layer=[],
        weight_init=initialize,
    )
    x, y = torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])
    output = model({"x": x, "y": y, "extra": x})
    torch.testing.assert_close(output["v"], 2 * y + 3 * x + 0.5)
    torch.testing.assert_close(output["u"], -y + 4 * x - 0.5)
    with pytest.raises(KeyError, match="y"):
        model({"x": x})


@pytest.mark.parametrize(
    "weight_init", [None, "kaiming", "he", "xavier", "glorot", zero_weights]
)
def test_weight_initialization_is_seeded_and_produces_finite_outputs(weight_init):
    df.manual_seed(42)
    first = df.PINN(width=4, length=1, weight_init=weight_init)
    df.manual_seed(42)
    second = df.PINN(width=4, length=1, weight_init=weight_init)
    inputs = {"x": torch.tensor([0.2, 0.7]), "y": torch.tensor([0.3, 0.4])}
    for key, value in first(inputs).items():
        torch.testing.assert_close(value, second(inputs)[key])
        assert torch.isfinite(value).all()
        if weight_init is zero_weights:
            torch.testing.assert_close(value, torch.zeros_like(value))


def test_unknown_initializer_is_rejected():
    with pytest.raises(ValueError, match="Unknown weight_init"):
        df.PINN(width=3, length=1, weight_init="unknown")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"embed_dim": 0},
        {"embed_dim": 3},
        {"embed_dim": True},
        {"embed_dim": 2.5},
        {"alpha": 0},
        {"alpha": -1},
        {"alpha": math.inf},
        {"alpha": math.nan},
        {"alpha": True},
    ],
)
def test_rff_rejects_invalid_embedding_parameters(kwargs):
    with pytest.raises(ValueError, match="embed_dim|alpha"):
        df.RFFPINN(width=3, length=1, **kwargs)


def test_rff_state_dict_roundtrip_preserves_random_features():
    inputs = {"x": torch.linspace(0, 1, 7), "y": torch.linspace(1, 2, 7)}
    df.manual_seed(1)
    model = df.RFFPINN(width=4, length=1, embed_dim=8)
    df.manual_seed(2)
    restored = df.RFFPINN(width=4, length=1, embed_dim=8)
    assert not torch.equal(model(inputs)["u"], restored(inputs)["u"])
    restored.load_state_dict(model.state_dict())
    for key, value in model(inputs).items():
        torch.testing.assert_close(restored(inputs)[key], value)


def test_hard_constraints_enforce_constants_on_multiple_walls_and_survive_pickle(
    tmp_path,
):
    area = df.geometry.rectangle([0, 1], [0, 1])
    for boundary in area:
        boundary.define_bc({"u": df.hard_constraint(2), "v": 0})
    model = df.PINN(width=4, length=1)
    model.apply_hard_constraints(area.bound_list)
    path = str(tmp_path / "constrained")
    model.save_as_pickle(path)
    restored = df.load_from_pickle(path)
    assert (tmp_path / "constrained.pkl").is_file()
    for boundary in area:
        x, y = boundary.sampling_line(7)
        inputs = {"x": x, "y": y}
        torch.testing.assert_close(model(inputs)["u"], torch.full_like(x, 2))
        for key, value in model(inputs).items():
            torch.testing.assert_close(restored(inputs)[key], value)
    interior = {"x": torch.tensor([0.3]), "y": torch.tensor([0.4])}
    assert model(interior)["u"].item() != 2


def test_conflicting_hard_constraints_fail_clearly():
    left, right = (
        df.geometry.line_vertical(0, [0, 1]),
        df.geometry.line_vertical(1, [0, 1]),
    )
    left.define_bc({"u": df.hard_constraint(0)})
    right.define_bc({"u": df.hard_constraint(1)})
    with pytest.raises(ValueError, match="Conflicting hard constraint constants.*u"):
        df.PINN(width=3, length=1).apply_hard_constraints([left, right])


@pytest.mark.parametrize("suffix", ["", ".pkl"])
def test_pickle_roundtrip_preserves_predictions_dtype_and_history(tmp_path, suffix):
    model = df.PINN(width=4, length=1)
    model.loss_history["total_loss"] = [3.0, 1.0]
    path = str(tmp_path / ("model" + suffix))
    model.save_as_pickle(path)
    restored = df.load_from_pickle(path)
    inputs = {"x": torch.tensor([0.1, 0.4]), "y": torch.tensor([0.6, 0.8])}
    assert restored.loss_history == model.loss_history
    for key, value in model(inputs).items():
        torch.testing.assert_close(restored(inputs)[key], value)


def test_missing_pickle_file_reports_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        df.load_from_pickle(str(tmp_path / "missing"))
