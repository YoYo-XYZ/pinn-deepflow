import warnings

import pytest
import torch
import torch.nn as nn

import deepflow as df


class CustomActivation(nn.Module):
    def forward(self, inputs):
        return inputs


class TanhSubclass(nn.Tanh):
    pass


@pytest.fixture(autouse=True)
def restore_deepflow_configuration():
    original_device = df.device
    original_dtype = df.dtype
    yield
    df.device = original_device
    df.dtype = original_dtype


def _state_copy(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _small_inputs(dtype=torch.float32):
    return {
        "x": torch.tensor([0.0, 0.25], dtype=dtype),
        "y": torch.tensor([0.5, 0.75], dtype=dtype),
    }


def test_pinn_pt_round_trip_through_public_interfaces(tmp_path):
    df.device = "cpu"
    df.dtype = torch.float64
    model = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=4,
        length=2,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(0.25)
    model.loss_history = {
        "total_loss": [1.0, 0.5],
        "bc_loss": [0.75, 0.25],
        "pde_loss": [0.25, 0.25],
    }
    model.train()
    inputs = _small_inputs(torch.float64)
    expected = model(inputs)
    expected_state = _state_copy(model)

    path = tmp_path / "model"
    assert model.save(path) is None
    loaded = df.load_model(path)

    assert isinstance(loaded, df.PINN)
    assert loaded.input_keys == model.input_keys
    assert loaded.output_keys == model.output_keys
    assert loaded.hidden_layer == model.hidden_layer
    assert loaded.loss_history == model.loss_history
    assert loaded.training is False
    assert all(parameter.dtype == torch.float64 for parameter in loaded.parameters())
    assert all(torch.equal(loaded.state_dict()[key], value) for key, value in expected_state.items())
    actual = loaded(inputs)
    assert torch.equal(actual["u"], expected["u"])


@pytest.mark.parametrize(
    ("model_class", "model_kwargs"),
    [
        (df.FNN, {"hidden_layer": [4, 3]}),
        (df.PINN, {"width": 4, "length": 2}),
        (df.RFFPINN, {"width": 4, "length": 2, "embed_dim": 8, "alpha": 2.0}),
    ],
)
def test_supported_model_types_round_trip(tmp_path, model_class, model_kwargs):
    df.device = "cpu"
    model = model_class(
        input_vars=["x", "y"],
        output_vars=["u"],
        **model_kwargs,
    )
    model.loss_history["total_loss"] = [3.0]
    model.loss_history["bc_loss"] = [1.0]
    model.loss_history["pde_loss"] = [2.0]
    expected = model(_small_inputs())

    path = tmp_path / f"{model_class.__name__}.pt"
    model.save(path)
    loaded = df.load_model(path, device="cpu")

    assert type(loaded) is model_class
    assert loaded.input_keys == model.input_keys
    assert loaded.output_keys == model.output_keys
    assert loaded.hidden_layer == model.hidden_layer
    assert loaded.loss_history == model.loss_history
    assert all(
        torch.equal(loaded_state, model_state)
        for loaded_state, model_state in zip(
            loaded.state_dict().values(), model.state_dict().values()
        )
    )
    actual = loaded(_small_inputs())
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    if isinstance(model, df.RFFPINN):
        assert loaded.embed_dim == model.embed_dim
        assert loaded.alpha == model.alpha
        assert torch.equal(loaded.net[0].B, model.net[0].B)


@pytest.mark.parametrize(
    ("activation", "activation_identifier", "activation_kwargs"),
    [
        (nn.Tanh(), "tanh", {}),
        (nn.Sigmoid(), "sigmoid", {}),
        (nn.ReLU(inplace=True), "relu", {"inplace": True}),
        (
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            "leaky_relu",
            {"negative_slope": 0.2, "inplace": True},
        ),
        (nn.ELU(alpha=2.0, inplace=True), "elu", {"alpha": 2.0, "inplace": True}),
        (nn.GELU(approximate="tanh"), "gelu", {"approximate": "tanh"}),
        (nn.SiLU(inplace=True), "silu", {"inplace": True}),
        (
            nn.Softplus(beta=2.0, threshold=10.0),
            "softplus",
            {"beta": 2.0, "threshold": 10.0},
        ),
        (nn.Identity(), "identity", {}),
    ],
)
def test_registered_activation_configuration_round_trip(
    tmp_path, activation, activation_identifier, activation_kwargs
):
    df.device = "cpu"
    model = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=3,
        length=1,
        activation=activation,
    )
    path = tmp_path / "activation.pt"
    model.save(path)

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    assert artifact["model"]["config"]["activation"] == {
        "type": activation_identifier,
        "kwargs": activation_kwargs,
    }

    loaded = df.load_model(path, device="cpu")

    assert type(loaded.activation) is type(activation)
    assert loaded(_small_inputs())["u"].equal(model(_small_inputs())["u"])
    for name, value in activation_kwargs.items():
        assert getattr(loaded.activation, name) == value


@pytest.mark.parametrize("activation", [CustomActivation(), TanhSubclass()])
def test_unsupported_activation_is_rejected_before_destination_changes(
    tmp_path, activation
):
    df.device = "cpu"
    path = tmp_path / "activation.pt"
    path.write_bytes(b"previous artifact")
    model = df.PINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=3,
        length=1,
        activation=activation,
    )

    with pytest.raises(df.ModelPersistenceError):
        model.save(path)

    assert path.read_bytes() == b"previous artifact"


def test_artifact_activation_identifier_cannot_direct_an_import(tmp_path):
    df.device = "cpu"
    path = tmp_path / "activation.pt"
    df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1).save(path)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    artifact["model"]["config"]["activation"] = {
        "type": "torch.nn.ReLU",
        "kwargs": {},
    }
    torch.save(artifact, path)

    with pytest.raises(df.ModelPersistenceError):
        df.load_model(path, device="cpu")


def test_dtype_and_default_device_are_not_global_load_settings(tmp_path):
    df.device = "cpu"
    df.dtype = torch.float32
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)
    model.to(torch.float64)
    path = tmp_path / "double.pt"
    model.save(path)

    random_state = torch.get_rng_state()
    default_dtype = torch.get_default_dtype()
    loaded = df.load_model(path, device="cpu")

    assert next(loaded.parameters()).dtype == torch.float64
    assert df.dtype == torch.float32
    assert df.device == "cpu"
    assert torch.get_default_dtype() == default_dtype
    assert torch.equal(torch.get_rng_state(), random_state)


def test_artifact_is_versioned_and_cpu_normalized(tmp_path):
    df.device = "cpu"
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)
    path = tmp_path / "model.pt"
    model.save(path)

    artifact = torch.load(path, map_location="cpu", weights_only=True)
    assert set(artifact) == {
        "format",
        "version",
        "versions",
        "model",
        "dtype",
        "state_dict",
        "loss_history",
    }
    assert artifact["format"] == "deepflow.model"
    assert artifact["version"] == 1
    assert set(artifact["versions"]) == {"deepflow", "torch"}
    assert all(value.device.type == "cpu" for value in artifact["state_dict"].values())
    assert artifact["dtype"] == "float32"


def test_suffix_rules_and_missing_parent_filesystem_error(tmp_path):
    df.device = "cpu"
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)

    model.save(tmp_path / "model")
    assert (tmp_path / "model.pt").exists()
    assert isinstance(df.load_model(tmp_path / "model"), df.PINN)

    with pytest.raises(df.ModelPersistenceError, match="suffix"):
        model.save(tmp_path / "model.pkl")
    with pytest.raises(df.ModelPersistenceError, match="suffix"):
        df.load_model(tmp_path / "model.pkl")

    missing_parent = tmp_path / "missing" / "model"
    with pytest.raises(FileNotFoundError):
        model.save(missing_parent)
    assert not missing_parent.parent.exists()


def test_failed_write_preserves_existing_artifact(tmp_path, monkeypatch):
    df.device = "cpu"
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)
    path = tmp_path / "model.pt"
    model.save(path)
    original_bytes = path.read_bytes()

    def fail_save(*args, **kwargs):
        raise RuntimeError("write failed")

    monkeypatch.setattr(torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="write failed"):
        model.save(path)

    assert path.read_bytes() == original_bytes
    assert list(tmp_path.glob(".model.pt.*.tmp")) == []


def test_unsupported_model_behavior_is_rejected_before_writing(tmp_path):
    df.device = "cpu"
    path = tmp_path / "model.pt"
    path.write_bytes(b"previous")

    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)
    model.hard_constraints = {"u": lambda coords: coords[0]}
    with pytest.raises(df.ModelPersistenceError, match="hard constraints"):
        model.save(path)
    assert path.read_bytes() == b"previous"

    model = df.PINN(
        input_vars=["x"],
        output_vars=["u"],
        width=3,
        length=1,
        weight_init=lambda current: None,
    )
    with pytest.raises(df.ModelPersistenceError, match="callable"):
        model.save(path)
    assert path.read_bytes() == b"previous"


def test_restricted_loader_uses_cpu_and_weights_only(tmp_path, monkeypatch):
    df.device = "cpu"
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)
    path = tmp_path / "model.pt"
    model.save(path)

    real_load = torch.load
    calls = []

    def tracked_load(*args, **kwargs):
        calls.append(kwargs)
        return real_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", tracked_load)
    df.load_model(path, device="cpu")

    assert calls == [{"map_location": "cpu", "weights_only": True}]


def test_version_mismatch_emits_one_warning_and_still_loads(tmp_path):
    df.device = "cpu"
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)
    path = tmp_path / "model.pt"
    model.save(path)
    artifact = torch.load(path, weights_only=True)
    artifact["versions"] = {"deepflow": "0.0.0", "torch": "0.0.0"}
    torch.save(artifact, path)

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        loaded = df.load_model(path, device="cpu")

    assert isinstance(loaded, df.PINN)
    assert len(recorded) == 1
    assert "0.0.0" in str(recorded[0].message)


@pytest.mark.parametrize("mutation", [
    lambda artifact: artifact.update(extra=True),
    lambda artifact: artifact.__setitem__("version", 99),
    lambda artifact: artifact.__setitem__("dtype", "float16"),
    lambda artifact: artifact["loss_history"].__setitem__("total_loss", [True]),
])
def test_malformed_artifacts_raise_public_persistence_error(tmp_path, mutation):
    df.device = "cpu"
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=3, length=1)
    path = tmp_path / "model.pt"
    model.save(path)
    artifact = torch.load(path, weights_only=True)
    mutation(artifact)
    torch.save(artifact, path)

    with pytest.raises(df.ModelPersistenceError):
        df.load_model(path, device="cpu")
