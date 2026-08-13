import copy

import pytest
import torch
import torch.nn as nn

import deepflow as df


def test_joint_rff_embedding_matches_formula():
    df.manual_seed(123)
    model = df.RFFPINN(
        input_vars=["x", "y", "t"],
        output_vars=["u"],
        width=4,
        length=1,
        embed_dim=8,
        alpha=2.0,
    )
    embedding = model.net[0]
    inputs = torch.tensor(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        dtype=df.get_dtype(),
    )

    actual = embedding(inputs)
    projection = inputs @ embedding.B
    expected = torch.cat((torch.cos(projection), torch.sin(projection)), dim=-1)

    assert torch.equal(actual, expected)
    assert actual.shape == (2, 8)
    assert isinstance(model.net[1], nn.Linear)
    assert model.net[1].in_features == 8


def test_frequency_matrix_is_fixed_buffer():
    model = df.RFFPINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=4,
        length=1,
        embed_dim=8,
    )
    initial_B = model.net[0].B.clone()

    assert "net.0.B" in model.state_dict()
    assert "net.0.B" not in dict(model.named_parameters())
    assert "net.0.B" in dict(model.named_buffers())

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    prediction = model({"x": torch.tensor([0.1, 0.2]), "y": torch.tensor([0.3, 0.4])})["u"]
    prediction.square().mean().backward()
    optimizer.step()

    assert torch.equal(model.net[0].B, initial_B)


def test_global_seed_reproduces_frequency_matrix():
    df.manual_seed(17)
    first = df.RFFPINN(input_vars=["x", "y"], output_vars=["u"], embed_dim=8)

    df.manual_seed(17)
    second = df.RFFPINN(input_vars=["x", "y"], output_vars=["u"], embed_dim=8)

    df.manual_seed(18)
    third = df.RFFPINN(input_vars=["x", "y"], output_vars=["u"], embed_dim=8)

    assert torch.equal(first.net[0].B, second.net[0].B)
    assert not torch.equal(first.net[0].B, third.net[0].B)


def test_rff_pinn_supports_coordinate_derivatives():
    model = df.RFFPINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=8,
        length=2,
        embed_dim=16,
    )
    x = torch.linspace(0.0, 1.0, 5, requires_grad=True)
    y = torch.linspace(1.0, 2.0, 5, requires_grad=True)

    prediction = model({"x": x, "y": y})["u"]
    first_derivative = torch.autograd.grad(prediction.sum(), x, create_graph=True)[0]
    second_derivative = torch.autograd.grad(first_derivative.sum(), x, create_graph=True)[0]

    assert torch.isfinite(first_derivative).all()
    assert torch.isfinite(second_derivative).all()


def test_rff_buffer_respects_dtype_and_copying():
    try:
        df.set_dtype(torch.float64)
        model = df.RFFPINN(
            input_vars=["x", "y"],
            output_vars=["u"],
            width=4,
            length=1,
            embed_dim=8,
        )
        copied = copy.deepcopy(model)

        assert model.net[0].B.dtype == torch.float64
        assert all(parameter.dtype == torch.float64 for parameter in model.parameters())
        assert torch.equal(copied.net[0].B, model.net[0].B)
    finally:
        df.set_dtype(torch.float32)


def test_rff_pinn_pickle_round_trip(tmp_path):
    model = df.RFFPINN(
        input_vars=["x", "y"],
        output_vars=["u"],
        width=4,
        length=1,
        embed_dim=8,
    )
    path = tmp_path / "rff_model.pkl"
    model.save_as_pickle(str(path))

    loaded = df.load_from_pickle(str(path))

    assert isinstance(loaded, df.RFFPINN)
    assert torch.equal(loaded.net[0].B, model.net[0].B)


@pytest.mark.parametrize("embed_dim", [0, -2, 7, 2.0, True])
def test_invalid_embed_dim_raises(embed_dim):
    with pytest.raises(ValueError, match="embed_dim"):
        df.RFFPINN(embed_dim=embed_dim)


@pytest.mark.parametrize("alpha", [0.0, -1.0, float("inf"), float("nan"), True])
def test_invalid_alpha_raises(alpha):
    with pytest.raises(ValueError, match="alpha"):
        df.RFFPINN(alpha=alpha)
