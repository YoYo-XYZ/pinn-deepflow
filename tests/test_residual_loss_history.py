import math

import numpy as np
import pytest
import torch

import deepflow as df
import deepflow.utility as _df_util


@pytest.fixture(autouse=True)
def _cpu_device():
    original_device = _df_util.device
    _df_util.device = "cpu"
    try:
        yield
    finally:
        _df_util.device = original_device


def _custom_domain(residual_names=("first", "second")):
    geometry = df.custom_data(
        {
            "x": torch.linspace(0.0, 1.0, 4),
            "y": torch.linspace(1.0, 2.0, 4),
        }
    )
    geometry.define_pde(
        df.CustomPDE(
            lambda values: (values["u"], 2.0 * values["u"]),
            residual_names=residual_names,
        )
    )
    return geometry, df.domain(geometry)


def _model(output_vars=("u",)):
    return df.PINN(
        input_vars=["x", "y"],
        output_vars=list(output_vars),
        width=4,
        length=1,
    )


def test_custom_pde_residual_names_and_fallbacks():
    values = {"u": torch.ones(3)}

    named = df.CustomPDE(
        lambda inputs: (inputs["u"], -inputs["u"]),
        residual_names=("positive", "negative"),
    )
    named.compute_residuals(values)
    assert named.get_residual_names() == ("positive", "negative")

    unnamed = df.CustomPDE(lambda inputs: (inputs["u"], -inputs["u"]))
    unnamed.compute_residuals(values)
    assert unnamed.get_residual_names() == ("residual_0", "residual_1")


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (("only_one",), "1 residual names for 2 residual fields"),
        (("", "valid"), "non-empty strings"),
        ((1, "valid"), "non-empty strings"),
        (("same", "same"), "unique within a PDE"),
    ],
)
def test_custom_pde_rejects_invalid_residual_names(names, message):
    pde = df.CustomPDE(
        lambda inputs: (inputs["u"], -inputs["u"]),
        residual_names=names,
    )
    pde.compute_residuals({"u": torch.ones(3)})

    with pytest.raises(ValueError, match=message):
        pde.get_residual_names()


def test_component_losses_preserve_simple_and_weighted_totals():
    _, domain = _custom_domain()
    model = _model()

    simple = df.calc_loss_simple(domain)(model)
    weighted = df.calc_loss_weighted(domain, pde_weights=3.0)(model)

    component_sum = simple["pde_loss_first"] + simple["pde_loss_second"]
    assert torch.allclose(simple["pde_loss"], component_sum)
    assert torch.allclose(simple["total_loss"], simple["pde_loss"])
    assert torch.allclose(weighted["total_loss"], 3.0 * weighted["pde_loss"])
    assert torch.allclose(
        weighted["pde_loss"],
        weighted["pde_loss_first"] + weighted["pde_loss_second"],
    )


def test_component_losses_sum_matching_names_across_geometries():
    geometry_a, _ = _custom_domain(residual_names=("shared", "other"))
    geometry_b, _ = _custom_domain(residual_names=("shared", "other"))
    domain = df.domain(geometry_a, geometry_b)

    losses = df.calc_loss_simple(domain)(_model())
    expected_shared = sum(
        torch.mean(geometry.residual_field_raw[0].square())
        for geometry in (geometry_a, geometry_b)
    )

    assert torch.allclose(losses["pde_loss_shared"], expected_shared)
    assert torch.allclose(
        losses["pde_loss"],
        losses["pde_loss_shared"] + losses["pde_loss_other"],
    )


def test_loss_recorder_backfills_new_component_histories(capsys):
    model = _model()
    model.loss_history = {
        "total_loss": [2.0, 1.0],
        "bc_loss": [0.0, 0.0],
        "pde_loss": [2.0, 1.0],
    }

    model._record_loss(
        {
            "total_loss": torch.tensor(0.5),
            "bc_loss": torch.tensor(0.0),
            "pde_loss": torch.tensor(0.5),
            "pde_loss_new": torch.tensor(0.25),
        }
    )

    history = model.loss_history["pde_loss_new"]
    assert len(history) == 3
    assert math.isnan(history[0]) and math.isnan(history[1])
    assert history[2] == pytest.approx(0.25)

    model.print_status()
    assert "pde_loss_new" not in capsys.readouterr().out


def test_navier_stokes_component_histories_are_exposed_by_evaluator():
    geometry = df.custom_data(
        {
            "x": torch.linspace(0.0, 1.0, 4),
            "y": torch.linspace(0.0, 1.0, 4),
        }
    )
    geometry.define_pde(df.NavierStokes(mu=1.0, rho=1.0))
    domain = df.domain(geometry)
    model = _model(output_vars=("u", "v", "p"))

    trained, _ = model.train_adam(
        learning_rate=0.001,
        epochs=2,
        calc_loss=df.calc_loss_simple(domain),
        print_every=10,
    )
    evaluator = geometry.evaluate(trained)

    component_keys = (
        "pde_loss_continuity",
        "pde_loss_x_momentum",
        "pde_loss_y_momentum",
    )
    for key in component_keys:
        assert key in evaluator.data_dict
        assert evaluator[key].shape == (2,)

    np.testing.assert_allclose(
        evaluator["pde_loss"],
        sum(evaluator[key] for key in component_keys),
        rtol=1e-5,
        atol=1e-7,
    )
