import numpy as np
import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


def test_area_evaluation_refreshes_predictions_residuals_and_custom_fields(
    polynomial_model,
):
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.BurgersEquation1D(0.2))
    evaluation = area.evaluate(polynomial_model)
    evaluation.sampling_area([4, 3])
    x, y = evaluation["x"], evaluation["y"]
    u = x**3 + y**3
    np.testing.assert_allclose(evaluation["u"], u)
    np.testing.assert_allclose(
        evaluation["pde_residual"], np.abs(3 * y**2 + u * 3 * x**2 - 1.2 * x)
    )
    assert all(isinstance(value, np.ndarray) for value in evaluation.data_dict.values())
    evaluation["speed"] = np.hypot(evaluation["u"], evaluation["v"])
    assert "speed" in str(evaluation)
    evaluation.sampling_area([2, 3])
    assert evaluation["u"].shape == (6,)
    assert "speed" not in evaluation.data_dict
    with pytest.raises(KeyError):
        evaluation["missing"]


def test_boundary_evaluation_resamples_fields_and_loss_history():
    boundary = df.geometry.line_vertical(1, [0, 1])
    boundary.sampling_line(4)
    boundary.define_bc({"u": 0})
    model = df.PINN(output_vars=["u"], width=4, length=1)
    model.loss_history["total_loss"] = [2.0, 1.0]
    evaluation = boundary.evaluate(model)
    np.testing.assert_array_equal(evaluation["total_loss"], [2, 1])
    np.testing.assert_allclose(evaluation["bc_residual"], np.abs(evaluation["u"]))
    assert not model.training
    evaluation.sampling_line(7)
    assert evaluation["u"].shape == (7,)
    np.testing.assert_array_equal(evaluation["x"], np.ones(7))


def test_custom_data_can_be_evaluated_without_a_training_domain(polynomial_model):
    x, y = torch.tensor([0.2, 0.7]), torch.tensor([0.3, 0.8])
    data = df.geometry.custom_data({"x": x, "y": y, "label": "measurements"})
    evaluation = data.evaluate(polynomial_model)
    np.testing.assert_allclose(evaluation["u"], (x**3 + y**3).numpy())
    assert data.data_dict["label"] == "measurements"
    np.testing.assert_array_equal(evaluation["x"], x.numpy())


def test_transient_evaluation_updates_predictions_at_fixed_time(polynomial_model):
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.HeatEquation(0.1))
    area.define_time([0, 1], sampling_scheme="uniform", expo_scaling=False)
    area.sampling_area([3, 2])
    evaluation = area.evaluate(polynomial_model)
    evaluation.define_time(0.5)
    np.testing.assert_allclose(evaluation["t"], np.full(6, 0.5))
    np.testing.assert_allclose(
        evaluation["u"], evaluation["x"] ** 3 + evaluation["y"] ** 3 + 0.125
    )
    np.testing.assert_allclose(
        evaluation["pde_residual"],
        np.abs(0.75 - 0.6 * (evaluation["x"] + evaluation["y"])),
    )


def test_navier_stokes_evaluation_exports_derived_physical_fields(polynomial_model):
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.NavierStokes(1, 1))
    area.sampling_area([3, 2])
    evaluation = area.evaluate(polynomial_model)
    x, y = evaluation["x"], evaluation["y"]
    np.testing.assert_allclose(evaluation["u_x"], 3 * x**2)
    np.testing.assert_allclose(evaluation["v_y"], 2 * x**2 * y)
    np.testing.assert_allclose(evaluation["p_x"], 2 * x)
    np.testing.assert_allclose(
        evaluation["continuity_residual"], 3 * x**2 + 2 * x**2 * y
    )


@pytest.mark.xfail(
    strict=True,
    raises=KeyError,
    reason="Evaluator checks range_t truthiness and omits the valid fixed time t=0",
)
def test_evaluation_exports_zero_time(polynomial_model):
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.sampling_line(4)
    boundary.define_time(0, sampling_scheme="uniform", expo_scaling=False)
    evaluation = boundary.evaluate(polynomial_model)
    np.testing.assert_array_equal(evaluation["t"], np.zeros(4))
