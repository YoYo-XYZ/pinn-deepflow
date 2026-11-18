import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


def test_boundary_loss_combines_function_values_and_derivatives(polynomial_model):
    boundary = df.geometry.line_vertical(1, [0, 1])
    boundary.define_bc({"u": df.function("y", lambda y: y**3), "u_x": 2, "v": 0})
    boundary.sampling_line(5)
    inputs = boundary.process_coordinates()
    residual = boundary.calc_residual_field(polynomial_model)
    torch.testing.assert_close(residual, 2 + inputs["y"] ** 2)
    torch.testing.assert_close(
        boundary.calc_loss(polynomial_model), (2 + inputs["y"] ** 4).mean()
    )
    assert inputs["x"].requires_grad and inputs["y"].requires_grad
    assert not boundary.target_output_tensor_dict["u"].requires_grad


def test_mixed_hard_and_soft_conditions_only_penalize_soft_values(polynomial_model):
    boundary = df.geometry.line_vertical(1, [0, 1])
    boundary.define_bc({"u": df.hard_constraint(2), "v": 1})
    boundary.sampling_line(5)
    boundary.process_coordinates()
    torch.testing.assert_close(
        boundary.calc_loss(polynomial_model), ((boundary.Y**2 - 1) ** 2).mean()
    )


def test_all_hard_conditions_have_zero_loss(polynomial_model):
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.define_bc({"u": df.hard_constraint(), "v": df.hard_constraint()})
    boundary.sampling_line(4)
    boundary.process_coordinates()
    torch.testing.assert_close(
        boundary.calc_residual_field(polynomial_model), torch.zeros(4)
    )
    assert boundary.calc_loss(polynomial_model).item() == 0


@pytest.mark.parametrize("condition", [None, ["x"], ["x", 2]])
def test_malformed_condition_fails_during_coordinate_processing(condition):
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.define_bc({"u": condition})
    with pytest.raises(ValueError, match="Invalid condition format.*u"):
        boundary.process_coordinates()


def test_derivative_condition_reports_missing_model_output(polynomial_model):
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.define_bc({"temperature_x": 0})
    boundary.process_coordinates()
    with pytest.raises(KeyError, match="Model output missing variable 'temperature'"):
        boundary.calc_loss(polynomial_model)


def test_coordinate_and_time_processing_requires_samples_and_range():
    area = df.geometry.rectangle([0, 1], [0, 1])
    with pytest.raises(ValueError, match="Coordinates X and Y"):
        area.process_coordinates()
    with pytest.raises(ValueError, match="X coordinate"):
        area.sampling_time()
    with pytest.raises(ValueError, match="Time range"):
        area.define_time()
    area.sampling_area(3)
    with pytest.raises(ValueError, match="Time range"):
        area.sampling_time()


@pytest.mark.parametrize("physics", ["BC", "IC", "PDE"])
@pytest.mark.parametrize("scheme", ["uniform", "random"])
def test_time_sampling_preserves_initial_slice_and_replays_seed(physics, scheme):
    area = df.geometry.rectangle([0, 1], [0, 1])
    if physics == "BC":
        area.define_bc({"u": df.func("t", lambda t: t**2)})
    elif physics == "IC":
        area.define_ic({"u": 0})
    else:
        area.define_pde(df.HeatEquation(0.1))
    area.sampling_area([4, 3])
    area.define_time([2, 5], sampling_scheme=scheme, expo_scaling=False)
    df.manual_seed(13)
    first = area.process_coordinates()["t"].detach().clone()
    assert first.shape == (12,)
    assert ((first >= 2) & (first <= 5)).all()
    if physics == "IC":
        torch.testing.assert_close(first, torch.full((12,), 2.0))
    elif scheme == "uniform":
        torch.testing.assert_close(first, torch.linspace(2, 5, 12))
    df.manual_seed(13)
    torch.testing.assert_close(first, area.process_coordinates()["t"])
    if physics == "BC":
        torch.testing.assert_close(area.target_output_tensor_dict["u"], first**2)


def test_fixed_time_including_zero_updates_model_inputs():
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.sampling_line(4)
    for time in [0.5, 0]:
        boundary.define_time(time, sampling_scheme="uniform", expo_scaling=False)
        inputs = boundary.process_coordinates()
        torch.testing.assert_close(inputs["t"], torch.full((4,), float(time)))


def test_parabolic_condition_has_specified_peak_and_wall_values():
    key, profile = df.parabolic("y", width=2, max_val=3, center_distance=4)
    assert key == "y"
    torch.testing.assert_close(
        profile(torch.tensor([3.0, 4.0, 5.0])), torch.tensor([0.0, 3.0, 0.0])
    )


def test_residual_point_selection_uses_absolute_error_and_clears_buffers(
    polynomial_model,
):
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.define_bc({"u": 0})
    boundary.sampling_line(5)
    boundary.process_coordinates()
    boundary.calc_residual_field(polynomial_model)
    x, y = boundary.get_residual_based_points_topk(2)
    torch.testing.assert_close(x, torch.tensor([1.0, 0.75]))
    torch.testing.assert_close(y, torch.zeros(2))
    boundary.apply_residual_based_points()
    assert len(boundary.X) == 7
    boundary.clear_residual_based_points()
    boundary.apply_residual_based_points()
    assert len(boundary.X) == 7
    boundary.sampling_line(5)
    boundary.process_coordinates()
    boundary.calc_residual_field(polynomial_model)
    x, _ = boundary.get_residual_based_points_threshold(threshold=0.2)
    torch.testing.assert_close(x, torch.tensor([0.75, 1.0]))


@pytest.mark.xfail(
    strict=True,
    raises=TypeError,
    reason="IC time sampling subscripts range_t even when a scalar fixed time was supplied",
)
def test_initial_condition_can_use_a_fixed_time():
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_ic({"u": 0})
    area.define_time(0, sampling_scheme="uniform", expo_scaling=False)
    area.sampling_area(3)
    torch.testing.assert_close(area.process_coordinates()["t"], torch.zeros(9))
