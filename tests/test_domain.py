import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.fixture
def mixed_domain():
    boundary = df.geometry.line_vertical(1, [0, 1])
    boundary.define_bc({"u_x": 0, "v": 0})
    second = df.geometry.line_horizontal(1, [0, 1])
    second.define_bc({"u": df.func("x", lambda x: x**3)})
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.HeatEquation(0.2))
    initial = df.geometry.rectangle([0, 1], [0, 1])
    initial.define_ic({"u": 0})
    domain = df.domain(boundary, second, [area, initial])
    for geometry in domain:
        geometry.define_time([0, 1], sampling_scheme="uniform", expo_scaling=False)
    domain.sampling_uniform([4, 3], [[3, 2], [2, 2]])
    return domain


def test_domain_collects_area_boundaries_and_preserves_list_area_workflow():
    area = df.geometry.rectangle([0, 1], [0, 1])
    initial = df.geometry.rectangle([0, 1], [0, 1])
    point = df.geometry.point(0, 0)
    domain = df.domain(area, initial.area_list, point)
    assert domain.area_list == [area, initial]
    assert domain.bound_list == area.bound_list + [point]
    assert list(domain) == domain.bound_list + domain.area_list


@pytest.mark.parametrize("geometry", [None, "rectangle", [df.geometry.point(0, 0), 1]])
def test_domain_rejects_unsupported_inputs(geometry):
    with pytest.raises(TypeError):
        df.domain(geometry)


@pytest.mark.parametrize(
    "method,scheme",
    [
        ("sampling_uniform", "uniform"),
        ("sampling_random", "random"),
        ("sampling_lhs", "lhs"),
    ],
)
def test_domain_samplers_prepare_differentiable_model_inputs(method, scheme):
    area = df.geometry.rectangle([0, 1], [0, 1])
    domain = df.domain(area)
    getattr(domain, method)([3, 4, 5, 6], [[4, 3]])
    assert domain.sampling_option == scheme
    assert [len(b.X) for b in domain.bound_list] == [3, 4, 5, 6]
    for geometry in domain:
        inputs = geometry.inputs_tensor_dict
        torch.testing.assert_close(inputs["x"], geometry.X)
        torch.testing.assert_close(inputs["y"], geometry.Y)
        assert inputs["x"].is_leaf and inputs["x"].requires_grad


def test_batched_loss_matches_individual_losses_and_known_boundary_error(
    mixed_domain, polynomial_model
):
    loss = df.calc_loss_simple(mixed_domain)(polynomial_model)
    expected_bc = sum(g.calc_loss(polynomial_model) for g in mixed_domain.bound_list)
    expected_pde = mixed_domain.area_list[0].calc_loss(polynomial_model)
    expected_ic = mixed_domain.area_list[1].calc_loss(polynomial_model)
    for key, expected in {
        "bc_loss": expected_bc,
        "pde_loss": expected_pde,
        "ic_loss": expected_ic,
    }.items():
        torch.testing.assert_close(loss[key], expected)
    first, second = mixed_domain.bound_list
    known_bc = (9 + first.Y**4).mean() + ((1 + second.T**3) ** 2).mean()
    torch.testing.assert_close(loss["bc_loss"], known_bc)
    torch.testing.assert_close(
        loss["total_loss"], expected_bc + expected_pde + expected_ic
    )


def test_weighted_loss_changes_total_without_changing_components(
    mixed_domain, polynomial_model
):
    simple = df.calc_loss_simple(mixed_domain)(polynomial_model)
    weighted = df.calc_loss_weighted(
        mixed_domain, bc_weights=2, ic_weights=3, pde_weights=0.5
    )(polynomial_model)
    for key in ["bc_loss", "ic_loss", "pde_loss"]:
        torch.testing.assert_close(weighted[key], simple[key])
    torch.testing.assert_close(
        weighted["total_loss"],
        2 * simple["bc_loss"] + 3 * simple["ic_loss"] + 0.5 * simple["pde_loss"],
    )


def test_batched_loss_gradients_match_individual_geometry_losses(mixed_domain):
    model = df.PINN(input_vars=["x", "y", "t"], width=5, length=2)
    batched_loss = df.calc_loss_simple(mixed_domain)(model)["total_loss"]
    batched_grads = torch.autograd.grad(batched_loss, tuple(model.parameters()))
    individual_loss = sum(geometry.calc_loss(model) for geometry in mixed_domain)
    individual_grads = torch.autograd.grad(individual_loss, tuple(model.parameters()))
    for actual, expected in zip(batched_grads, individual_grads):
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-10)


def test_custom_data_is_processed_and_can_supply_supervised_values(polynomial_model):
    x, y = torch.tensor([0.1, 0.4, 0.9]), torch.tensor([0.3, 0.2, 0.5])
    values = x**3 + y**3
    data = df.geometry.custom_data({"x": x, "y": y, "measured_u": values})
    data.define_bc({"u": df.func("x", lambda _: values)})
    domain = df.domain(data)
    assert len(domain.bound_list) == 0 and domain.area_list == [data]
    assert data.data_dict["measured_u"] is values
    assert df.calc_loss_simple(domain)(polynomial_model)["total_loss"].item() == 0


@pytest.mark.parametrize("budget", [2, 5, 8])
def test_r3_keeps_largest_errors_within_fixed_budget(budget, polynomial_model):
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.define_bc({"u": 0})
    domain = df.domain(boundary)
    domain.sampling_uniform([7])
    boundary.calc_residual_field(polynomial_model)
    domain.sampling_R3(bound_sampling_res=[budget])
    assert len(boundary.X) == budget
    assert torch.any(boundary.X == 1)
    if budget == 2:
        torch.testing.assert_close(boundary.X, torch.tensor([1.0, 5 / 6]))
    assert boundary.inputs_tensor_dict["x"].shape == (budget,)


def test_transient_r3_refreshes_time_at_full_budget(polynomial_model):
    area = df.geometry.rectangle([0, 1], [0, 1])
    area.define_pde(df.HeatEquation(0.1))
    area.define_time([2, 5], sampling_scheme="uniform", expo_scaling=False)
    domain = df.domain([area])
    domain.sampling_lhs(area_sampling_res=[12])
    for _ in range(2):
        area.calc_residual_field(polynomial_model)
        domain.sampling_R3(area_sampling_res=[12])
        torch.testing.assert_close(
            area.inputs_tensor_dict["t"], torch.linspace(2, 5, 12)
        )


def test_domain_can_save_and_restore_coordinate_values(mixed_domain):
    original = [(g.X.clone(), g.Y.clone()) for g in mixed_domain]
    mixed_domain.save_coordinates()
    mixed_domain.sampling_lhs([2, 2], [3, 3])
    mixed_domain.load_coordinates()
    for geometry, (x, y) in zip(mixed_domain, original):
        torch.testing.assert_close(geometry.X, x)
        torch.testing.assert_close(geometry.Y, y)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="RAR saves original coordinates but appends selected points to candidates, losing the original samples",
)
def test_rar_retains_original_points_and_appends_only_requested_count(polynomial_model):
    boundary = df.geometry.line_horizontal(0, [0, 1])
    boundary.define_bc({"u": 0})
    domain = df.domain(boundary)
    domain.sampling_uniform([5])
    original = boundary.X.clone()
    boundary.calc_residual_field(polynomial_model)
    domain.sampling_RAR(bound_top_k_list=[2], bound_candidates_num_list=[8])
    assert len(boundary.X) == 7
    assert all(torch.any(boundary.X == x) for x in original)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="load_coordinates restores X/Y but leaves processed model inputs at the later samples",
)
def test_restored_coordinates_are_used_by_loss(mixed_domain, polynomial_model):
    loss = df.calc_loss_simple(mixed_domain)
    expected = loss(polynomial_model)["total_loss"].detach()
    mixed_domain.save_coordinates()
    mixed_domain.sampling_lhs([2, 2], [3, 3])
    mixed_domain.load_coordinates()
    torch.testing.assert_close(loss(polynomial_model)["total_loss"], expected)


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="R3 draws only the remaining candidate count, so rejection by holes shrinks the promised fixed budget",
)
def test_r3_preserves_point_budget_in_domains_with_holes(polynomial_model):
    area = df.geometry.rectangle([-1, 1], [-1, 1]) - df.geometry.circle(0, 0, 0.9)
    area.define_pde(df.CustomPDE(lambda values: (values["u"],)))
    domain = df.domain([area])
    domain.sampling_lhs(area_sampling_res=[100])
    area.calc_residual_field(polynomial_model)
    domain.sampling_R3(area_sampling_res=[40])
    assert len(area.X) == 40
