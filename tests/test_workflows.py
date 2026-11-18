import numpy as np
import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


def test_heat_workflow_trains_resamples_saves_and_evaluates(tmp_path):
    area = df.geometry.rectangle([0, 1], [0, 1])
    initial = df.geometry.rectangle([0, 1], [0, 1])
    domain = df.domain(area, initial.area_list)
    for i, boundary in enumerate(domain.bound_list):
        boundary.define_bc({"u": 1 if i == 3 else 0})
    area.define_pde(df.HeatEquation(0.1))
    initial.define_ic({"u": 0})
    for geometry in domain:
        geometry.define_time([0, 1], sampling_scheme="random", expo_scaling=False)
    domain.sampling_lhs([5] * 4, [12, 8])
    model = df.PINN(input_vars=["x", "y", "t"], output_vars=["u"], width=6, length=2)
    loss = df.calc_loss_simple(domain)
    starting_loss = loss(model)["total_loss"].item()
    trained, best = model.train_adam(learning_rate=0.01, epochs=3, calc_loss=loss)
    assert min(trained.loss_history["total_loss"]) < starting_loss
    assert all(
        len(trained.loss_history[key]) == 3
        for key in ["bc_loss", "ic_loss", "pde_loss"]
    )
    loss(best)
    domain.sampling_R3(bound_sampling_res=[5] * 4, area_sampling_res=[12, 8])
    torch.testing.assert_close(initial.inputs_tensor_dict["t"], torch.zeros(8))
    trained, best = best.train_lbfgs(epochs=1, calc_loss=loss)
    assert np.isfinite(trained.loss_history["total_loss"]).all()
    path = str(tmp_path / "heat")
    best.save_as_pickle(path)
    restored = df.load_from_pickle(path)
    evaluation = area.evaluate(restored)
    evaluation.sampling_area([4, 3])
    evaluation.define_time(0.5)
    assert evaluation["u"].shape == evaluation["pde_residual"].shape == (12,)
    assert (
        np.isfinite(evaluation["u"]).all()
        and np.isfinite(evaluation["pde_residual"]).all()
    )
    np.testing.assert_array_equal(evaluation["t"], np.full(12, 0.5))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cylinder_workflow_computes_gradients_and_evaluates_outlet(dtype):
    df.dtype = dtype
    cylinder = df.geometry.circle(0.2, 0.2, 0.05)
    area = df.geometry.rectangle([0, 1.1], [0, 0.41]) - cylinder
    domain = df.domain(area, cylinder.bound_list)
    for i, boundary in enumerate(domain.bound_list):
        if i == 0:
            boundary.define_bc({"u": df.parabolic("y", 0.41, 1, 0.205), "v": 0})
        elif i == 2:
            boundary.define_bc({"p": 0})
        else:
            boundary.define_bc({"u": 0, "v": 0})
    area.define_pde(df.NavierStokes(U=1, L=1, mu=0.02, rho=1))
    domain.sampling_lhs([5] * 6, [24])
    assert ((area.X - 0.2) ** 2 + (area.Y - 0.2) ** 2 > 0.05**2).all()
    model = df.PINN(width=6, length=2)
    loss = df.calc_loss_simple(domain)
    values = loss(model)
    values["total_loss"].backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    trained, _ = model.train_adam(learning_rate=0.01, epochs=2, calc_loss=loss)
    outlet = domain.bound_list[2].evaluate(trained)
    outlet.sampling_line(9)
    np.testing.assert_allclose(outlet["x"], np.full(9, 1.1), atol=1e-7, rtol=0)
    assert all(
        np.isfinite(outlet[field]).all() for field in ["u", "v", "p", "bc_residual"]
    )


def test_burgers_workflow_with_hard_walls_trains_through_pde():
    area = df.geometry.rectangle([-1, 1], [0, 1])
    domain = df.domain(area)
    for i, boundary in enumerate(domain.bound_list):
        if i in {0, 2}:
            boundary.define_bc({"u": df.hard_constraint(0)})
        else:
            boundary.define_bc({"u": df.func("x", lambda x: -torch.sin(torch.pi * x))})
    area.define_pde(df.BurgersEquation1D(0.01 / np.pi))
    domain.sampling_uniform([5] * 4, [[4, 3]])
    model = df.PINN(output_vars=["u"], width=6, length=2)
    model.apply_hard_constraints(domain.bound_list)
    trained, _ = model.train_adam(
        learning_rate=0.01, epochs=2, calc_loss=df.calc_loss_simple(domain)
    )
    for boundary in [domain.bound_list[0], domain.bound_list[2]]:
        np.testing.assert_allclose(boundary.evaluate(trained)["u"], 0, atol=1e-14)
    assert np.isfinite(trained.loss_history["pde_loss"]).all()
