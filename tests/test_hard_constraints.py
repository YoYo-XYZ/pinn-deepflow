import pytest
import torch
import torch.nn as nn

import deepflow as df
from deepflow.physicsinformed import PhysicsAttach


class _ZeroModel(nn.Module):
    def forward(self, inputs):
        return {"u": torch.zeros_like(inputs["x"])}


def test_unconfigured_hard_condition_is_still_part_of_loss():
    attachment = PhysicsAttach()
    attachment.set_coordinates(torch.zeros(3), torch.zeros(3))
    attachment.define_bc({"u": df.hard_constraint(3.0)})
    attachment.process_coordinates()

    loss = attachment.calc_loss(_ZeroModel())

    assert loss.item() == pytest.approx(9.0)


def test_domain_loss_configures_hard_boundary_automatically():
    bound = df.line_horizontal(0.0, [0.0, 1.0])
    bound.define_bc({"u": df.hard_constraint(4.0)})
    domain = df.domain(bound)
    domain.sampling_uniform([4], [])

    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=4, length=1)
    losses = df.calc_loss_simple(domain)(model)

    output = model(bound.inputs_tensor_dict)["u"]
    assert torch.allclose(output, torch.full_like(output, 4.0))
    assert losses["bc_loss"].item() == pytest.approx(0.0)


def test_area_initial_hard_condition_uses_time_factor():
    area = df.rectangle([0.0, 1.0], [0.0, 1.0])
    area.define_ic({"u": df.hard_constraint(2.0)})
    area.define_time([0.0, 1.0])
    area.sampling_area([3, 3])
    area.process_coordinates()

    model = df.PINN(
        input_vars=["x", "y", "t"],
        output_vars=["u"],
        width=4,
        length=1,
    )
    model.apply_hard_constraints([area])

    output = model(area.inputs_tensor_dict)["u"]
    assert torch.allclose(output, torch.full_like(output, 2.0))


def test_parameterized_curve_hard_condition_fails_clearly():
    curve = df.curve(
        [0.0, 1.0],
        lambda t: t,
        lambda t: t.square(),
        ref_axis="t",
    )
    curve.define_bc({"u": df.hard_constraint(0.0)})
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=4, length=1)

    with pytest.raises(ValueError, match="parameterized curves"):
        model.apply_hard_constraints([curve])


def test_pde_area_without_conditions_is_ignored_by_hard_setup():
    area = df.rectangle([0.0, 1.0], [0.0, 1.0])
    area.define_pde(df.pde.BurgersEquation1D(nu=0.1))
    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=4, length=1)

    model.apply_hard_constraints([area])

    assert model.hard_constraints == {}
