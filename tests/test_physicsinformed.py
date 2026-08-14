import inspect

import torch
import torch.nn as nn

from deepflow.physicsinformed import PhysicsAttach


class _TwoOutputModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.u_scale = nn.Parameter(torch.tensor(2.0))
        self.v_scale = nn.Parameter(torch.tensor(-3.0))
        self.forward_calls = 0

    def forward(self, inputs):
        self.forward_calls += 1
        x = inputs["x"]
        return {
            "u": self.u_scale * x + 1.0,
            "v": self.v_scale * x - 2.0,
        }


def _boundary_attachment():
    attachment = PhysicsAttach()
    attachment.set_coordinates(
        torch.tensor([0.0, 1.0, 2.0]),
        torch.zeros(3),
    )
    attachment.define_bc({"u": 0.0, "v": 0.0})
    attachment.process_coordinates(device=torch.device("cpu"))
    return attachment


def test_physics_attach_methods_have_no_unused_arguments():
    calc_output_parameters = inspect.signature(PhysicsAttach.calc_output).parameters
    calc_loss_parameters = inspect.signature(PhysicsAttach.calc_loss).parameters

    assert "model" not in calc_output_parameters
    assert "loss_fn" not in calc_loss_parameters


def test_calc_output_uses_cached_model_outputs():
    attachment = _boundary_attachment()
    model = _TwoOutputModel()

    cached_outputs = attachment.process_model(model)
    processed_outputs = attachment.calc_output()

    assert model.forward_calls == 1
    assert processed_outputs.keys() == cached_outputs.keys()
    for key in cached_outputs:
        assert torch.equal(processed_outputs[key], cached_outputs[key])


def test_calc_loss_preserves_per_point_sum_of_squares_and_gradients():
    attachment = _boundary_attachment()
    model = _TwoOutputModel()

    loss = attachment.calc_loss(model)
    expected_raw = torch.stack(
        (
            model.u_scale.detach() * attachment.X_ + 1.0,
            model.v_scale.detach() * attachment.X_ - 2.0,
        )
    )
    expected_loss = torch.mean(expected_raw.square().sum(dim=0))

    assert torch.allclose(loss.detach(), expected_loss)

    loss.backward()
    assert model.u_scale.grad is not None
    assert model.v_scale.grad is not None
