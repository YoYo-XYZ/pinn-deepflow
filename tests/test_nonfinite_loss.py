import pytest
import torch

import deepflow as df


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_adam_stops_on_nonfinite_loss(value):
    model = df.PINN(width=2, length=1, input_vars=["x"], output_vars=["u"])
    callbacks = []

    def calc_loss(current_model):
        parameter = next(current_model.parameters())
        loss = parameter.sum() * 0 + parameter.new_tensor(value)
        return {"total_loss": loss}

    trained_model, _ = model.train_adam(
        learning_rate=0.01,
        epochs=3,
        calc_loss=calc_loss,
        do_between_epochs=lambda epoch, current_model: callbacks.append(epoch),
    )

    assert trained_model.loss_history["total_loss"] == []
    assert callbacks == []


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_lbfgs_stops_and_restores_model_on_nonfinite_loss(value):
    model = df.PINN(width=2, length=1, input_vars=["x"], output_vars=["u"])
    initial_state = {
        key: state.detach().clone() for key, state in model.state_dict().items()
    }
    calls = 0
    callbacks = []

    def calc_loss(current_model):
        nonlocal calls
        calls += 1
        parameter = next(current_model.parameters())
        if calls == 1:
            loss = parameter.square().sum()
        else:
            loss = parameter.sum() * 0 + parameter.new_tensor(value)
        return {"total_loss": loss}

    trained_model, _ = model.train_lbfgs(
        epochs=3,
        calc_loss=calc_loss,
        do_between_epochs=lambda epoch, current_model: callbacks.append(epoch),
    )

    assert calls >= 2
    assert trained_model.loss_history["total_loss"] == []
    assert callbacks == []
    for key, state in trained_model.state_dict().items():
        assert torch.equal(state.cpu(), initial_state[key].cpu())
