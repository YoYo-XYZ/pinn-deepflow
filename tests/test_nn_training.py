import torch

import deepflow as df
import deepflow.utility as deepflow_utility


def _quadratic_loss(model):
    parameter = next(model.parameters())
    total_loss = parameter.square().sum()
    return {
        "total_loss": total_loss,
        "bc_loss": total_loss,
        "pde_loss": total_loss,
    }


def test_adam_scheduler_supports_fewer_than_twenty_epochs():
    original_device = deepflow_utility.device
    deepflow_utility.device = "cpu"
    try:
        model = df.PINN(width=2, length=1, input_vars=["x"], output_vars=["u"])
        trained_model, _ = model.train_adam(
            learning_rate=0.01,
            epochs=2,
            calc_loss=_quadratic_loss,
            use_scheduler=True,
            print_every=10,
        )
    finally:
        deepflow_utility.device = original_device

    assert len(trained_model.loss_history["total_loss"]) == 2


def test_adam_best_model_matches_recorded_best_loss():
    original_device = deepflow_utility.device
    deepflow_utility.device = "cpu"
    try:
        model = df.PINN(width=2, length=1, input_vars=["x"], output_vars=["u"])
        _, best_model = model.train_adam(
            learning_rate=0.01,
            epochs=2,
            calc_loss=_quadratic_loss,
            print_every=10,
        )
    finally:
        deepflow_utility.device = original_device

    actual_loss = _quadratic_loss(best_model)["total_loss"].item()
    recorded_best_loss = min(best_model.loss_history["total_loss"])
    assert actual_loss == recorded_best_loss


def test_lbfgs_best_model_matches_recorded_best_loss():
    original_device = deepflow_utility.device
    deepflow_utility.device = "cpu"
    try:
        model = df.PINN(width=2, length=1, input_vars=["x"], output_vars=["u"])
        _, best_model = model.train_lbfgs(
            epochs=2,
            calc_loss=_quadratic_loss,
            print_every=10,
        )
    finally:
        deepflow_utility.device = original_device

    actual_loss = _quadratic_loss(best_model)["total_loss"].item()
    recorded_best_loss = min(best_model.loss_history["total_loss"])
    assert actual_loss == recorded_best_loss


def test_training_does_not_move_original_model():
    if not torch.cuda.is_available():
        return

    original_device = deepflow_utility.device
    deepflow_utility.device = "cuda"
    try:
        model = df.PINN(width=2, length=1, input_vars=["x"], output_vars=["u"])
        assert next(model.parameters()).device.type == "cpu"

        trained_model, _ = model.train_adam(
            learning_rate=0.01,
            epochs=1,
            calc_loss=_quadratic_loss,
            print_every=10,
        )
    finally:
        deepflow_utility.device = original_device

    assert next(model.parameters()).device.type == "cpu"
    assert next(trained_model.parameters()).device.type == "cuda"
