import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.fixture
def regression_problem():
    x = torch.linspace(-1, 1, 9)
    data = df.geometry.custom_data({"x": x, "y": torch.zeros_like(x)})
    data.define_bc({"u": df.func("x", lambda x: 2 * x - 0.5)})
    domain = df.domain(data)
    model = df.FNN(output_vars=["u"], hidden_layer=[])
    return model, domain, df.calc_loss_simple(domain)


@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_training_reduces_loss_preserves_source_and_returns_best_model(
    method, regression_problem
):
    model, domain, loss = regression_problem
    inputs = domain.area_list[0].inputs_tensor_dict
    original = model(inputs)["u"].detach().clone()
    initial_loss = loss(model)["total_loss"].item()
    observed = []

    def callback(epoch, current):
        observed.append(
            (
                epoch,
                loss(current)["total_loss"].item(),
                current(inputs)["u"].detach().clone(),
            )
        )

    if method == "adam":
        trained, best = model.train_adam(
            learning_rate=0.1,
            epochs=40,
            calc_loss=loss,
            do_between_epochs=callback,
            max_grad_norm=None,
        )
    else:
        trained, best = model.train_lbfgs(
            epochs=2, calc_loss=loss, do_between_epochs=callback
        )
    history = trained.loss_history["total_loss"]
    assert len(history) == len(observed) == (40 if method == "adam" else 2)
    assert [e for e, _, _ in observed] == list(range(1, len(history) + 1))
    assert all(isinstance(value, float) for value in history)
    assert min(history) < initial_loss * 0.02
    assert loss(best)["total_loss"].item() == pytest.approx(min(history))
    torch.testing.assert_close(
        best(inputs)["u"], observed[history.index(min(history))][2]
    )
    torch.testing.assert_close(model(inputs)["u"], original)
    assert model.loss_history["total_loss"] == []
    assert trained is not model and best is not trained


@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_training_threshold_stops_early_and_history_continues(
    method, regression_problem
):
    model, _, loss = regression_problem
    options = {"calc_loss": loss, "epochs": 4, "threshold_loss": 1000}
    if method == "adam":
        options["learning_rate"] = 0.01
    train = getattr(model, "train_" + method)
    trained, _ = train(**options)
    assert len(trained.loss_history["total_loss"]) == 1
    continued, _ = getattr(trained, "train_" + method)(**options)
    assert len(continued.loss_history["total_loss"]) == 2
    assert len(trained.loss_history["total_loss"]) == 1


@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_zero_epochs_returns_unchanged_copies(method, regression_problem):
    model, domain, loss = regression_problem
    options = {"epochs": 0, "calc_loss": loss}
    if method == "adam":
        options["learning_rate"] = 0.01
    trained, best = getattr(model, "train_" + method)(**options)
    inputs = domain.area_list[0].inputs_tensor_dict
    torch.testing.assert_close(trained(inputs)["u"], model(inputs)["u"])
    torch.testing.assert_close(best(inputs)["u"], model(inputs)["u"])
    assert trained.loss_history["total_loss"] == []


@pytest.mark.parametrize("method", ["adam", "lbfgs"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_nonfinite_initial_loss_stops_without_changing_model(
    method, nonfinite, regression_problem
):
    model, domain, _ = regression_problem
    inputs = domain.area_list[0].inputs_tensor_dict

    def loss(current):
        return {"total_loss": current(inputs)["u"].sum() * nonfinite}

    options = {"epochs": 3, "calc_loss": loss}
    if method == "adam":
        options["learning_rate"] = 0.01
    trained, best = getattr(model, "train_" + method)(**options)
    assert trained.loss_history["total_loss"] == []
    for current in [trained, best]:
        torch.testing.assert_close(current(inputs)["u"], model(inputs)["u"])


def test_lbfgs_rolls_back_parameters_when_line_search_encounters_nonfinite_loss(
    regression_problem,
):
    model, domain, real_loss = regression_problem
    calls = 0

    def loss(current):
        nonlocal calls
        calls += 1
        values = real_loss(current)
        if calls > 1:
            values["total_loss"] = values["total_loss"] * float("nan")
        return values

    trained, best = model.train_lbfgs(epochs=2, calc_loss=loss)
    assert calls > 1
    inputs = domain.area_list[0].inputs_tensor_dict
    torch.testing.assert_close(trained(inputs)["u"], model(inputs)["u"])
    torch.testing.assert_close(best(inputs)["u"], model(inputs)["u"])
    assert trained.loss_history["total_loss"] == []


def test_adam_scheduler_and_clipping_produce_finite_training(regression_problem):
    model, _, loss = regression_problem
    trained, _ = model.train_adam(
        learning_rate=0.05,
        epochs=3,
        calc_loss=loss,
        use_scheduler=True,
        max_grad_norm=0.1,
    )
    assert len(trained.loss_history["total_loss"]) == 3
    assert all(torch.isfinite(p).all() for p in trained.parameters())


@pytest.mark.parametrize("max_grad_norm", [0, -1])
def test_adam_requires_positive_gradient_clip_limit(max_grad_norm, regression_problem):
    model, _, loss = regression_problem
    with pytest.raises(ValueError, match="max_grad_norm"):
        model.train_adam(
            learning_rate=0.01, epochs=1, calc_loss=loss, max_grad_norm=max_grad_norm
        )


@pytest.mark.parametrize("method", ["adam", "lbfgs"])
def test_training_requires_positive_print_interval(method, regression_problem):
    model, _, loss = regression_problem
    options = {"epochs": 1, "calc_loss": loss, "print_every": 0}
    if method == "adam":
        options["learning_rate"] = 0.01
    with pytest.raises(ValueError, match="print_every"):
        getattr(model, "train_" + method)(**options)
