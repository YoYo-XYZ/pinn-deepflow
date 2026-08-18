"""End-to-end reproducibility checks for DeepFlow training."""

import copy

import pytest
import torch

import deepflow as df
from torch import pi, sin


N_ADAM = 500
N_LBFGS = 100
N_IC = 2000
N_BC = 1000
N_PDE = 4000


def _tensors_equal(first, second):
    if isinstance(first, list):
        return all(torch.equal(left, right) for left, right in zip(first, second))
    return torch.equal(first, second)


def _samples_equal(first, second):
    for key in first:
        first_x, first_y = first[key]
        second_x, second_y = second[key]
        if not (torch.equal(first_x, second_x) and torch.equal(first_y, second_y)):
            return False
    return True


def _loss_dicts_equal(first, second):
    for key in first:
        if key not in second or len(first[key]) != len(second[key]):
            return False
        if any(left != right for left, right in zip(first[key], second[key])):
            return False
    return True


def _create_burgers(seed=69, deterministic=False, device="cpu"):
    """Create a seeded Burgers domain and model."""
    import deepflow.utility as utility

    utility.device = device
    df.manual_seed(seed, deterministic=deterministic)

    area = df.geometry.rectangle([-1, 1], [0, 1])
    line_ic = df.geometry.line_horizontal(y=0, range_x=[-1, 1])
    line_bc1 = df.geometry.line_vertical(x=-1, range_y=[0, 1])
    line_bc2 = df.geometry.line_vertical(x=1, range_y=[0, 1])
    domain = df.domain(area.area_list, line_ic, line_bc1, line_bc2)

    domain.area_list[0].define_pde(df.pde.BurgersEquation1D(nu=0.01 / pi))
    domain.bound_list[0].define_bc({"u": ["x", lambda x: -sin(pi * x)]})
    domain.bound_list[1].define_bc({"u": 0})
    domain.bound_list[2].define_bc({"u": 0})

    model = df.PINN(input_vars=["x", "y"], output_vars=["u"], width=16, length=4)
    return domain, model


def _capture_samples(domain):
    samples = {}
    for index, geometry in enumerate(domain.bound_list):
        samples[f"bound_{index}"] = (geometry.X.clone(), geometry.Y.clone())
    for index, geometry in enumerate(domain.area_list):
        samples[f"area_{index}"] = (geometry.X.clone(), geometry.Y.clone())
    return samples


def _train_burgers(domain, model):
    """Run the reproducibility training workload and capture diagnostics."""
    calc_loss = df.calc_loss_weighted(domain, bc_weights=1)

    def resample(epoch, _model):
        if epoch % 500 == 0:
            domain.sampling_R3([N_IC, N_BC, N_BC], [N_PDE])

    domain.sampling_lhs([N_IC, N_BC, N_BC], [N_PDE])
    initial_samples = _capture_samples(domain)
    initial_weights = [parameter.clone() for parameter in model.parameters()]

    model, _ = model.train_adam(
        calc_loss=calc_loss,
        learning_rate=0.004,
        epochs=N_ADAM,
        do_between_epochs=resample,
        print_every=N_ADAM,
    )
    loss_history_adam = copy.deepcopy(model.loss_history)

    model, _ = model.train_lbfgs(
        calc_loss=calc_loss,
        epochs=N_LBFGS,
        print_every=N_LBFGS,
    )

    return {
        "model": model,
        "initial_samples": initial_samples,
        "initial_weights": initial_weights,
        "loss_history_adam": loss_history_adam,
        "loss_history": copy.deepcopy(model.loss_history),
        "final_weights": [parameter.clone() for parameter in model.parameters()],
    }


def _evaluate_on_grid(model):
    """Evaluate a model on a fixed grid and return a CPU tensor."""
    device = next(model.parameters()).device
    x = torch.linspace(-1, 1, 101)
    y = torch.linspace(0, 1, 101)
    grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")
    inputs = {"x": grid_x.reshape(-1).to(device), "y": grid_y.reshape(-1).to(device)}
    model.eval()
    with torch.no_grad():
        return model(inputs)["u"].cpu()


@pytest.mark.slow
def test_reproducibility():
    """Verify seeded CPU training is bit-exact and distinct seeds diverge."""
    df.manual_seed(42)
    lhs_first = df.latin_hypercube_sampling(100, 2, [0.0, 0.0], [1.0, 1.0])
    df.manual_seed(42)
    lhs_second = df.latin_hypercube_sampling(100, 2, [0.0, 0.0], [1.0, 1.0])
    df.manual_seed(99)
    lhs_other = df.latin_hypercube_sampling(100, 2, [0.0, 0.0], [1.0, 1.0])
    assert torch.equal(lhs_first, lhs_second)
    assert not torch.equal(lhs_first, lhs_other)

    first_domain, first_model = _create_burgers(69)
    first = _train_burgers(first_domain, first_model)
    second_domain, second_model = _create_burgers(69)
    second = _train_burgers(second_domain, second_model)

    assert _samples_equal(first["initial_samples"], second["initial_samples"])
    assert _tensors_equal(first["initial_weights"], second["initial_weights"])
    assert _loss_dicts_equal(first["loss_history_adam"], second["loss_history_adam"])
    assert _loss_dicts_equal(first["loss_history"], second["loss_history"])
    assert _tensors_equal(first["final_weights"], second["final_weights"])

    first_prediction = _evaluate_on_grid(first["model"])
    second_prediction = _evaluate_on_grid(second["model"])
    assert torch.equal(first_prediction, second_prediction)

    deterministic_domain, deterministic_model = _create_burgers(69, deterministic=True)
    deterministic = _train_burgers(deterministic_domain, deterministic_model)
    assert _samples_equal(first["initial_samples"], deterministic["initial_samples"])
    assert _tensors_equal(first["initial_weights"], deterministic["initial_weights"])
    assert _loss_dicts_equal(first["loss_history"], deterministic["loss_history"])
    assert _tensors_equal(first["final_weights"], deterministic["final_weights"])
    assert torch.equal(first_prediction, _evaluate_on_grid(deterministic["model"]))

    different_domain, different_model = _create_burgers(42)
    different = _train_burgers(different_domain, different_model)
    assert not torch.equal(first_prediction, _evaluate_on_grid(different["model"]))

    if torch.cuda.is_available():
        try:
            cuda_domain, cuda_model = _create_burgers(69, deterministic=True, device="cuda")
            cuda_prediction = _evaluate_on_grid(_train_burgers(cuda_domain, cuda_model)["model"])
            assert torch.isfinite(cuda_prediction).all()
        finally:
            import deepflow.utility as utility

            utility.device = "cpu"
