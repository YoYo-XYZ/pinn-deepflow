import math

import pytest
import torch

import deepflow as df

pytestmark = pytest.mark.usefixtures("deterministic_cpu")


@pytest.fixture
def coordinates():
    return {
        key: torch.tensor(values, requires_grad=True)
        for key, values in {
            "x": [0.2, 0.5, 0.8],
            "y": [0.3, 0.7, 0.4],
            "t": [0.1, 0.2, 0.6],
        }.items()
    }


def test_custom_pde_residual_and_loss_reductions():
    pde = df.CustomPDE(lambda values: (values["u"], -2 * values["u"]))
    residuals = pde.compute_residuals({"u": torch.tensor([-2.0, 1.0, 3.0])})
    torch.testing.assert_close(residuals[0], torch.tensor([-2.0, 1.0, 3.0]))
    torch.testing.assert_close(
        pde.calc_residual_field_raw(),
        torch.tensor([[-2.0, 1.0, 3.0], [4.0, -2.0, -6.0]]),
    )
    torch.testing.assert_close(pde.calc_residual_field(), torch.tensor([6.0, 3.0, 9.0]))
    assert pde.calc_residuals().item() == 6
    torch.testing.assert_close(pde.calc_loss_field(), torch.tensor([20.0, 5.0, 45.0]))
    assert pde.calc_loss().item() == pytest.approx(70 / 3)


@pytest.mark.parametrize("kind", ["heat", "wave"])
def test_diffusion_and_wave_eigenmodes_have_zero_residual(kind, coordinates):
    x, y, t = (coordinates[k] for k in ("x", "y", "t"))
    spatial = torch.sin(math.pi * x) * torch.sin(2 * math.pi * y)
    if kind == "heat":
        pde = df.HeatEquation(0.3)
        u = spatial * torch.exp(-0.3 * 5 * math.pi**2 * t)
    else:
        pde = df.WaveEquation(1.7)
        u = spatial * torch.cos(1.7 * math.sqrt(5) * math.pi * t)
    (residual,) = pde.compute_residuals(coordinates | {"u": u})
    torch.testing.assert_close(residual, torch.zeros_like(x), atol=1e-12, rtol=0)


@pytest.mark.parametrize("kind", ["heat", "wave", "burgers"])
def test_scalar_pde_nonzero_polynomial_residuals(kind, coordinates):
    x, y, t = (coordinates[k] for k in ("x", "y", "t"))
    u = x**3 + 2 * y**3 + 3 * t**3
    if kind == "heat":
        pde, expected = df.HeatEquation(0.4), 9 * t**2 - 0.4 * (6 * x + 12 * y)
    elif kind == "wave":
        pde, expected = df.WaveEquation(1.5), 18 * t - 1.5**2 * (6 * x + 12 * y)
    else:
        pde, expected = df.BurgersEquation1D(0.2), 6 * y**2 + u * 3 * x**2 - 0.2 * 6 * x
    (residual,) = pde.compute_residuals(coordinates | {"u": u})
    torch.testing.assert_close(residual, expected)


@pytest.mark.parametrize("transient", [False, True])
def test_navier_stokes_all_terms_against_polynomial_derivatives(transient, coordinates):
    x, y, t = (coordinates[k] for k in ("x", "y", "t"))
    u = x**3 + y**3 + (t**3 if transient else 0)
    v, p = x**2 * y**2, x**2 + y**2
    pde = df.NavierStokes(mu=2, rho=3, U=4, L=5)
    inputs = {"x": x, "y": y, "u": u, "v": v, "p": p}
    if transient:
        inputs["t"] = t
    actual = pde.compute_residuals(inputs)
    nu = 1 / 30
    expected = (
        3 * x**2 + 2 * x**2 * y,
        (3 * t**2 if transient else 0)
        + u * 3 * x**2
        + v * 3 * y**2
        + 2 * x
        - nu * (6 * x + 6 * y),
        u * 2 * x * y**2 + v * 2 * x**2 * y + 2 * y - nu * (2 * y**2 + 2 * x**2),
    )
    for result, target in zip(actual, expected):
        torch.testing.assert_close(result, target)
    torch.testing.assert_close(pde.var["u_x"], 3 * x**2)
    torch.testing.assert_close(pde.var["p_y"], 2 * y)


def test_stream_function_derives_velocity_and_momentum(coordinates):
    x, y = coordinates["x"], coordinates["y"]
    pde = df.StreamFunctionNavierStokes(mu=0.5, rho=1)
    residuals = pde.compute_residuals(
        {"x": x, "y": y, "psi": x**3 * y**3, "p": x**2 + y**2}
    )
    u, v = 3 * x**3 * y**2, -3 * x**2 * y**3
    torch.testing.assert_close(pde.var["u"], u)
    torch.testing.assert_close(pde.var["v"], v)
    torch.testing.assert_close(
        pde.var["continuity_residual"], torch.zeros_like(x), atol=1e-14, rtol=0
    )
    expected = (
        u * 9 * x**2 * y**2
        + v * 6 * x**3 * y
        + 2 * x
        - 0.5 * (18 * x * y**2 + 6 * x**3),
        u * (-6 * x * y**3)
        + v * (-9 * x**2 * y**2)
        + 2 * y
        - 0.5 * (-6 * y**3 - 18 * x**2 * y),
    )
    for result, target in zip(residuals, expected):
        torch.testing.assert_close(result, target)


def test_stream_function_rejects_transient_inputs(coordinates):
    with pytest.raises(ValueError, match="steady problems only"):
        df.StreamFunctionNavierStokes(1, 1).compute_residuals(coordinates)


def test_navier_stokes_taylor_green_vortex_is_an_exact_transient_solution(coordinates):
    x, y, t = (coordinates[key] for key in ("x", "y", "t"))
    nu = 0.2
    decay = torch.exp(-2 * nu * t)
    u = -torch.cos(x) * torch.sin(y) * decay
    v = torch.sin(x) * torch.cos(y) * decay
    p = -0.25 * (torch.cos(2 * x) + torch.cos(2 * y)) * decay**2
    residuals = df.NavierStokes(mu=nu, rho=1).compute_residuals(
        coordinates | {"u": u, "v": v, "p": p}
    )
    for residual in residuals:
        torch.testing.assert_close(residual, torch.zeros_like(x), atol=1e-13, rtol=0)


def test_stream_function_harmonic_potential_flow_is_an_exact_solution(coordinates):
    x, y = coordinates["x"], coordinates["y"]
    psi = torch.sin(x) * torch.sinh(y)
    u, v = torch.sin(x) * torch.cosh(y), -torch.cos(x) * torch.sinh(y)
    p = -0.5 * (u**2 + v**2)
    pde = df.StreamFunctionNavierStokes(mu=0.3, rho=1)
    residuals = pde.compute_residuals({"x": x, "y": y, "psi": psi, "p": p})
    for residual in residuals:
        torch.testing.assert_close(residual, torch.zeros_like(x), atol=1e-13, rtol=0)


@pytest.mark.parametrize("pde_type", [df.NavierStokes, df.StreamFunctionNavierStokes])
def test_nondimensionalization_uses_physical_scales_without_mutating_inputs(pde_type):
    pde = pde_type(mu=2, rho=3, U=4, L=5)
    inputs = {
        key: torch.tensor([value])
        for key, value in {
            "x": 10.0,
            "y": 15.0,
            "u": 8.0,
            "v": 12.0,
            "p": 96.0,
            "psi": 40.0,
            "extra": 7.0,
        }.items()
    }
    scaled = pde.nondimensionalize_inputs(inputs)
    for key, expected in {"x": 2, "y": 3, "u": 2, "v": 3, "p": 2, "extra": 7}.items():
        assert scaled[key].item() == expected
    if pde_type is df.StreamFunctionNavierStokes:
        assert scaled["psi"].item() == 2
    else:
        assert pde.nondimensionalize_inputs({"t": torch.tensor([2.5])})["t"].item() == 2
    assert inputs["x"].item() == 10


def test_pde_loss_backpropagates_to_learnable_field(coordinates):
    amplitude = torch.tensor(2.0, requires_grad=True)
    pde = df.HeatEquation(0.1)
    u = amplitude * (
        coordinates["x"] ** 3 + coordinates["y"] ** 3 + coordinates["t"] ** 3
    )
    pde.compute_residuals(coordinates | {"u": u})
    loss = pde.calc_loss()
    loss.backward()
    torch.testing.assert_close(amplitude.grad, 2 * loss.detach() / amplitude.detach())


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="Autograd helpers cannot differentiate constant fields or constant first derivatives",
)
@pytest.mark.parametrize("kind", ["heat", "wave", "burgers", "navier", "stream"])
def test_constant_fields_are_valid_zero_residual_solutions(kind, coordinates):
    zero = torch.zeros_like(coordinates["x"])
    inputs = {"x": coordinates["x"], "y": coordinates["y"]}
    if kind in {"heat", "wave"}:
        inputs["t"] = coordinates["t"]
    pde = {
        "heat": df.HeatEquation(0.1),
        "wave": df.WaveEquation(1),
        "burgers": df.BurgersEquation1D(0.1),
        "navier": df.NavierStokes(1, 1),
        "stream": df.StreamFunctionNavierStokes(1, 1),
    }[kind]
    residuals = pde.compute_residuals(
        inputs | {"u": zero, "v": zero, "p": zero, "psi": zero}
    )
    for residual in residuals:
        torch.testing.assert_close(residual, zero)


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="A linear field's first derivative has no grad_fn for second differentiation",
)
def test_burgers_linear_field_has_known_nonzero_residual(coordinates):
    x, y = coordinates["x"], coordinates["y"]
    (residual,) = df.BurgersEquation1D(0.1).compute_residuals(
        {"x": x, "y": y, "u": x + y}
    )
    torch.testing.assert_close(residual, 1 + x + y)
