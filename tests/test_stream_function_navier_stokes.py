import os
import sys

import torch

_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from deepflow.pde import StreamFunctionNavierStokes


def test_stream_function_derives_velocity_and_residuals():
    x = torch.tensor([0.2, 0.7], dtype=torch.float64, requires_grad=True)
    y = torch.tensor([0.3, 0.4], dtype=torch.float64, requires_grad=True)
    psi = x.square() * y + y.pow(3)
    p = 2.0 * x - 3.0 * y

    pde = StreamFunctionNavierStokes(mu=1.0, rho=1.0)
    residuals = pde.compute_residuals({'x': x, 'y': y, 'psi': psi, 'p': p})

    expected_u = x.square() + 3.0 * y.square()
    expected_v = -2.0 * x * y
    expected_x = 2.0 * x.pow(3) - 6.0 * x * y.square() - 6.0
    expected_y = 2.0 * x.square() * y - 6.0 * y.pow(3) - 3.0

    assert len(residuals) == 2
    assert torch.allclose(pde.var['u'], expected_u)
    assert torch.allclose(pde.var['v'], expected_v)
    assert torch.allclose(pde.var['continuity_residual'], torch.zeros_like(x))
    assert torch.allclose(residuals[0], expected_x)
    assert torch.allclose(residuals[1], expected_y)


def test_stream_function_form_rejects_transient_inputs():
    x = torch.zeros(2, requires_grad=True)
    y = torch.zeros(2, requires_grad=True)
    t = torch.zeros(2, requires_grad=True)
    psi = x + y + t
    p = x - y
    pde = StreamFunctionNavierStokes(mu=1.0, rho=1.0)

    try:
        pde.compute_residuals({'x': x, 'y': y, 't': t, 'psi': psi, 'p': p})
    except ValueError as exc:
        assert 'steady problems only' in str(exc)
    else:
        raise AssertionError('Expected transient stream-function input to fail')