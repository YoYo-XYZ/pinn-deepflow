"""Built-in differential-equation residual models for PINN training."""

from abc import ABC, abstractmethod

import torch
from typing import Dict, Tuple, Optional
from .utility import calc_grad, calc_grads

class PDE(ABC):
    """Base class for PDE residuals and losses.

    Subclasses set ``residual_fields`` in ``compute_residuals``. ``var``
    exposes derived fields during evaluation.
    """
    def __init__(self):
        self.var = {}
        pass

    @abstractmethod
    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Compute and store one residual tensor per equation; return them as a
        tuple.

        ``inputs_dict`` contains the PDE's coordinate and predicted-field
        tensors.
        """
        pass

    def calc_residual_field(self)-> torch.Tensor:
        """Return the pointwise sum of absolute residual components."""
        return torch.stack(self.residual_fields, dim=0).abs().sum(dim=0)
    
    def calc_residual_field_raw(self)-> torch.Tensor:
        """Return residual components without taking absolute values."""
        return torch.stack(self.residual_fields, dim=0)
    
    def calc_residuals(self)-> torch.Tensor:
        """Return the mean pointwise sum of absolute residual components."""
        return torch.mean(torch.stack(self.residual_fields, dim=0).abs().sum(dim=0))

    def calc_loss_field(self)-> torch.Tensor:
        """Return the pointwise sum of squared residual components."""
        return torch.stack(self.residual_fields, dim=0).pow(2).sum(dim=0)

    def calc_loss(self)-> torch.Tensor:
        """Return the mean pointwise sum of squared residual components."""
        return torch.mean(torch.stack(self.residual_fields, dim=0).pow(2).sum(dim=0))

class CustomPDE(PDE):
    """Wrap ``func(inputs_dict)`` returning a tuple of residual tensors."""

    def __init__(self, func):
        """Initialize a custom residual wrapper."""
        super().__init__()
        self.func = func

    def compute_residuals(self, inputs_dict):
        """Evaluate the user-supplied residual function."""
        self.residual_fields = self.func(inputs_dict)
        return self.residual_fields

class NavierStokes(PDE):
    """2D incompressible Navier-Stokes equations; include ``t`` for unsteady flow.

    Args:
        mu: Dynamic viscosity.
        rho: Fluid density.
        U: Reference velocity.
        L: Reference length.
    """
    def __init__(self, mu: float, rho: float, U: float = 1.0, L: float = 1.0):
        """Initialize the nondimensionalized Navier-Stokes model."""
        super().__init__()
        self.U = U
        self.L = L
        self.mu = mu
        self.rho = rho
        self.Re = (rho * U * L) / mu
        
        self.scale_map = {
            'x': self.L, 'y': self.L,
            'u': self.U, 'v': self.U,
            'p': self.rho * (self.U**2),
            't': self.L / self.U
        }

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Return ``(continuity, x_momentum, y_momentum)`` residual tensors.

        ``inputs_dict`` requires ``x``, ``y``, ``u``, ``v``, ``p``, and
        optional ``t``.
        """
        x, y = inputs_dict['x'], inputs_dict['y']
        u, v, p = inputs_dict['u'], inputs_dict['v'], inputs_dict['p']
        t = inputs_dict.get('t', None)

        if t is None:
            u_t, v_t = 0.0, 0.0
            u_x, u_y = calc_grads(u, (x, y))
            v_x, v_y = calc_grads(v, (x, y))
            p_x, p_y = calc_grads(p, (x, y))
        else:
            u_x, u_y, u_t = calc_grads(u, (x, y, t))
            v_x, v_y, v_t = calc_grads(v, (x, y, t))
            p_x, p_y = calc_grads(p, (x, y))

        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)

        continuity_residual = (u_x + v_y)

        x_momentum_residual = (u_t + (u * u_x) + (v * u_y)) + p_x - ((u_xx + u_yy) / self.Re)

        y_momentum_residual = (v_t + (u * v_x) + (v * v_y)) + p_y - ((v_xx + v_yy) / self.Re)

        self.residual_fields = (continuity_residual, x_momentum_residual, y_momentum_residual)

        self.var.update(x= x, y=y, u=u, v=v, p=p, u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y, p_x=p_x, p_y=p_y,
                        continuity_residual=continuity_residual, x_momentum_residual=x_momentum_residual, y_momentum_residual=y_momentum_residual)
        
        return self.residual_fields

    def nondimensionalize_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return inputs divided by their reference scales; copy unknown entries
        unchanged.
        """
        new_inputs = {}
        for key, val in inputs.items():
            if key in self.scale_map:
                new_inputs[key] = val / self.scale_map[key]
            else:
                new_inputs[key] = val
        return new_inputs


class StreamFunctionNavierStokes(PDE):
    """Steady 2D Navier-Stokes using ``psi`` and ``p``.

    ``u = dpsi/dy`` and ``v = -dpsi/dx`` satisfy continuity, leaving two
    momentum residuals. Transient coordinates are unsupported.

    Args:
        mu: Dynamic viscosity.
        rho: Fluid density.
        U: Reference velocity.
        L: Reference length.
    """
    def __init__(self, mu: float, rho: float, U: float = 1.0, L: float = 1.0):
        """Initialize the steady stream-function formulation."""
        super().__init__()
        self.U = U
        self.L = L
        self.mu = mu
        self.rho = rho
        self.Re = (rho * U * L) / mu

        self.scale_map = {
            'x': self.L, 'y': self.L,
            'u': self.U, 'v': self.U,
            'psi': self.U * self.L,
            'p': self.rho * (self.U**2),
        }

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Return ``(x_momentum, y_momentum)`` residual tensors.

        ``inputs_dict`` requires ``x``, ``y``, ``psi``, and ``p``. Supplying
        ``t`` raises ``ValueError``.
        """
        if 't' in inputs_dict:
            raise ValueError(
                "StreamFunctionNavierStokes supports steady problems only; "
                "remove the 't' input or use NavierStokes."
            )

        x, y = inputs_dict['x'], inputs_dict['y']
        psi, p = inputs_dict['psi'], inputs_dict['p']

        psi_x, psi_y = calc_grads(psi, (x, y))
        u = psi_y
        v = -psi_x

        u_x, u_y = calc_grads(u, (x, y))
        v_x, v_y = calc_grads(v, (x, y))
        p_x, p_y = calc_grads(p, (x, y))

        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)

        x_momentum_residual = (u * u_x) + (v * u_y) + p_x - ((u_xx + u_yy) / self.Re)
        y_momentum_residual = (u * v_x) + (v * v_y) + p_y - ((v_xx + v_yy) / self.Re)

        self.residual_fields = (x_momentum_residual, y_momentum_residual)
        self.var.update(
            x=x, y=y, psi=psi, u=u, v=v, p=p,
            psi_x=psi_x, psi_y=psi_y,
            u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y,
            p_x=p_x, p_y=p_y,
            continuity_residual=u_x + v_y,
            x_momentum_residual=x_momentum_residual,
            y_momentum_residual=y_momentum_residual,
        )

        return self.residual_fields

    def nondimensionalize_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return inputs divided by their reference scales."""
        return {
            key: val / self.scale_map[key] if key in self.scale_map else val
            for key, val in inputs.items()
        }

class HeatEquation(PDE):
    """2D heat equation ``u_t = alpha * (u_xx + u_yy)`` with thermal diffusivity
    ``alpha``.
    """
    def __init__(self, alpha: float):
        """Initialize the heat-equation residual model."""
        super().__init__()
        self.alpha = alpha

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Return ``(u_t - alpha * (u_xx + u_yy),)`` from ``x``, ``y``, ``t``, and
        ``u``.
        """
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']

        u_x, u_y, u_t = calc_grads(u, (x, y, t))

        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)

        heat_residual = u_t - self.alpha * (u_xx + u_yy)

        self.residual_fields = (heat_residual,)
        return self.residual_fields
    
class WaveEquation(PDE):
    """2D wave equation ``u_tt = c**2 * (u_xx + u_yy)`` with wave speed ``c``."""
    def __init__(self, c: float):
        """Initialize the wave-equation residual model."""
        super().__init__()
        self.c = c

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Return ``(u_tt - c**2 * (u_xx + u_yy),)`` from ``x``, ``y``, ``t``, and
        ``u``.
        """
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']

        u_x, u_y, u_t = calc_grads(u, (x, y, t))

        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)
        u_tt = calc_grad(u_t, t)

        wave_residual = u_tt - self.c**2 * (u_xx + u_yy)

        self.residual_fields = (wave_residual,)
        return self.residual_fields

class BurgersEquation1D(PDE):
    """Steady Burgers equation ``u_y + u * u_x = nu * u_xx`` with viscosity ``nu``."""
    def __init__(self, nu: float):
        """Initialize the steady Burgers residual model."""
        super().__init__()
        self.nu = nu

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Return ``(u_y + u * u_x - nu * u_xx,)`` from ``x``, ``y``, and ``u``."""
        x = inputs_dict['x']
        y = inputs_dict['y']
        u = inputs_dict['u']

        u_x, u_y = calc_grads(u, (x, y))

        u_xx = calc_grad(u_x, x)

        burgers_residual = u_y + u * u_x - self.nu * u_xx

        self.residual_fields = (burgers_residual,)
        return self.residual_fields
