"""Built-in differential-equation residual models for PINN training."""

from abc import ABC, abstractmethod

import torch
from typing import Dict, Tuple, Optional
from .utility import calc_grad, calc_grads

class PDE(ABC):
    """Base class for physics-informed differential equations.

    Subclasses populate ``residual_fields`` in ``compute_residuals``.
    Residual and loss helpers then reduce those fields over the sample points.

    Attributes:
        var: Derived variables and residuals exposed during evaluation.
        residual_fields: Tuple of per-equation residual tensors after a
            residual computation.
    """
    def __init__(self):
        self.var = {}
        pass

    @abstractmethod
    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Compute and store equation residuals for model outputs.

        Args:
            inputs_dict: Mapping containing coordinate tensors and predicted
                field tensors required by the PDE.

        Returns:
            Tuple of residual tensors, one for each equation.
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
    """Wrap a user-supplied residual function as a ``PDE``.

    Args:
        func: Callable accepting an input dictionary and returning a tuple of
            residual tensors.
    """

    def __init__(self, func):
        """Initialize a custom residual wrapper."""
        super().__init__()
        self.func = func

    def compute_residuals(self, inputs_dict):
        """Evaluate the user-supplied residual function."""
        self.residual_fields = self.func(inputs_dict)
        return self.residual_fields

class NavierStokes(PDE):
    """
    Incompressible Navier-Stokes equations (2D).
    Handles both Steady and Unsteady states automatically based on input 't'.

    Args:
        mu: Dynamic viscosity.
        rho: Fluid density.
        U: Reference velocity used for nondimensionalization.
        L: Reference length used for nondimensionalization.
    """
    def __init__(self, mu: float, rho: float, U: float = 1.0, L: float = 1.0):
        """Initialize the nondimensionalized Navier-Stokes model."""
        super().__init__()
        self.U = U
        self.L = L
        self.mu = mu
        self.rho = rho
        self.Re = (rho * U * L) / mu
        
        # Scaling factors
        self.scale_map = {
            'x': self.L, 'y': self.L,
            'u': self.U, 'v': self.U,
            'p': self.rho * (self.U**2),
            't': self.L / self.U
        }

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """Compute continuity and momentum residuals.

        Args:
            inputs_dict: Must contain ``x``, ``y``, ``u``, ``v``, and ``p``;
                include ``t`` for an unsteady problem.

        Returns:
            Tuple ``(continuity, x_momentum, y_momentum)``.
        """
        x, y = inputs_dict['x'], inputs_dict['y']
        u, v, p = inputs_dict['u'], inputs_dict['v'], inputs_dict['p']
        t = inputs_dict.get('t', None)

        # First derivatives
        if t is None:
            u_t, v_t = 0.0, 0.0
            u_x, u_y = calc_grads(u, (x, y))
            v_x, v_y = calc_grads(v, (x, y))
            p_x, p_y = calc_grads(p, (x, y))
        else:
            u_x, u_y, u_t = calc_grads(u, (x, y, t))
            v_x, v_y, v_t = calc_grads(v, (x, y, t))
            p_x, p_y = calc_grads(p, (x, y))

        # Second derivatives
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)
        v_xx = calc_grad(v_x, x)
        v_yy = calc_grad(v_y, y)

        # 1. Continuity Equation (Mass Conservation)
        continuity_residual = (u_x + v_y)

        # 2. X-Momentum Equation
        x_momentum_residual = (u_t + (u * u_x) + (v * u_y)) + p_x - ((u_xx + u_yy) / self.Re)

        # 3. Y-Momentum Equation
        y_momentum_residual = (v_t + (u * v_x) + (v * v_y)) + p_y - ((v_xx + v_yy) / self.Re)

        self.residual_fields = (continuity_residual, x_momentum_residual, y_momentum_residual)

        self.var.update(x= x, y=y, u=u, v=v, p=p, u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y, p_x=p_x, p_y=p_y,
                        continuity_residual=continuity_residual, x_momentum_residual=x_momentum_residual, y_momentum_residual=y_momentum_residual)
        
        return self.residual_fields

    def nondimensionalize_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Scale recognized physical inputs to nondimensional form.

        Args:
            inputs: Mapping of coordinate and field names to tensors.

        Returns:
            A new mapping with known quantities divided by their reference
            scales; unknown entries are copied unchanged.
        """
        new_inputs = {}
        for key, val in inputs.items():
            if key in self.scale_map:
                new_inputs[key] = val / self.scale_map[key]
            else:
                new_inputs[key] = val
        return new_inputs


class StreamFunctionNavierStokes(PDE):
    """
    Steady incompressible Navier-Stokes equations using ``psi`` and ``p``.

    The velocity is derived from the stream function using
    ``u = dpsi/dy`` and ``v = -dpsi/dx``.  Continuity is therefore satisfied
    identically, and the PDE contributes only the two momentum residuals.
    This formulation currently supports steady problems with ``x`` and ``y``
    coordinates only.

    Args:
        mu: Dynamic viscosity.
        rho: Fluid density.
        U: Reference velocity used for nondimensionalization.
        L: Reference length used for nondimensionalization.
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
        """Compute the two momentum residuals from ``psi`` and ``p``.

        Args:
            inputs_dict: Must contain ``x``, ``y``, ``psi``, and ``p``.

        Returns:
            Tuple ``(x_momentum, y_momentum)``.

        Raises:
            ValueError: If transient coordinate ``t`` is supplied.
        """
        if 't' in inputs_dict:
            raise ValueError(
                "StreamFunctionNavierStokes supports steady problems only; "
                "remove the 't' input or use NavierStokes."
            )

        x, y = inputs_dict['x'], inputs_dict['y']
        psi, p = inputs_dict['psi'], inputs_dict['p']

        # Keep x and y as the differentiation inputs.  They are the original
        # coordinate leaves used by the model, which is required for the
        # higher-order derivatives below.
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
        """Scale recognized stream-function inputs to nondimensional form.

        Args:
            inputs: Mapping of coordinate and field names to tensors.

        Returns:
            A new mapping with known quantities divided by their reference
            scales.
        """
        return {
            key: val / self.scale_map[key] if key in self.scale_map else val
            for key, val in inputs.items()
        }

class HeatEquation(PDE):
    """
    2D Heat Equation: u_t = alpha * (u_xx + u_yy)

    Args:
        alpha: Thermal diffusivity.
    """
    def __init__(self, alpha: float):
        """Initialize the heat-equation residual model."""
        super().__init__()
        self.alpha = alpha

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Compute the heat-equation residual.

        Args:
            inputs_dict: Mapping containing ``x``, ``y``, ``t``, and ``u``.

        Returns:
            A one-element tuple containing ``u_t - alpha * (u_xx + u_yy)``.
        """
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']

        # First derivatives
        u_x, u_y, u_t = calc_grads(u, (x, y, t))

        # Second derivatives
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)

        # Residual
        heat_residual = u_t - self.alpha * (u_xx + u_yy)

        self.residual_fields = (heat_residual,)
        return self.residual_fields
    
class WaveEquation(PDE):
    """
    2D Wave Equation: u_tt = c^2 * (u_xx + u_yy)

    Args:
        c: Wave propagation speed.
    """
    def __init__(self, c: float):
        """Initialize the wave-equation residual model."""
        super().__init__()
        self.c = c

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Compute the wave-equation residual.

        Args:
            inputs_dict: Mapping containing ``x``, ``y``, ``t``, and ``u``.

        Returns:
            A one-element tuple containing ``u_tt - c**2 * (u_xx + u_yy)``.
        """
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']

        # First derivatives
        u_x, u_y, u_t = calc_grads(u, (x, y, t))

        # Second derivatives
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)
        u_tt = calc_grad(u_t, t)

        # Residual
        wave_residual = u_tt - self.c**2 * (u_xx + u_yy)

        self.residual_fields = (wave_residual,)
        return self.residual_fields

class BurgersEquation1D(PDE):
    """
    Steady 2D Burgers equation in the x-y domain:
    u_y + u * u_x = nu * u_xx.

    Args:
        nu: Viscosity coefficient.
    """
    def __init__(self, nu: float):
        """Initialize the steady Burgers residual model."""
        super().__init__()
        self.nu = nu

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
        """Compute the steady Burgers residual.

        Args:
            inputs_dict: Mapping containing ``x``, ``y``, and ``u``.

        Returns:
            A one-element tuple containing ``u_y + u * u_x - nu * u_xx``.
        """
        x = inputs_dict['x']
        y = inputs_dict['y']
        u = inputs_dict['u']

        # First derivatives
        u_x, u_y = calc_grads(u, (x, y))

        # Second derivative
        u_xx = calc_grad(u_x, x)

        # Residual
        burgers_residual = u_y + u * u_x - self.nu * u_xx

        self.residual_fields = (burgers_residual,)
        return self.residual_fields
