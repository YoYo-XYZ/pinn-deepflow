from abc import ABC, abstractmethod

import torch
from typing import Dict, Tuple, Optional
from .utility import calc_grad, calc_grads

class PDE(ABC):
    """
    Base class for Physics-Informed Differential Equations.
    """
    def __init__(self):
        self.var = {}
        pass

    @abstractmethod
    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        """
        Must be implemented by child classes to populate self.residual_fields.
        """
        pass

    def calc_residual_field(self)-> torch.Tensor:
        """Calculates the absolute residuals field."""
        return torch.stack(self.residual_fields, dim=0).abs().sum(dim=0)
    
    def calc_residuals(self)-> torch.Tensor:
        """
        Calculates the Mean Absolute Error (MAE) of the residuals.
        """
        return torch.mean(torch.stack(self.residual_fields, dim=0).abs().sum(dim=0))

    def calc_loss_field(self)-> torch.Tensor:
        """Calculates the squared residuals field."""
        return torch.stack(self.residual_fields, dim=0).pow(2).sum(dim=0)

    def calc_loss(self)-> torch.Tensor:
        """
        Calculates the Mean Squared Error (MSE) of the residuals.
        """
        return torch.mean(torch.stack(self.residual_fields, dim=0).pow(2).sum(dim=0))

class CustomPDE(PDE):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def compute_residuals(self, inputs_dict):
        # Assuming func returns a tuple of residuals
        self.residual_fields = self.func(inputs_dict)
        return self.residual_fields

class NavierStokes(PDE):
    """
    Incompressible Navier-Stokes equations (2D).
    Handles both Steady and Unsteady states automatically based on input 't'.
    """
    def __init__(self, mu: float, rho: float, U: float = 1.0, L: float = 1.0):
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
        continuity_residual = u_x + v_y

        # 2. X-Momentum Equation
        x_momentum_residual = (u_t + (u * u_x) + (v * u_y)) + p_x - ((u_xx + u_yy) / self.Re)

        # 3. Y-Momentum Equation
        y_momentum_residual = (v_t + (u * v_x) + (v * v_y)) + p_y - ((v_xx + v_yy) / self.Re)

        self.residual_fields = (continuity_residual, x_momentum_residual, y_momentum_residual)

        self.var.update(x= x, y=y, u=u, v=v, p=p, u_x=u_x, u_y=u_y, v_x=v_x, v_y=v_y, p_x=p_x, p_y=p_y,
                        continuity_residual=continuity_residual, x_momentum_residual=x_momentum_residual, y_momentum_residual=y_momentum_residual)
        
        return self.residual_fields

    def nondimensionalize_inputs(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Scales physical inputs to non-dimensional form.
        New_val = val / scale
        """
        new_inputs = {}
        for key, val in inputs.items():
            if key in self.scale_map:
                new_inputs[key] = val / self.scale_map[key]
            else:
                new_inputs[key] = val
        return new_inputs

class HeatEquation(PDE):
    """
    2D Heat Equation: u_t = alpha * (u_xx + u_yy)
    """
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def compute_residuals(self, inputs_dict: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor]:
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