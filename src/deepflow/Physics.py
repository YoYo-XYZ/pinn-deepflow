import torch
from .Utility import calc_grad

class PDE():
    def __init__(self):
        pass

    def calc_residual_field(self):
        residuals_field = 0
        for residual_field in self.residual_fields:
            residuals_field += torch.abs(residual_field)
        return residuals_field

    def cal_loss_field(self):
        loss_field = 0
        for residual_field in self.residual_fields:
            loss_field += residual_field**2
        return loss_field

    def calc_loss(self):
        loss_field = 0
        for residual_field in self.residual_fields:
            loss_field += residual_field**2
        loss = torch.mean(loss_field)
        return loss

class NVS(PDE):
    def __init__(self, vu=0.001/1000, rho=1000.0):
        super().__init__()
        self.vu = vu
        self.rho = rho
        self.var = {}

    def calc(self, inputs_dict):
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']
        v = inputs_dict['v']
        p = inputs_dict['p']

        """
        Calculates the residuals of the incompressible Navier-Stokes equations.
        The residuals represent how well the network's predictions satisfy the PDEs.
        """
        # First-order derivatives

        if t is None:
            u_t = None
            v_t = None
        else:
            u_t = calc_grad(u, t)
            v_t = calc_grad(v, t)
            
        self.var['u_x'] = u_x = calc_grad(u, x)
        self.var['v_x'] = v_x = calc_grad(v, x)
        self.var['p_x'] = p_x = calc_grad(p, x)
        
        self.var['u_y'] = u_y = calc_grad(u, y)
        self.var['v_y'] = v_y = calc_grad(v, y)
        self.var['p_y'] = p_y = calc_grad(p, y)

        # Second-order derivatives
        self.var['u_xx'] = u_xx = calc_grad(u_x, x)
        self.var['v_xx'] = v_xx = calc_grad(v_x, x)
        self.var['u_yy'] = u_yy = calc_grad(u_y, y)
        self.var['v_yy'] = v_yy = calc_grad(v_y, y)
        
        # PDE residuals
        # Continuity equation (mass conservation)
        mass_residual = u_x + v_y
          
        if t is None:
            # X-momentum equation
            x_momentum_residual = (u * u_x + v * u_y -
                                self.vu * (u_xx + u_yy) +
                                p_x / self.rho)
                                
            # Y-momentum equation
            y_momentum_residual = (u * v_x + v * v_y -
                                self.vu * (v_xx + v_yy) +
                                p_y / self.rho)

        else:
            # X-momentum equation
            x_momentum_residual = (u_t + u * u_x + v * u_y -
                                self.vu * (u_xx + u_yy) +
                                p_x / self.rho)
                                
            # Y-momentum equation
            y_momentum_residual = (v_t + u * v_x + v * v_y -
                                self.vu * (v_xx + v_yy) +
                                p_y / self.rho)    

        self.residual_fields = (mass_residual, x_momentum_residual, y_momentum_residual)

        return self.residual_fields

class NVS_nondimensional(PDE):
    def __init__(self, U, L, mu, rho):
        super().__init__()
        self.U = U
        self.L = L
        self.mu = mu
        self.rho = rho
        self.Re = rho*U*L/mu
        self.var = {}

    def calc(self, inputs_dict):
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']
        v = inputs_dict['v']
        p = inputs_dict['p']

        """
        Calculates the residuals of the incompressible Navier-Stokes equations.
        The residuals represent how well the network's predictions satisfy the PDEs.
        """
        # First-order derivatives

        if t is None:
            u_t = None
            v_t = None
        else:
            u_t = calc_grad(u, t)
            v_t = calc_grad(v, t)
            
        self.var['u_x'] = u_x = calc_grad(u, x)
        self.var['v_x'] = v_x = calc_grad(v, x)
        self.var['p_x'] = p_x = calc_grad(p, x)
        
        self.var['u_y'] = u_y = calc_grad(u, y)
        self.var['v_y'] = v_y = calc_grad(v, y)
        self.var['p_y'] = p_y = calc_grad(p, y)

        # Second-order derivatives
        self.var['u_xx'] = u_xx = calc_grad(u_x, x)
        self.var['v_xx'] = v_xx = calc_grad(v_x, x)
        self.var['u_yy'] = u_yy = calc_grad(u_y, y)
        self.var['v_yy'] = v_yy = calc_grad(v_y, y)
        
        # PDE residuals
        # Continuity equation (mass conservation)
        mass_residual = u_x + v_y
          
        if t is None:
            # X-momentum equation
            x_momentum_residual = (u * u_x + v * u_y -
                                (u_xx + u_yy)/self.Re +
                                p_x)
                                
            # Y-momentum equation
            y_momentum_residual = (u * v_x + v * v_y -
                                (v_xx + v_yy)/self.Re +
                                p_y)

        else:
            # X-momentum equation
            x_momentum_residual = (u_t + u * u_x + v * u_y -
                                (u_xx + u_yy)/self.Re +
                                p_x)
                                
            # Y-momentum equation
            y_momentum_residual = (v_t + u * v_x + v * v_y -
                                (v_xx + v_yy)/self.Re +
                                p_y)    

        self.residual_fields = (mass_residual, x_momentum_residual, y_momentum_residual)

        return self.residual_fields

    def dimensionalize(self):
        self.var['x'] = self.var['x'] / self.L
        self.var['y'] = self.var['y'] / self.L
        self.var['u'] = self.var['u'] / self.U
        self.var['v'] = self.var['v'] / self.U
        self.var['p'] = self.var['p'] / (self.rho*self.U**2)
        if self.var['t']:
            self.var['t'] = self.var['t'] * self.U/self.L

    def non_dimensionalize(self):
        self.var['x'] = self.var['x'] * self.L
        self.var['y'] = self.var['y'] * self.L
        self.var['u'] = self.var['u'] * self.U
        self.var['v'] = self.var['v'] * self.U
        self.var['p'] = self.var['p'] * (self.rho*self.U**2)
        if self.var['t']:
            self.var['t'] = self.var['t'] / (self.U/self.L)

class Heat(PDE):
    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.var = {}

    def calc_residual(self, inputs_dict):
        x = inputs_dict['x']
        y = inputs_dict['y']
        t = inputs_dict['t']
        u = inputs_dict['u']

        # Derivatives
        u_t = calc_grad(u, t)
        u_x = calc_grad(u, x)
        u_y = calc_grad(u, y)
        u_xx = calc_grad(u_x, x)
        u_yy = calc_grad(u_y, y)

        # Heat equation residual
        heat_residual = u_t - self.alpha * (u_xx + u_yy)

        self.residuals = (heat_residual,)
        return self.residuals