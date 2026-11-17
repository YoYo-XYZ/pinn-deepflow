"""Physics constraints and coordinate processing attached to geometries."""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import warnings
from copy import deepcopy

import torch
import torch.nn as nn

from .nn import HardConstraint
from .pde import PDE
from .utility import calc_grad, get_device, get_dtype, _next_seed

class PhysicsAttach:
    """Attach conditions, PDEs, and samples to a geometry.

    Attributes:
        physics_type: ``"BC"``, ``"IC"``, or ``"PDE"``.
        inputs_tensor_dict: Model coordinate tensors.
        model_outputs: Cached predictions.
        residual_field: Pointwise absolute residuals.
    """

    def __init__(self):
        """Initialize empty physics and coordinate state."""
        self.is_sampled: bool = False
        self.physics_type: Optional[list[str]] = []
        self.range_t: Optional[Union[Tuple[float, float], torch.Tensor]] = None
        self.is_converged: bool = False
        
        self.condition_dict: Optional[Dict] = None
        self.condition_num: int = 0
        self.t: Optional[torch.Tensor] = None
        self.PDE: Optional[PDE] = None
        
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None
        self.X_: Optional[torch.Tensor] = None
        self.Y_: Optional[torch.Tensor] = None
        self.T_: Optional[torch.Tensor] = None
        self.X_residual_container: list[torch.Tensor] = []
        self.Y_residual_container: list[torch.Tensor] = []
        self._amounts_before_add: int = 0

        self.expo_scaling = None
        self.scheme = None
        
        self.inputs_tensor_dict: Dict[str, Optional[torch.Tensor]] = {}
        self.target_output_tensor_dict: Dict[str, torch.Tensor] = {}
        self.model_inputs: Dict = {}
        self.model_outputs: Dict = {}
        
        self.loss_field: Union[int, torch.Tensor] = 0
        self.loss_threshold: Optional[float] = None
        self.top_k_loss_threshold: Optional[float] = None


    def define_bc(self, condition_dict: Dict[str, Any]) -> None:
        """Set boundary conditions as a mapping of variable names to conditions."""
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.physics_type = "BC"
    
    def define_ic(self, condition_dict: Dict[str, Any]) -> None:
        """Set initial conditions as a mapping of variable names to conditions."""
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.physics_type = "IC"
    
    def define_pde(self, pde: PDE) -> None:
        """Attach an initialized ``deepflow.pde.PDE`` instance."""
        self.PDE = pde
        self.physics_type = "PDE"


    def set_coordinates(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Store matching ``x`` and ``y`` tensors; call ``process_coordinates`` to
        prepare them.
        """
        self.X = x
        self.Y = y
    
    def define_time(self, range_t: Union[Tuple[float, float], int, float] = None, sampling_scheme: str = None, expo_scaling = None) -> None:
        """Set time coordinates and sampling options.

        Args:
            range_t: ``(start, end)`` interval or fixed time.
            sampling_scheme: ``"uniform"`` or ``"random"`` for intervals.
            expo_scaling: Apply exponential time scaling.

        Raises ``ValueError`` if neither a new nor a saved range exists.
        """
        if range_t is not None: self.range_t = range_t
        elif self.range_t is None:
            raise ValueError("Time range must be defined before sampling time coordinates.")
        
        if sampling_scheme is not None:
            self.scheme = sampling_scheme
        elif self.scheme is None:
            self.scheme = "uniform"
            warnings.warn("Sampling has not yet defined. Uniform scheme is used.")

        if expo_scaling is not None:
            self.expo_scaling = expo_scaling
        elif self.expo_scaling is None:
            self.expo_scaling = False
            warnings.warn("expo_scaling has not yet defined. False is set as default.")

    def sampling_time(self) -> None:
        """Sample time into ``t`` and ``T`` using the configured range.

        Missing spatial coordinates or time range raises ``ValueError``.
        """
        if self.X is None: raise ValueError("Before sampling time t, X coordinate must be sampling first")
        n_points = len(self.X)
        device = get_device()

        if self.physics_type == "IC":
            self.t = self.range_t[0] * torch.ones_like(self.X, device=device)
        elif isinstance(self.range_t, (tuple, list)):
            if self.scheme == "uniform":
                self.t = torch.linspace(self.range_t[0], self.range_t[1], n_points, dtype=get_dtype())
            elif self.scheme == "random":
                gen_seed = _next_seed()
                if gen_seed is not None:
                    gen = torch.Generator().manual_seed(gen_seed)
                    self.t = torch.empty(n_points, dtype=get_dtype()).uniform_(self.range_t[0], self.range_t[1], generator=gen)
                else:
                    self.t = torch.empty(n_points, dtype=get_dtype()).uniform_(self.range_t[0], self.range_t[1])
            if self.expo_scaling:
                T1 = self.range_t[1]
                self.t = (1 + self.t)**(self.t/T1) - 1
        elif isinstance(self.range_t, (int, float)):
            self.t =  self.range_t * torch.ones_like(self.X, device=device)
        elif self.range_t is None:
            raise ValueError("Time range must be defined before sampling time coordinates.")

        self.T = self.t
        self.T_ = self.t.to(device).detach().requires_grad_()
        self.inputs_tensor_dict['t'] = self.T_

    def process_coordinates(self, device: Optional[torch.device] = None) -> Dict[str, Optional[torch.Tensor]]:
        """Return differentiable coordinate tensors on ``device``.

        ``None`` uses the configured device. Missing coordinates raise
        ``ValueError``.
        """
        if self.X is None or self.Y is None:    
            raise ValueError("Coordinates X and Y must be set before processing.")
        if self.range_t is not None: self.sampling_time()

        device = get_device() if device is None else device
        
        self.X_ = self.X.to(device).detach().requires_grad_()
        self.Y_ = self.Y.to(device).detach().requires_grad_()

        self.inputs_tensor_dict['x'] = self.X_
        self.inputs_tensor_dict['y'] = self.Y_

        if self.physics_type == "BC" or self.physics_type == "IC":
            self._prepare_target_outputs(device)

        return self.inputs_tensor_dict

    def _prepare_target_outputs(self, device: torch.device) -> None:
        from .nn import HardConstraint
        target_output_tensor_dict = {}

        for key, condition in self.condition_dict.items():
            if isinstance(condition, HardConstraint):continue
    
            if isinstance(condition, (float, int)):
                target_output_tensor_dict[key] = condition * torch.ones_like(self.X_, device=device)
            else:
                try:
                    variable_key, func = condition
                    input_var = self.inputs_tensor_dict[variable_key].detach().clone()
                    target_output_tensor_dict[key] = func(input_var)
                except (ValueError, IndexError, TypeError) as e:
                    raise ValueError(f"Invalid condition format for key '{key}'. Expected (var_name, func).") from e
        
        self.target_output_tensor_dict = target_output_tensor_dict


    def calc_output(self) -> Dict[str, torch.Tensor]:
        """Return cached predictions or derivatives for target conditions such as
        ``u_x``.
        """
        prediction_dict = self.model_outputs
        pred_dict = {}
        
        for key in self.target_output_tensor_dict:
            if '_' in key:
                var_name, grad_var = key.split('_')
                if var_name not in prediction_dict:
                    raise KeyError(f"Model output missing variable '{var_name}' required for condition '{key}'.")
                pred_dict[key] = calc_grad(prediction_dict[var_name], self.model_inputs[grad_var]) 
            else:
                pred_dict[key] = prediction_dict[key]
                
        return pred_dict
    
    def calc_loss(self, model: nn.Module) -> torch.Tensor:
        """Return the mean sum of squared residuals for ``model`` as a scalar tensor."""
        self.calc_residual_field(model)
        self.loss = torch.mean(self.residual_field_raw.square().sum(dim=0))
        if torch.isnan(self.loss):
            print("Warning: NaN loss encountered. Check model outputs and conditions.")
        return self.loss

    def calc_residual_field(self, model: nn.Module) -> Union[int, torch.Tensor]:
        """Return a 1D tensor of pointwise absolute residual magnitudes for
        ``model``.
        """
        self.process_model(model)
        return self._compute_residual_field()

    def _compute_residual_field(self) -> Union[int, torch.Tensor]:
        if self.physics_type in ["BC", "IC"]:
            if all(isinstance(cond, HardConstraint) for cond in self.condition_dict.values()):
                device = self.X_.device
                n_points = self.X_.shape[0]
                self.residual_field_raw = torch.zeros(1, n_points, device=device)
                self.residual_field = torch.zeros(n_points, device=device)
                return self.residual_field

            pred_dict = self.calc_output()
            self.residual_field_raw = torch.stack(tuple(pred_dict[key] - self.target_output_tensor_dict[key] for key in pred_dict), dim = 0)

        if  self.physics_type == "PDE":
            self.process_pde()
            self.residual_field_raw = self.PDE.calc_residual_field_raw()
        
        self.residual_field = self.residual_field_raw.abs().sum(dim=0)
        return self.residual_field

    def set_threshold(self, loss: float = None, top_k_loss: float = None) -> None:
        """Set mean-loss ``loss`` and high-residual selection ``top_k_loss``
        thresholds.
        """
        self.loss_threshold = loss
        self.top_k_loss_threshold = top_k_loss


    def save_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Save spatial coordinates and return copies as ``(x, y)`` tensors."""
        self.X_saved = self.X.clone()
        self.Y_saved = self.Y.clone()
        return self.X_saved, self.Y_saved

    def get_residual_based_points_topk(self, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the ``top_k`` highest-residual ``(x, y)`` tensors and append to
        buffers.
        """
        if isinstance(self.residual_field, (int, float)): 
            return torch.tensor([]), torch.tensor([])

        _, top_k_index = torch.topk(self.residual_field, top_k, dim=0)
        
        top_k_index = top_k_index.flatten().cpu()
        
        X_residual = self.X[top_k_index]
        Y_residual = self.Y[top_k_index]

        self.X_residual_container.append(X_residual)
        self.Y_residual_container.append(Y_residual)

        return X_residual, Y_residual
    
    def get_residual_based_points_threshold(self, threshold: float = None, maintain_points = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return selected ``(x, y)`` tensors and store them in residual buffers.

        Args:
            threshold: Residual cutoff; ``None`` uses the field mean.
            maintain_points: Avoid re-adding points during accumulating
                resampling.
        """
        if threshold is None: threshold = torch.mean(self.residual_field).item()

        if isinstance(self.residual_field, (int, float)): 
            return torch.tensor([]), torch.tensor([])
        
        mask = (self.residual_field > threshold)
        if self.X_residual_container and maintain_points:
            mask[:(self.residual_field.shape[0] - self._amounts_before_add)] = False
        mask = mask.cpu()

        X_residual = self.X[mask]
        Y_residual = self.Y[mask]

        self.X_residual_container = [X_residual]
        self.Y_residual_container = [Y_residual]

        return X_residual, Y_residual
    
    def apply_residual_based_points(self) -> None:
        """Incorporate residual-based sampled points into the training set."""
        if not self.X_residual_container or not self.Y_residual_container:
            return
        self._amounts_before_add = len(self.X)
        self.X = torch.cat(self.X_residual_container + [self.X], dim=0)
        self.Y = torch.cat(self.Y_residual_container + [self.Y], dim=0)

    def clear_residual_based_points(self) -> None:
        """Clear buffered residual-based points."""
        self.X_residual_container.clear()
        self.Y_residual_container.clear()


    def process_model(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Run ``model`` on ``inputs_tensor_dict``; cache and return its output
        tensors.
        """
        self.model_inputs = self.inputs_tensor_dict
        self.model_outputs = model(self.inputs_tensor_dict)
        return self.model_outputs
        
    def process_pde(self) -> None:
        """Compute PDE residuals from the cached inputs and outputs."""
        self.PDE.compute_residuals(inputs_dict = self.model_inputs | self.model_outputs)

    def evaluate(self, model: nn.Module):
        """Return an ``Evaluator`` for this geometry and PINN ``model``."""
        from .evaluation import Evaluator
        return Evaluator(model, self)
    
def function(input_key: str, function: Callable) -> List[Union[str, Callable]]:
    """Return ``[input_key, function]`` for use in ``define_bc`` or ``define_ic``.

    ``input_key`` selects the coordinate passed to the target-value callable.
    """
    return [input_key, function]
func = function


def parabolic_func(
    input_key: str,
    width,
    max_val,
    center_distance=0,
) -> List[Union[str, Callable]]:
    """Return a parabolic condition as ``[input_key, callable]``.

    Args:
        input_key: Coordinate passed to the callable.
        width: Parabola width.
        max_val: Value at the center.
        center_distance: Center coordinate.
    """
    return [input_key, lambda x: (-4*max_val/width**2)*(x - center_distance)**2 + max_val]
parabolic = parabolic_func
