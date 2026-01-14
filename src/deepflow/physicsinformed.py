from typing import Dict, Optional, Tuple, Union, Callable, Any

import torch
import torch.nn as nn

# Assuming these modules exist in your package structure
from .neuralnetwork import HardConstraint
from .pde import PDE
from .utility import calc_grad, get_device

class PhysicsAttach:
    """
    Manages physics constraints, boundary conditions, and data sampling 
    for Physics-Informed Neural Networks (PINNs).
    """

    def __init__(self):
        self.is_sampled: bool = False
        self.physics_type: Optional[str] = None
        self.range_t: Optional[Union[Tuple[float, float], torch.Tensor]] = None
        self.is_converged: bool = False
        
        # Internal state placeholders
        self.condition_dict: Dict = {}
        self.condition_num: int = 0
        self.t: Optional[torch.Tensor] = None
        self.PDE: Optional[PDE] = None
        
        # Coordinate placeholders
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None
        self.X_: Optional[torch.Tensor] = None
        self.Y_: Optional[torch.Tensor] = None
        self.T_: Optional[torch.Tensor] = None
        
        # Data dictionaries
        self.inputs_tensor_dict: Dict[str, Optional[torch.Tensor]] = {}
        self.target_output_tensor_dict: Dict[str, torch.Tensor] = {}
        self.model_inputs: Dict = {}
        self.model_outputs: Dict = {}
        
        # Loss tracking
        self.loss_field: Union[int, torch.Tensor] = 0
        self.loss_threshold: Optional[float] = None
        self.top_k_loss_threshold: Optional[float] = None

    # --------------------------------------------------------------------------
    # Condition Definitions
    # --------------------------------------------------------------------------

    def define_bc(self, condition_dict: Dict[str, Any], range_t: Optional[Tuple[float, float]] = None) -> None:
        """
        Define Boundary Conditions (BC).

        Args:
            condition_dict: Dictionary mapping variable names to conditions.
            range_t: Tuple of (min_t, max_t) or None.
        """
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.range_t = range_t
        self.physics_type = "BC"
    
    def define_ic(self, condition_dict: Dict[str, Any], t: float = 0.0) -> None:
        """
        Define Initial Conditions (IC).

        Args:
            condition_dict: Dictionary mapping variable names to conditions.
            t: The time instance for the initial condition.
        """
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.t = torch.tensor(t) if not isinstance(t, torch.Tensor) else t
        self.physics_type = "IC"
    
    def define_pde(self, pde_class: PDE, range_t: Optional[Tuple[float, float]] = None) -> None:
        """
        Define the Partial Differential Equation (PDE) to enforce.

        Args:
            pde_class: Instance or class of the PDE physics module.
            range_t: Tuple of (min_t, max_t) or None.
        """
        self.range_t = range_t
        self.PDE = pde_class
        self.physics_type = "PDE"

    # --------------------------------------------------------------------------
    # Data Preparation & Coordinate Processing
    # --------------------------------------------------------------------------

    def set_coordinates(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Manually set the spatial coordinates (required before processing).
        """
        self.X = x
        self.Y = y

    def sampling_time(self, n_points: int, random: bool = False) -> None:
        """
        Generate time coordinates based on the defined time range.
        """
        if self.range_t is None:
            self.t = None
            return

        if random:
            self.t = torch.empty(n_points).uniform_(self.range_t[0], self.range_t[1])
        else:
            # linspace creates a 1D tensor; [None] adds a batch dimension if needed
            self.t = torch.linspace(self.range_t[0], self.range_t[1])[None]

    def process_coordinates(self, device: Optional[torch.device] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        Prepare coordinate data and move to the specified device for PINN training.
        
        Args:
            device: ""'cuda' or 'cpu'. If None, auto-detects"".
        """
        if self.X is None or self.Y is None:    
            raise ValueError("Coordinates X and Y must be set before processing.")

        device = get_device() if device is None else device
        
        # Enable gradients for physics calculation (autograd)
        self.X_ = self.X.to(device).requires_grad_()
        self.Y_ = self.Y.to(device).requires_grad_()

        if self.range_t:
            if self.t is None:
                raise ValueError("Time range is set but time 't' has not been sampled.")
            self.T_ = self.t.to(device).requires_grad_()
            self.inputs_tensor_dict = {'x': self.X_, 'y': self.Y_, 't': self.T_}
        else:
            self.inputs_tensor_dict = {'x': self.X_, 'y': self.Y_, 't': None}

        # Pre-calculate target values for BC/IC
        if self.physics_type in ["IC", "BC"]:
            self._prepare_target_outputs(device)

        return self.inputs_tensor_dict

    def _prepare_target_outputs(self, device: torch.device) -> None:
        """Internal helper to prepare target tensors for BC/IC."""
        from .neuralnetwork import HardConstraint
        target_output_tensor_dict = {}

        for key, condition in self.condition_dict.items():
            if isinstance(condition, HardConstraint):continue
    
            if isinstance(condition, (float, int)):
                # Constant condition
                target_output_tensor_dict[key] = condition * torch.ones_like(self.X_, device=device)
            else:
                # Function-based condition (tuple: (variable_key, function))
                try:
                    variable_key, func = condition
                    input_var = self.inputs_tensor_dict[variable_key].detach().clone()
                    target_output_tensor_dict[key] = func(input_var)
                except (ValueError, IndexError, TypeError) as e:
                    raise ValueError(f"Invalid condition format for key '{key}'. Expected (var_name, func).") from e
        
        self.target_output_tensor_dict = target_output_tensor_dict

    # --------------------------------------------------------------------------
    # Model Execution & Loss Calculation
    # --------------------------------------------------------------------------

    def calc_output(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """
        Post-process the model's output to match target conditions.
        Handles derivative constraints (e.g., if key is 'u_x').
        """
        prediction_dict = model(self.inputs_tensor_dict)
        pred_dict = {}
        
        for key in self.target_output_tensor_dict:
            if '_' in key:
                # Example: key='u_x' -> compute grad(u, x)
                var_name, grad_var = key.split('_')
                if var_name not in prediction_dict:
                    raise KeyError(f"Model output missing variable '{var_name}' required for condition '{key}'.")
                pred_dict[key] = calc_grad(prediction_dict[var_name], self.inputs_tensor_dict[grad_var]) 
            else:
                pred_dict[key] = prediction_dict[key]
                
        return pred_dict
    
    def calc_loss(self, model: nn.Module, loss_fn: Callable = nn.MSELoss()) -> torch.Tensor:
        """
        Calculate the total loss for the current physics type.
        """
        if self.physics_type in ["IC", "BC"]:
            loss = 0.0
            # If all conditions are HardConstraints, the loss is structurally zero
            if all(isinstance(cond, HardConstraint) for cond in self.condition_dict.values()):
                return

            pred_dict = self.calc_output(model)
            
            for key in pred_dict:
                loss += loss_fn(pred_dict[key], self.target_output_tensor_dict[key])
            return loss
        
        else:
            # PDE Loss
            self.process_model(model)
            self.process_pde()
            return self.PDE.calc_loss()

    def calc_loss_field(self, model: nn.Module) -> Union[int, torch.Tensor]:
        """
        Calculate the element-wise loss field (absolute error or residual).
        """
        loss_field = 0

        if self.physics_type in ["IC", "BC"]:
            pred_dict = self.calc_output(model)
            for key in pred_dict:
                loss_field += torch.abs(pred_dict[key] - self.target_output_tensor_dict[key])
        
        elif self.physics_type == "PDE":
            self.process_model(model)
            self.process_pde()
            loss_field = self.PDE.calc_residual_field()

        self.loss_field = loss_field
        return loss_field

    def set_threshold(self, loss: float = None, top_k_loss: float = None) -> None:
        """Set loss thresholds for adaptive sampling or convergence checks."""
        self.loss_threshold = loss
        self.top_k_loss_threshold = top_k_loss

    def sampling_residual_based(self, top_k: int, model: nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adaptive sampling: Add points where the residual loss is highest.
        """
        # Calculate loss field (Consider caching this if called repeatedly in one step)
        self.calc_loss_field(model)
        
        if isinstance(self.loss_field, (int, float)): 
            # Loss field not calculated or zero
            return torch.tensor([]), torch.tensor([])

        # Select top K points with highest error
        _, top_k_index = torch.topk(self.loss_field, top_k, dim=0)
        
        # Flatten and move to CPU for indexing data
        top_k_index = top_k_index.flatten().cpu()
        
        # Append new points to existing dataset
        # Note: This increases dataset size indefinitely; consider memory management
        new_X = self.X[top_k_index]
        new_Y = self.Y[top_k_index]
        
        self.X = torch.cat([self.X, new_X])
        self.Y = torch.cat([self.Y, new_Y])

        return new_X, new_Y

    # --------------------------------------------------------------------------
    # PDE Processing Helpers
    # --------------------------------------------------------------------------

    def process_model(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Feed inputs to the model and cache outputs."""
        self.model_inputs = self.inputs_tensor_dict
        self.model_outputs = model(self.inputs_tensor_dict)
        return self.model_outputs
        
    def process_pde(self) -> None:
        """Pass model inputs and outputs to the PDE engine."""
        self.PDE.compute_residuals(inputs_dict = self.model_inputs | self.model_outputs)

    def evaluate(self, model: nn.Module):
        """Initialize evaluation module."""
        from .evaluation import Evaluator # Import inside method to avoid circular dependency if Evaluation imports PhysicsAttach
        return Evaluator(model, self)