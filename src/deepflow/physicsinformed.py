"""Physics constraints and coordinate processing attached to geometries."""

from typing import Dict, List, Optional, Tuple, Union, Callable, Any
import warnings
from copy import deepcopy

import torch
import torch.nn as nn

# Assuming these modules exist in your package structure
from .nn import HardConstraint
from .pde import PDE
from .utility import calc_grad, get_device, get_dtype, _next_seed

class PhysicsAttach:
    """Manage constraints, residuals, and samples for a PINN geometry.

    The class is inherited by geometry objects and normally used through
    methods such as ``define_bc``, ``define_pde``, and
    ``process_coordinates``.

    Attributes:
        physics_type: Attached physics kind: ``"BC"``, ``"IC"``, or ``"PDE"``.
        inputs_tensor_dict: Coordinate tensors used by the model.
        model_outputs: Most recently cached model predictions.
        residual_field: Pointwise absolute residual magnitude.
    """

    def __init__(self):
        """Initialize empty physics and coordinate state."""
        self.is_sampled: bool = False
        self.physics_type: Optional[list[str]] = []
        self.range_t: Optional[Union[Tuple[float, float], torch.Tensor]] = None
        self.is_converged: bool = False
        
        # Internal state placeholders
        self.condition_dict: Optional[Dict] = None
        self.condition_num: int = 0
        self.t: Optional[torch.Tensor] = None
        self.PDE: Optional[PDE] = None
        
        # Coordinate placeholders
        self.X: Optional[torch.Tensor] = None
        self.Y: Optional[torch.Tensor] = None
        self.X_: Optional[torch.Tensor] = None
        self.Y_: Optional[torch.Tensor] = None
        self.T_: Optional[torch.Tensor] = None
        self.X_residual_container: list[torch.Tensor] = []
        self.Y_residual_container: list[torch.Tensor] = []
        self._amounts_before_add: int = 0

        # Time
        self.expo_scaling = None
        self.scheme = None
        
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

    def define_bc(self, condition_dict: Dict[str, Any]) -> None:
        """
        Define Boundary Conditions (BC).

        Args:
            condition_dict: Dictionary mapping variable names to conditions.

        Returns:
            None. The geometry is marked as a boundary-condition geometry.
        """
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.physics_type = "BC"
    
    def define_ic(self, condition_dict: Dict[str, Any]) -> None:
        """
        Define Initial Conditions (IC).

        Args:
            condition_dict: Dictionary mapping variable names to conditions.

        Returns:
            None. The geometry is marked as an initial-condition geometry.
        """
        self.condition_dict = condition_dict
        self.condition_num = len(condition_dict)
        self.physics_type = "IC"
    
    def define_pde(self, pde: PDE) -> None:
        """
        Define the Partial Differential Equation (PDE) to enforce.

        Args:
            pde: Initialized ``deepflow.pde.PDE`` instance.

        Returns:
            None. The geometry is marked as a PDE geometry.
        """
        self.PDE = pde
        self.physics_type = "PDE"

    # --------------------------------------------------------------------------
    # Data Preparation & Coordinate Processing
    # --------------------------------------------------------------------------

    def set_coordinates(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Set spatial coordinates manually.

        Args:
            x: Tensor of x coordinates.
            y: Tensor of y coordinates with the same shape as ``x``.

        Returns:
            None. Coordinates are stored until ``process_coordinates`` is
            called.
        """
        self.X = x
        self.Y = y
    
    def define_time(self, range_t: Union[Tuple[float, float], int, float] = None, sampling_scheme: str = None, expo_scaling = None) -> None:
        """
        Configure the time coordinate range and sampling scheme.

        Args:
            range_t: A ``(start, end)`` interval or a fixed time value.
            sampling_scheme: ``"uniform"`` or ``"random"`` for intervals.
            expo_scaling: Whether to apply exponential time scaling.

        Raises:
            ValueError: If no range is supplied and no previous range exists.
        """
        # Handle args: Define or use existing range_t, scheme, expo_scaling
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
        """
        Generate time coordinates from the configured time range.

        Returns:
            None. The generated values are stored in ``t`` and ``T``.

        Raises:
            ValueError: If spatial coordinates or a time range are missing.
        """
        if self.X is None: raise ValueError("Before sampling time t, X coordinate must be sampling first")
        n_points = len(self.X)
        device = get_device()

        # Generate time coordinates
        if self.physics_type == "IC":
            # For IC, time is always zero
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
        # .detach() ensures T_ is always a fresh leaf tensor, even when
        # self.t was previously given requires_grad in-place by an earlier
        # process_coordinates call (which makes .to(device) return the same
        # tensor).  Without .detach(), R3 resampling can produce a non-leaf
        # T_ carrying a stale grad_fn, which breaks the next backward().
        self.T_ = self.t.to(device).detach().requires_grad_()
        self.inputs_tensor_dict['t'] = self.T_

    def process_coordinates(self, device: Optional[torch.device] = None) -> Dict[str, Optional[torch.Tensor]]:
        """
        Prepare coordinate tensors for PINN training.
        
        Args:
            device: Target device. If ``None``, use DeepFlow's configured
                device.

        Returns:
            Dictionary of differentiable model-input tensors.

        Raises:
            ValueError: If spatial coordinates are not set.
        """
        if self.X is None or self.Y is None:    
            raise ValueError("Coordinates X and Y must be set before processing.")
        if self.range_t is not None: self.sampling_time()

        device = get_device() if device is None else device
        
        # .detach() ensures X_/Y_ are always fresh leaf tensors, even when
        # self.X/self.Y were previously given requires_grad in-place by an
        # earlier process_coordinates call (which makes .to(device) return
        # the same tensor).  Without .detach(), R3 resampling can produce
        # non-leaf X_/Y_ carrying a stale grad_fn from the previous graph,
        # which breaks the next backward().
        self.X_ = self.X.to(device).detach().requires_grad_()
        self.Y_ = self.Y.to(device).detach().requires_grad_()

        self.inputs_tensor_dict['x'] = self.X_
        self.inputs_tensor_dict['y'] = self.Y_

        # Pre-calculate target values for BC/IC
        if self.physics_type == "BC" or self.physics_type == "IC":
            self._prepare_target_outputs(device)

        return self.inputs_tensor_dict

    def _prepare_target_outputs(self, device: torch.device) -> None:
        """Internal helper to prepare target tensors for BC/IC."""
        from .nn import HardConstraint
        target_output_tensor_dict = {}

        for key, condition in self.condition_dict.items():
            if isinstance(condition, HardConstraint):
                condition = condition.constant
    
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

    def calc_output(self) -> Dict[str, torch.Tensor]:
        """
        Post-process cached model outputs to match target conditions.
        Handles derivative constraints (e.g., if key is 'u_x').

        Uses ``self.model_inputs`` (not ``self.inputs_tensor_dict``) for
        derivative computation so that batched forward passes work correctly:
        ``model_inputs`` points to the tensors actually fed to the model (which
        are in the autograd graph), whereas ``inputs_tensor_dict`` may hold the
        original per-geometry tensors that are not in a batched graph.

        Returns:
            Predictions or derivatives corresponding to the configured target
            conditions.
        """
        prediction_dict = self.model_outputs
        pred_dict = {}
        
        for key in self.target_output_tensor_dict:
            if '_' in key:
                # Example: key='u_x' -> compute grad(u, x)
                var_name, grad_var = key.split('_')
                if var_name not in prediction_dict:
                    raise KeyError(f"Model output missing variable '{var_name}' required for condition '{key}'.")
                pred_dict[key] = calc_grad(prediction_dict[var_name], self.model_inputs[grad_var]) 
            else:
                pred_dict[key] = prediction_dict[key]
                
        return pred_dict
    
    def calc_loss(self, model: nn.Module) -> torch.Tensor:
        """
        Calculate the mean per-point sum of squared residual components.

        Args:
            model: Model used to produce predictions.

        Returns:
            Scalar loss tensor.
        """
        self.calc_residual_field(model)
        self.loss = torch.mean(self.residual_field_raw.square().sum(dim=0))
        if torch.isnan(self.loss):
            print("Warning: NaN loss encountered. Check model outputs and conditions.")
        return self.loss

    def calc_residual_field(self, model: nn.Module) -> Union[int, torch.Tensor]:
        """
        Calculate the pointwise absolute residual field.

        Args:
            model: Model used to produce predictions.

        Returns:
            One-dimensional tensor containing the absolute residual magnitude
            at each sample point.
        """
        self.process_model(model)
        return self._compute_residual_field()

    def _compute_residual_field(self) -> Union[int, torch.Tensor]:
        """
        Compute the residual field from cached ``model_inputs`` / ``model_outputs``.

        This is the residual-computation half of ``calc_residual_field``,
        extracted so that callers which have already run the forward pass
        (e.g. batched loss in ``ProblemDomain``) can skip the redundant
        ``process_model`` call.
        """
        if self.physics_type in ["BC", "IC"]:
            pred_dict = self.calc_output()
            self.residual_field_raw = torch.stack(tuple(pred_dict[key] - self.target_output_tensor_dict[key] for key in pred_dict), dim = 0)

        if  self.physics_type == "PDE":
            self.process_pde()
            self.residual_field_raw = self.PDE.calc_residual_field_raw()
        
        self.residual_field = self.residual_field_raw.abs().sum(dim=0)
        return self.residual_field

    def set_threshold(self, loss: float = None, top_k_loss: float = None) -> None:
        """Set thresholds used by adaptive sampling or convergence checks.

        Args:
            loss: Mean-loss threshold.
            top_k_loss: Threshold for selecting high-residual points.
        """
        self.loss_threshold = loss
        self.top_k_loss_threshold = top_k_loss

    # --------------------------------------------------------------------------
    # Residual-Based Adaptive Sampling Related
    # --------------------------------------------------------------------------

    def save_coordinates(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Save the current spatial coordinates for later restoration.

        Returns:
            Tuple of saved x and y tensors.
        """
        self.X_saved = self.X.clone()
        self.Y_saved = self.Y.clone()
        return self.X_saved, self.Y_saved

    def get_residual_based_points_topk(self, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select points with the largest residual magnitudes.

        Args:
            top_k: Number of points to select.

        Returns:
            Tuple of selected x and y tensors. The points are also appended to
            the residual-point buffers.
        """
        if isinstance(self.residual_field, (int, float)): 
            # Loss field not calculated or zero
            return torch.tensor([]), torch.tensor([])

        # Select top K points with highest error
        _, top_k_index = torch.topk(self.residual_field, top_k, dim=0)
        
        # Flatten and move to CPU for indexing data
        top_k_index = top_k_index.flatten().cpu()
        
        X_residual = self.X[top_k_index]
        Y_residual = self.Y[top_k_index]

        self.X_residual_container.append(X_residual)
        self.Y_residual_container.append(Y_residual)

        return X_residual, Y_residual
    
    def get_residual_based_points_threshold(self, threshold: float = None, maintain_points = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Select points whose residual exceeds a threshold.

        Args:
            threshold: Residual threshold. If ``None``, use the field mean.
            maintain_points: Avoid re-adding points already selected during an
                accumulating resampling cycle.

        Returns:
            Tuple of selected x and y tensors stored in the residual buffers.
        """
        if threshold is None: threshold = torch.mean(self.residual_field).item()

        if isinstance(self.residual_field, (int, float)): 
            # Loss field not calculated or zero
            return torch.tensor([]), torch.tensor([])
        
        mask = (self.residual_field > threshold)
        if self.X_residual_container and maintain_points:
            mask[:(self.residual_field.shape[0] - self._amounts_before_add)] = False  # Avoid adding previously added points
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

    # --------------------------------------------------------------------------
    # PDE Processing Helpers
    # --------------------------------------------------------------------------

    def process_model(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Run a model on the processed inputs and cache its outputs.

        Args:
            model: Model accepting ``inputs_tensor_dict``.

        Returns:
            Dictionary of model output tensors.
        """
        self.model_inputs = self.inputs_tensor_dict
        self.model_outputs = model(self.inputs_tensor_dict)
        return self.model_outputs
        
    def process_pde(self) -> None:
        """Compute PDE residuals from the cached inputs and outputs."""
        self.PDE.compute_residuals(inputs_dict = self.model_inputs | self.model_outputs)

    def evaluate(self, model: nn.Module):
        """Create an evaluator for this geometry and model.

        Args:
            model: PINN model used for prediction and residual evaluation.

        Returns:
            A ``deepflow.evaluation.Evaluator`` instance.
        """
        from .evaluation import Evaluator # Import inside method to avoid circular dependency if Evaluation imports PhysicsAttach
        return Evaluator(model, self)
    
def function(input_key: str, function: Callable) -> List[Union[str, Callable]]:
    """Package a coordinate key and callable boundary condition.

    Args:
        input_key: Coordinate name passed to ``function``.
        function: Callable producing the target value.

    Returns:
        Two-item list ``[input_key, function]`` accepted by ``define_bc`` or
        ``define_ic``.
    """
    return [input_key, function]
func = function


def parabolic_func(
    input_key: str,
    width,
    max_val,
    center_distance=0,
) -> List[Union[str, Callable]]:
    """Create a parabolic boundary-condition function descriptor.

    Args:
        input_key: Coordinate name used by the generated callable.
        width: Width parameter of the parabola.
        max_val: Value at the parabola center.
        center_distance: Coordinate of the center.

    Returns:
        Two-item list compatible with ``function``.
    """
    return [input_key, lambda x: (-4*max_val/width**2)*(x - center_distance)**2 + max_val]
parabolic = parabolic_func
