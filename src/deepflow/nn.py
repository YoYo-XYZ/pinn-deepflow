import copy
import platform
from typing import List, Dict, Callable, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .utility import get_device

class HardConstraint:
    """
    Defines a hard constraint for the PINN.
    """
    def __init__(self, constant: float = 0.0):
        self.constant = constant

    @staticmethod
    def define_zero_func(bound):
        def zero_func(coords):
            return coords[bound.axes_sec[0]] - bound.funcs[bound.ax][0](coords[bound.ax])
        return zero_func

    def __str__(self):
        return str(self.constant)

hard_constraint = lambda constant = 0: HardConstraint(constant)

class NN(ABC, nn.Module):
    """
    Physics-Informed Neural Network (PINN) model.

    A feedforward neural network that takes input coordinates (e.g., x, y, t)
    and outputs physical quantities (e.g., u, v, p), with support for 
    hard constraints.
    """

    def __init__(
        self,
        input_vars: Optional[List[str]] = None, 
        output_vars: Optional[List[str]] = None,
    ):
        """
        Args:
            input_vars (list): List of input variable names (e.g., ['x', 'y']).
            output_vars (list): List of output variable names (e.g., ['u']).
        """
        super().__init__()
        
        # Handle mutable default arguments
        self.input_keys = input_vars if input_vars is not None else ['x', 'y']
        self.output_keys = output_vars if output_vars is not None else ['u', 'v', 'p']
        
        self.input_num = len(self.input_keys)
        self.output_num = len(self.output_keys)
        
        # Hard constraint containers
        self.hard_constraints: Optional[Dict] = None
        self.hard_constants: Optional[Dict] = None

        self._init_history()
        #self._build_network()

    @abstractmethod
    def _build_network(self):
        """Define the network architecture in subclasses."""
        pass

    def _init_history(self):
        """Initializes the loss history dictionary."""
        base_keys = ['total_loss', 'bc_loss', 'pde_loss']
        if 't' in self.input_keys:
            base_keys.append('ic_loss')
            
        self.loss_history = {key: [] for key in base_keys}

    def forward(self, inputs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            inputs_dict: Dictionary containing input tensors (e.g., {'x': tensor, 'y': tensor})
        """
        # Efficient stacking
        input_tensor = torch.stack([inputs_dict[key] for key in self.input_keys], dim=1)
        
        pred = self.net(input_tensor)

        output_dict = {}

        for i, key in enumerate(self.output_keys):
            val = pred[:, i]
            
            # Apply hard constraints if they exist for this variable
            if self.hard_constraints and key in self.hard_constraints:
                coords = {0: inputs_dict.get("x"), 1: inputs_dict.get("y")}
                constraint_mod = self.hard_constraints[key](coords)
                val = constraint_mod * val + self.hard_constants[key]
                
            output_dict[key] = val

        return output_dict

    def apply_hard_constraints(self, bound_list: list):
        """
        Configures hard constraints based on a list of boundary conditions.
        """
        self.hard_constraints = {}
        self.hard_constants = {}
        
        for key in self.output_keys:
            # Filter bounds relevant to this output key that are HardConstraints
            relevant_bounds = [b for b in bound_list if isinstance(b.condition_dict.get(key), HardConstraint)]

            if relevant_bounds:
                # Check if the constants of relevant bounds are all the same
                constants_list = [b.condition_dict[key].constant for b in relevant_bounds]
                if not all(c == constants_list[0] for c in constants_list):
                    raise ValueError(f"Conflicting hard constraint constants for output '{key}'. All must be the same.")
                
                # Take constant from the first bound
                self.hard_constants[key] = relevant_bounds[0].condition_dict[key].constant

                # Create closure for the constraint functiont(key)
                def constraint_func(coords, bounds = relevant_bounds):
                    result = 1.0
                    for bound in bounds:
                        zero_func = HardConstraint.define_zero_func(bound)
                        result *= zero_func(coords)
                    return result

                self.hard_constraints[key] = constraint_func

    def _record_loss(self, loss_dict: Dict[str, torch.Tensor]):
        """Helper to append current losses to history."""
        for key, val in loss_dict.items():
            value_to_store = val.detach().item() if isinstance(val, torch.Tensor) else val
            if key in self.loss_history: self.loss_history[key].append(value_to_store)

    def print_status(self):
        """Prints the current training status."""
        string_parts = [f"Epoch: {len(self.loss_history['total_loss'])}"]
        for k, v in self.loss_history.items():
            if v:
                string_parts.append(f"{k}: {v[-1]:.5f}")
        print(", ".join(string_parts))
    # ------------------------------------------------------------------
    # Training Methods
    # ------------------------------------------------------------------

    def train_adam(
        self, 
        learning_rate: float, 
        epochs: int, 
        calc_loss: Callable, 
        use_scheduler: Optional[bool] = False, 
        print_every: int = 200, 
        threshold_loss: Optional[float] = None,
        do_between_epochs: Optional[Callable] = None
    )-> tuple['NN', 'NN']:
        """
        Trains the model using the Adam optimizer.
        """
        model = copy.deepcopy(self.to(get_device()))

        model.train() # Set to training mode
                
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        scheduler = None
        if use_scheduler:
            # Allow custom scheduler config or default
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, epochs//20, gamma=0.9)

        best_loss = float('inf')
        try:
            for epoch in range(1,epochs+1):
                optimizer.zero_grad(set_to_none=True)
                
                loss_dict = calc_loss(model)
                total_loss = loss_dict['total_loss']
                if torch.isnan(total_loss):
                    print("Detected NaN in loss. Stop the training.")
                    break
                total_loss_num = total_loss.item()
                
                total_loss.backward()
                optimizer.step()
                
                if scheduler: scheduler.step()
                
                model._record_loss(loss_dict)

                # Save best state
                if total_loss_num < best_loss:
                    best_loss = total_loss_num
                    best_model = copy.deepcopy(model)
                    
                    if threshold_loss and best_loss < threshold_loss:
                        print(f"Stop: Loss {best_loss:.5f} < Threshold {threshold_loss}")
                        break

                if epoch % print_every == 0 or epoch == 1:
                    model.print_status()
                
                if do_between_epochs: do_between_epochs(epoch, model)

        except KeyboardInterrupt:
            print('Training interrupted by user.')
            return model, best_model
            
        model.print_status()
        return model, best_model

    def train_lbfgs(
        self, 
        epochs: int, 
        calc_loss: Callable, 
        print_every: int = 50, 
        threshold_loss: Optional[float] = None,
        do_between_epochs: Optional[Callable] = None,
    ) -> tuple['NN', 'NN']:
        """
        Trains the model using the L-BFGS optimizer.
        
        Returns:
            tuple: (model, best_model) — the final model and the model with the lowest loss.
        """
        model = copy.deepcopy(self.to(get_device()))

        model.train()

        # Strong Wolfe line search is standard for PINNs
        optimizer = torch.optim.LBFGS(
            model.parameters(), 
            history_size=100, 
            max_iter=20, 
            line_search_fn="strong_wolfe"
        )

        best_loss = float('inf')
        best_model = copy.deepcopy(model)

        try:
            for epoch in range(epochs):
                # Container to extract loss from closure
                loss_dict_container = {}
                nan_detected = False

                def closure():
                    nonlocal nan_detected
                    optimizer.zero_grad(set_to_none=True)
                    loss_dict = calc_loss(model)
                    total_loss = loss_dict['total_loss']
                    if torch.isnan(total_loss):
                        print("Detected NaN in loss. Stop the training.")
                        nan_detected = True
                        return 0.0
                    # retain_graph=True is required because:
                    # 1. calc_loss stores residual field tensors with grad_fn
                    #    on geometry objects (via calc_grad with create_graph=True).
                    # 2. LBFGS line search calls closure() multiple times per step
                    #    with the same underlying graph context (the same input
                    #    leaf tensors are reused across trial evaluations). When
                    #    create_graph=True is used for 2nd-order PDE residuals,
                    #    autograd frees saved intermediate values after the first
                    #    backward(), breaking the next closure evaluation. Without
                    #    retaining the graph, the line search hits
                    #    "RuntimeError: Trying to backward through the graph a
                    #    second time".
                    total_loss.backward(retain_graph=True)
                    
                    loss_dict_container.update(loss_dict) # Store loss_dict in the container
                    return total_loss.detach()  # detach to avoid keeping graph alive after step
                
                optimizer.step(closure)
                if nan_detected:
                    # Reinitialize LBFGS optimizer with fresh state
                    optimizer = torch.optim.LBFGS(
                        model.parameters(),
                        history_size=100,
                        max_iter=20,
                        line_search_fn="strong_wolfe")
                else:   
                    # Record loss after the step
                    total_loss_num = loss_dict_container['total_loss'].item()
                    model._record_loss(loss_dict_container)

                    # Track best model
                    if total_loss_num < best_loss:
                        best_loss = total_loss_num
                        best_model = copy.deepcopy(model)

                if epoch % print_every == 0:
                    model.print_status()
                
                if threshold_loss and total_loss_num < threshold_loss:
                     print(f"Stop: Loss {total_loss_num:.5f} < Threshold {threshold_loss}")
                     break
                
                # Call do_between_epochs callback if provided
                if do_between_epochs: 
                    do_between_epochs(epoch, model)

        except KeyboardInterrupt:
            print('Training interrupted by user.')
            return model, best_model
        
        model.print_status()
        return model, best_model
    
    def save_as_pickle(self, file_name: str = "model.pkl") -> None:
        """Saves the model as a pickle file."""
        import pickle
        if file_name[-4:] != '.pkl': file_name += '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
    

def load_from_pickle(file_name: str) -> None:
    """Loads the model from a pickle file."""
    import pickle
    if file_name[-4:] != '.pkl': file_name += '.pkl'
    with open(file_name, 'rb') as f:
        return pickle.load(f)
    
class FNN(NN):
    def __init__(
        self,
        input_vars: Optional[List[str]] = None, 
        output_vars: Optional[List[str]] = None,
        hidden_layer: List[int] = [50, 50, 50, 50],
        activation: nn.Module = nn.Tanh()
    ):
        super().__init__(input_vars, output_vars)
        self.activation = activation
        self.hidden_layer = hidden_layer
        self._build_network()

    def _build_network(self) -> None:
        """Builds the feedforward neural network architecture."""
        self.layer_list = [self.input_num] + self.hidden_layer + [self.output_num]

        layers = []
        # Add all hidden layers with activations
        for i in range(len(self.layer_list) - 2):
            layers.append(nn.Linear(self.layer_list[i], self.layer_list[i+1]))
            layers.append(self.activation)
            
        # Add the final output layer without activation
        layers.append(nn.Linear(self.layer_list[-2], self.layer_list[-1]))

        self.net = nn.Sequential(*layers)

class PINN(FNN):
    def __init__(
        self,
        input_vars: Optional[List[str]] = None, 
        output_vars: Optional[List[str]] = None,
        width: int = 32,
        length: int = 4,
        activation: nn.Module = nn.Tanh()
    ):
        super().__init__(input_vars, output_vars, [width for _ in range(length)], activation)