import copy
from typing import List, Dict, Callable, Optional, Tuple, Union

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

class PINN(nn.Module):
    """
    Physics-Informed Neural Network (PINN) model.

    A feedforward neural network that takes input coordinates (e.g., x, y, t)
    and outputs physical quantities (e.g., u, v, p), with support for 
    hard constraints.
    """

    def __init__(
        self, 
        width: int, 
        length: int, 
        input_vars: Optional[List[str]] = None, 
        output_vars: Optional[List[str]] = None,
        activation: nn.Module = nn.Tanh()
    ):
        """
        Args:
            width (int): Number of neurons per hidden layer.
            length (int): Number of hidden layers.
            input_vars (list): List of input variable names (e.g., ['x', 'y']).
            output_vars (list): List of output variable names (e.g., ['u']).
            activation (nn.Module): Activation function class (default: nn.Tanh()).
        """
        super().__init__()
        
        # Handle mutable default arguments
        self.input_keys = input_vars if input_vars is not None else ['x', 'y']
        self.output_keys = output_vars if output_vars is not None else ['u', 'v', 'p']
        
        self.input_num = len(self.input_keys)
        self.output_num = len(self.output_keys)
        
        self.activation = activation
        
        # Hard constraint containers
        self.hard_constraints: Optional[Dict] = None
        self.hard_constants: Optional[Dict] = None

        self._init_history()
        self._build_network(width, length)

    def _build_network(self, width: int, length: int):
        layers = []
        # Input layer
        layers.append(nn.Linear(self.input_num, width))
        layers.append(self.activation)
        
        # Hidden layers
        for _ in range(length):
            layers.append(nn.Linear(width, width))
            layers.append(self.activation)
        
        # Output layer
        layers.append(nn.Linear(width, self.output_num))
        
        self.net = nn.Sequential(*layers)

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
                constraint_mod = self.hard_constraints[key](inputs_dict)
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

                # Create closure for the constraint function
                def make_constraint_func(coords):
                    result = 1.0
                    for bound in relevant_bounds:
                        zero_func = HardConstraint.define_zero_func(bound)
                        result *= zero_func(coords)
                    return result

                self.hard_constraints[key] = make_constraint_func

    def _record_loss(self, loss_dict: Dict[str, torch.Tensor]):
        """Helper to append current losses to history."""
        for key, val in loss_dict.items():
            value_to_store = val.detach().item() if isinstance(val, torch.Tensor) else val
            if key in self.loss_history: self.loss_history[key].append(value_to_store)

    def print_status(self):
        """Prints the current training status."""
        string_parts = [f"Epoch: {len(self.loss_history['total_loss'])}"]
        for k, v in self.loss_history.items():
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
        scheduler_config: Optional[Dict] = None, 
        print_every: int = 200, 
        threshold_loss: Optional[float] = None
    )-> tuple['PINN', 'PINN']:
        """
        Trains the model using the Adam optimizer.
        """
        model = copy.deepcopy(self.to(get_device()))
        model.train() # Set to training mode
                
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        scheduler = None
        if scheduler_config:
            # Allow custom scheduler config or default
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=100, min_lr=1e-4
            )

        best_loss = float('inf')
        try:
            for epoch in range(1,epochs+1):
                optimizer.zero_grad(set_to_none=True)
                
                loss_dict = calc_loss(model)
                total_loss = loss_dict['total_loss']
                total_loss_num = total_loss.item()
                
                total_loss.backward()
                optimizer.step()
                
                if scheduler: scheduler.step(total_loss_num)
                
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

        except KeyboardInterrupt:
            print('Training interrupted by user.')
            return model, best_model
            
        return model, best_model

    def train_lbfgs(
        self, 
        epochs: int, 
        calc_loss: Callable, 
        print_every: int = 200, 
        threshold_loss: Optional[float] = None
    ) -> 'PINN':
        """
        Trains the model using the L-BFGS optimizer.
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

        try:
            for epoch in range(epochs):
                # Container to extract loss from closure
                loss_dict_container = {}

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss_dict = calc_loss(model)
                    total_loss = loss_dict['total_loss']
                    total_loss.backward()
                    
                    loss_dict_container.update(loss_dict) # Store loss_dict in the container
                    return total_loss

                optimizer.step(closure)
                
                # Record loss after the step
                model._record_loss(loss_dict_container)
                total_loss_num = loss_dict_container['total_loss'].item()

                if epoch % print_every == 0:
                    model.print_status()
                
                if threshold_loss and total_loss_num < threshold_loss:
                     print(f"Stop: Loss {total_loss_num:.5f} < Threshold {threshold_loss}")
                     break

        except KeyboardInterrupt:
            print('Training interrupted by user.')
            return model

        return model
    
    def save_as_pickle(self, file_name: str = "model.pkl") -> None:
        """Saves the model as a pickle file."""
        import pickle
        if file_name[-4:] != '.pkl': file_name += '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)
    
    def load_from_pickle(self, file_name: str) -> None:
        """Loads the model from a pickle file."""
        import pickle
        if file_name[-4:] != '.pkl': file_name += '.pkl'
        with open(file_name, 'rb') as f:
            self = pickle.load(f)
