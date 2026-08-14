import copy
import math
import platform
from numbers import Real
from typing import List, Dict, Callable, Optional, Tuple, Union
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .utility import get_device, get_dtype


class _NonFiniteLossError(RuntimeError):
    """Internal signal used to stop optimizer closures immediately."""


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
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        """
        Args:
            input_vars (list): List of input variable names (e.g., ['x', 'y']).
            output_vars (list): List of output variable names (e.g., ['u']).
            weight_init: Weight initialization scheme for ``nn.Linear`` layers.
                Supported string aliases: ``'kaiming'``/``'he'`` (default),
                ``'xavier'``/``'glorot'``.  ``None`` leaves PyTorch's default
                initialization untouched.  A callable receives the model and
                can apply arbitrary initialization.
        """
        super().__init__()
        
        # Handle mutable default arguments
        self.input_keys = input_vars if input_vars is not None else ['x', 'y']
        self.output_keys = output_vars if output_vars is not None else ['u', 'v', 'p']
        
        self.input_num = len(self.input_keys)
        self.output_num = len(self.output_keys)
        
        self.weight_init = weight_init
        # Hard constraint containers
        self.hard_constraints: Optional[Dict] = None
        self.hard_constants: Optional[Dict] = None

        self._init_history()
        # Ensure subsequent nn.Linear/Parameter construction uses the
        # globally configured deepflow dtype (FP32 by default, FP64 when
        # the user sets df.dtype = torch.float64).
        torch.set_default_dtype(get_dtype())
        #self._build_network()

    @abstractmethod
    def _build_network(self):
        """Define the network architecture in subclasses."""
        pass

    def _init_weights(self) -> None:
        """
        Apply the selected weight initialization scheme to all ``nn.Linear``
        layers in the network. The scheme is controlled by ``self.weight_init``.

        Supported string aliases:
            - ``'kaiming'`` / ``'he'``: Kaiming uniform (PyTorch ``nn.Linear``
              default). Weights use ``a = sqrt(5)``; biases are uniform in
              ``[-1/sqrt(fan_in), 1/sqrt(fan_in)]``.
            - ``'xavier'`` / ``'glorot'``: Xavier normal. Biases are zero.

        Passing ``None`` skips initialization, leaving PyTorch's defaults. A
        callable receives the model instance and can apply a custom scheme.
        """
        if self.weight_init is None:
            return

        if callable(self.weight_init):
            self.weight_init(self)
            return

        init_name = str(self.weight_init).lower()
        if init_name not in ('kaiming', 'he', 'xavier', 'glorot'):
            raise ValueError(
                f"Unknown weight_init='{self.weight_init}'. "
                f"Supported values are: 'kaiming', 'he', 'xavier', 'glorot', "
                f"None, or a callable."
            )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_name in ('kaiming', 'he'):
                    nn.init.kaiming_uniform_(m.weight, a=5 ** 0.5)
                    if m.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                        bound = 1 / (fan_in ** 0.5) if fan_in > 0 else 0
                        nn.init.uniform_(m.bias, -bound, bound)
                else:  # xavier / glorot
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

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

    def _maybe_print_status(
        self,
        print_every: int,
        last_printed_epoch: Optional[int],
    ) -> Optional[int]:
        """Print on the shared, one-based history epoch schedule."""
        current_epoch = len(self.loss_history['total_loss'])
        if current_epoch > 0 and (current_epoch == 1 or current_epoch % print_every == 0):
            self.print_status()
            return current_epoch
        return last_printed_epoch
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
        do_between_epochs: Optional[Callable] = None,
        compile_model: bool = False,
        max_grad_norm: Optional[float] = 1.0,
    )-> tuple['NN', 'NN']:
        """
        Trains the model using the Adam optimizer.

        Args:
            compile_model: If ``True``, wraps the model with ``torch.compile``
                for kernel fusion and reduced overhead. Requires PyTorch 2.0+
                and the Triton backend (Linux). The first epoch will be slower
                due to compilation; subsequent epochs benefit from fused kernels.
            max_grad_norm: Maximum global gradient norm used for clipping. Set
                to ``None`` to disable gradient clipping.
        """
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if print_every <= 0:
            raise ValueError("print_every must be positive")

        model = copy.deepcopy(self).to(get_device())
        if compile_model:
            model = torch.compile(model)

        model.train() # Set to training mode

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = None
        if use_scheduler:
            # Allow custom scheduler config or default
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                max(1, epochs // 20),
                gamma=0.9,
            )

        best_loss = float('inf')
        best_state = None
        initial_history_len = len(model.loss_history['total_loss'])
        last_printed_epoch = None
        try:
            for epoch in range(1,epochs+1):
                optimizer.zero_grad(set_to_none=True)
                
                training_loss_dict = calc_loss(model)
                training_loss = training_loss_dict['total_loss']
                if not torch.isfinite(training_loss).all():
                    print("Detected non-finite loss. Stop the training.")
                    break
                
                training_loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=max_grad_norm
                    )
                optimizer.step()
                
                if scheduler: scheduler.step()

                # Evaluate the updated parameters so the recorded loss and
                # best-state snapshot always describe the same model.
                loss_dict = calc_loss(model)
                total_loss = loss_dict['total_loss']
                if not torch.isfinite(total_loss).all():
                    print("Detected non-finite loss. Stop the training.")
                    break
                total_loss_num = total_loss.item()
                
                model._record_loss(loss_dict)

                # Save best state (just parameters, not entire object graph)
                threshold_reached = False
                if total_loss_num < best_loss:
                    best_loss = total_loss_num
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

                    if threshold_loss is not None and best_loss < threshold_loss:
                        threshold_reached = True

                last_printed_epoch = model._maybe_print_status(print_every, last_printed_epoch)

                if threshold_reached:
                    print(f"Stop: Loss {best_loss:.5f} < Threshold {threshold_loss}")
                    break
                
                if do_between_epochs: do_between_epochs(epoch, model)

        except KeyboardInterrupt:
            print('Training interrupted by user.')
            
        # Reconstruct best model from saved parameters
        best_model = copy.deepcopy(model)
        if best_state is not None:
            best_model.load_state_dict(best_state)

        current_epoch = len(model.loss_history['total_loss'])
        if current_epoch > initial_history_len and current_epoch != last_printed_epoch:
            model.print_status()
        return model, best_model

    def train_lbfgs(
        self, 
        epochs: int, 
        calc_loss: Callable, 
        print_every: int = 50, 
        threshold_loss: Optional[float] = None,
        do_between_epochs: Optional[Callable] = None,
        compile_model: bool = False,
    ) -> tuple['NN', 'NN']:
        """
        Trains the model using the L-BFGS optimizer.

        Args:
            compile_model: If ``True``, wraps the model with ``torch.compile``
                for kernel fusion and reduced overhead. Requires PyTorch 2.0+.
        
        Returns:
            tuple: (model, best_model) — the final model and the model with the lowest loss.
        """
        if print_every <= 0:
            raise ValueError("print_every must be positive")

        model = copy.deepcopy(self).to(get_device())
        if compile_model:
            model = torch.compile(model)

        model.train()

        # Strong Wolfe line search is standard for PINNs
        optimizer = torch.optim.LBFGS(
            model.parameters(), 
            history_size=100, 
            max_iter=20, 
            line_search_fn="strong_wolfe"
        )

        best_loss = float('inf')
        best_state = None
        initial_history_len = len(model.loss_history['total_loss'])
        last_printed_epoch = None

        try:
            for epoch in range(1, epochs + 1):
                epoch_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

                def closure():
                    optimizer.zero_grad(set_to_none=True)
                    loss_dict = calc_loss(model)
                    total_loss = loss_dict['total_loss']
                    if not torch.isfinite(total_loss).all():
                        print("Detected non-finite loss. Stop the training.")
                        raise _NonFiniteLossError
                    # retain_graph=False (default): each closure call performs
                    # a fresh forward pass that builds a new graph. The LBFGS
                    # line search only consumes the scalar loss and parameter
                    # .grad attributes — it never traverses the autograd graph
                    # across calls. Freeing the graph immediately after backward()
                    # reduces peak memory (especially important with
                    # create_graph=True in calc_grad for 2nd-order PDE residuals).
                    #
                    # This requires process_coordinates() to produce fresh leaf
                    # tensors (via .detach().requires_grad_()) so that R3
                    # resampling doesn't leave stale grad_fn references on X_/Y_.
                    total_loss.backward()
                    return total_loss.detach()  # detach to avoid keeping graph alive after step
                
                try:
                    optimizer.step(closure)
                except _NonFiniteLossError:
                    model.load_state_dict(epoch_state)
                    break

                # Re-evaluate the accepted parameters. The last closure call is
                # not a reliable public contract for the optimizer's final
                # state, especially when a line search is used.
                loss_dict = calc_loss(model)
                total_loss = loss_dict['total_loss']
                if not torch.isfinite(total_loss).all():
                    print("Detected non-finite loss. Stop the training.")
                    model.load_state_dict(epoch_state)
                    break
                total_loss_num = total_loss.item()
                model._record_loss(loss_dict)

                # Track best model (just parameters, not entire object graph)
                if total_loss_num < best_loss:
                    best_loss = total_loss_num
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

                last_printed_epoch = model._maybe_print_status(print_every, last_printed_epoch)
                
                if threshold_loss is not None and total_loss_num < threshold_loss:
                     print(f"Stop: Loss {total_loss_num:.5f} < Threshold {threshold_loss}")
                     break
                
                # Call do_between_epochs callback if provided
                if do_between_epochs: 
                    do_between_epochs(epoch, model)

        except KeyboardInterrupt:
            print('Training interrupted by user.')
        
        # Reconstruct best model from saved parameters
        best_model = copy.deepcopy(model)
        if best_state is not None:
            best_model.load_state_dict(best_state)

        current_epoch = len(model.loss_history['total_loss'])
        if current_epoch > initial_history_len and current_epoch != last_printed_epoch:
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


class _JointRFFEmbedding(nn.Module):
    """Fixed random Fourier features over the joint input coordinates."""

    def __init__(self, input_dim: int, embed_dim: int, alpha: float):
        super().__init__()

        if (
            isinstance(embed_dim, bool)
            or not isinstance(embed_dim, int)
            or embed_dim <= 0
            or embed_dim % 2
        ):
            raise ValueError("embed_dim must be a positive even integer")
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, Real)
            or not math.isfinite(float(alpha))
            or alpha <= 0
        ):
            raise ValueError("alpha must be a positive finite scalar")

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.alpha = float(alpha)
        frequencies = torch.randn(
            input_dim,
            embed_dim // 2,
            dtype=get_dtype(),
        ) * self.alpha
        self.register_buffer("B", frequencies)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        projection = inputs @ self.B
        return torch.cat((torch.cos(projection), torch.sin(projection)), dim=-1)
    
class FNN(NN):
    def __init__(
        self,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        hidden_layer: List[int] = [50, 50, 50, 50],
        activation: nn.Module = nn.Tanh(),
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        super().__init__(input_vars, output_vars, weight_init=weight_init)
        self.activation = activation
        self.hidden_layer = hidden_layer
        self._build_network()
        self.to(get_dtype())

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
        self._init_weights()

class PINN(FNN):
    def __init__(
        self,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        width: int = 32,
        length: int = 4,
        activation: nn.Module = nn.Tanh(),
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        super().__init__(
            input_vars, output_vars,
            [width for _ in range(length)],
            activation,
            weight_init=weight_init,
        )


class RFFPINN(FNN):
    """PINN with a fixed joint random Fourier feature input embedding."""

    def __init__(
        self,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        width: int = 32,
        length: int = 4,
        embed_dim: int = 256,
        alpha: float = 5.0,
        activation: nn.Module = nn.Tanh(),
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        self.embed_dim = embed_dim
        self.alpha = alpha
        super().__init__(
            input_vars, output_vars,
            [width for _ in range(length)],
            activation,
            weight_init=weight_init,
        )

    def _build_network(self) -> None:
        """Build the fixed Joint RFF embedding followed by an MLP."""
        self.layer_list = [self.embed_dim] + self.hidden_layer + [self.output_num]

        layers = [_JointRFFEmbedding(self.input_num, self.embed_dim, self.alpha)]
        for i in range(len(self.layer_list) - 2):
            layers.append(nn.Linear(self.layer_list[i], self.layer_list[i+1]))
            layers.append(self.activation)

        layers.append(nn.Linear(self.layer_list[-2], self.layer_list[-1]))

        self.net = nn.Sequential(*layers)
        self._init_weights()
