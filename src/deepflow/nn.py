"""Neural-network models and training helpers for PINNs."""

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
    pass


class HardConstraint:
    """A constant boundary or initial condition, also created by ``hard_constraint``."""
    def __init__(self, constant: float = 0.0):
        """Initialize a constant hard constraint."""
        self.constant = constant

    @staticmethod
    def define_zero_func(bound):
        """Return a function of coordinate mappings that vanishes on ``bound``."""
        def zero_func(coords):
            return coords[bound.axes_sec[0]] - bound.funcs[bound.ax][0](coords[bound.ax])
        return zero_func

    def __str__(self):
        return str(self.constant)

def hard_constraint(constant=0):
    """Return a ``HardConstraint`` imposing ``constant`` at a boundary or initial
    slice.
    """
    return HardConstraint(constant)

class NN(ABC, nn.Module):
    """Base PINN mapping coordinates to fields with optional hard constraints.

    Args:
        input_vars: Coordinate names, such as ``["x", "y"]``.
        output_vars: Field names, such as ``["u", "v", "p"]``.
        weight_init: ``"kaiming"``/``"he"``, ``"xavier"``/``"glorot"``,
            ``None`` for PyTorch defaults, or a callable receiving the model.

    ``loss_history`` stores total, boundary, PDE, and optional
    initial-condition losses.
    """

    def __init__(
        self,
        input_vars: Optional[List[str]] = None, 
        output_vars: Optional[List[str]] = None,
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        """Set coordinate names, field names, and weight initialization."""
        super().__init__()
        
        self.input_keys = input_vars if input_vars is not None else ['x', 'y']
        self.output_keys = output_vars if output_vars is not None else ['u', 'v', 'p']
        
        self.input_num = len(self.input_keys)
        self.output_num = len(self.output_keys)
        
        self.weight_init = weight_init
        self.hard_constraints: Optional[Dict] = None
        self.hard_constants: Optional[Dict] = None

        self._init_history()
        torch.set_default_dtype(get_dtype())

    @abstractmethod
    def _build_network(self):
        pass

    def _init_weights(self) -> None:
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
                else:
                    nn.init.xavier_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def _init_history(self):
        base_keys = ['total_loss', 'bc_loss', 'pde_loss']
        if 't' in self.input_keys:
            base_keys.append('ic_loss')
            
        self.loss_history = {key: [] for key in base_keys}

    def forward(self, inputs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Return predictions by field name for coordinate tensors in
        ``inputs_dict``.

        Configured hard constraints are applied.
        """
        input_tensor = torch.stack([inputs_dict[key] for key in self.input_keys], dim=1)
        
        pred = self.net(input_tensor)

        output_dict = {}

        for i, key in enumerate(self.output_keys):
            val = pred[:, i]
            
            if self.hard_constraints and key in self.hard_constraints:
                coords = {0: inputs_dict.get("x"), 1: inputs_dict.get("y")}
                constraint_mod = self.hard_constraints[key](coords)
                val = constraint_mod * val + self.hard_constants[key]
                
            output_dict[key] = val

        return output_dict

    def apply_hard_constraints(self, bound_list: list):
        """Apply ``HardConstraint`` values from boundaries in ``bound_list``.

        Conflicting constants for one output field raise ``ValueError``.
        """
        self.hard_constraints = {}
        self.hard_constants = {}
        
        for key in self.output_keys:
            relevant_bounds = [b for b in bound_list if isinstance(b.condition_dict.get(key), HardConstraint)]

            if relevant_bounds:
                constants_list = [b.condition_dict[key].constant for b in relevant_bounds]
                if not all(c == constants_list[0] for c in constants_list):
                    raise ValueError(f"Conflicting hard constraint constants for output '{key}'. All must be the same.")
                
                self.hard_constants[key] = relevant_bounds[0].condition_dict[key].constant

                def constraint_func(coords, bounds = relevant_bounds):
                    result = 1.0
                    for bound in bounds:
                        zero_func = HardConstraint.define_zero_func(bound)
                        result *= zero_func(coords)
                    return result

                self.hard_constraints[key] = constraint_func

    def _record_loss(self, loss_dict: Dict[str, torch.Tensor]):
        for key, val in loss_dict.items():
            value_to_store = val.detach().item() if isinstance(val, torch.Tensor) else val
            if key in self.loss_history: self.loss_history[key].append(value_to_store)

    def print_status(self):
        """Print the most recently recorded training losses."""
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
        current_epoch = len(self.loss_history['total_loss'])
        if current_epoch > 0 and (current_epoch == 1 or current_epoch % print_every == 0):
            self.print_status()
            return current_epoch
        return last_printed_epoch

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
        """Train a copy with Adam; return ``(trained_model, best_model)``.

        Args:
            learning_rate: Adam step size.
            epochs: Training epoch limit.
            calc_loss: ``loss(model)`` returning a dict with ``total_loss``.
            use_scheduler: Enable stepwise learning-rate decay.
            print_every: Positive status-print interval in epochs.
            threshold_loss: Stop below this best-loss value.
            do_between_epochs: Call ``callback(epoch, model)`` after each
                epoch.
            compile_model: Use ``torch.compile`` with PyTorch 2+ and Triton on
                Linux.
            max_grad_norm: Positive gradient clip limit; ``None`` disables
                clipping.

        ``best_model`` holds the lowest recorded loss. Compilation adds startup
        overhead.
        """
        if max_grad_norm is not None and max_grad_norm <= 0:
            raise ValueError("max_grad_norm must be positive or None")
        if print_every <= 0:
            raise ValueError("print_every must be positive")

        model = copy.deepcopy(self).to(get_device())
        if compile_model:
            model = torch.compile(model)

        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        scheduler = None
        if use_scheduler:
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

                loss_dict = calc_loss(model)
                total_loss = loss_dict['total_loss']
                if not torch.isfinite(total_loss).all():
                    print("Detected non-finite loss. Stop the training.")
                    break
                total_loss_num = total_loss.item()
                
                model._record_loss(loss_dict)

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
        """Train a copy with L-BFGS; return ``(trained_model, best_model)``.

        Args:
            epochs: Training epoch limit.
            calc_loss: ``loss(model)`` returning a dict with ``total_loss``.
            print_every: Positive status-print interval in epochs.
            threshold_loss: Stop below this loss.
            do_between_epochs: Call ``callback(epoch, model)`` after each
                epoch.
            compile_model: Use ``torch.compile``; requires PyTorch 2+.

        ``best_model`` holds the lowest recorded loss. Compilation adds startup
        overhead.
        """
        if print_every <= 0:
            raise ValueError("print_every must be positive")

        model = copy.deepcopy(self).to(get_device())
        if compile_model:
            model = torch.compile(model)

        model.train()

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
                    total_loss.backward()
                    return total_loss.detach()
                
                try:
                    optimizer.step(closure)
                except _NonFiniteLossError:
                    model.load_state_dict(epoch_state)
                    break

                loss_dict = calc_loss(model)
                total_loss = loss_dict['total_loss']
                if not torch.isfinite(total_loss).all():
                    print("Detected non-finite loss. Stop the training.")
                    model.load_state_dict(epoch_state)
                    break
                total_loss_num = total_loss.item()
                model._record_loss(loss_dict)

                if total_loss_num < best_loss:
                    best_loss = total_loss_num
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

                last_printed_epoch = model._maybe_print_status(print_every, last_printed_epoch)
                
                if threshold_loss is not None and total_loss_num < threshold_loss:
                     print(f"Stop: Loss {total_loss_num:.5f} < Threshold {threshold_loss}")
                     break
                
                if do_between_epochs: 
                    do_between_epochs(epoch, model)

        except KeyboardInterrupt:
            print('Training interrupted by user.')
        
        best_model = copy.deepcopy(model)
        if best_state is not None:
            best_model.load_state_dict(best_state)

        current_epoch = len(model.loss_history['total_loss'])
        if current_epoch > initial_history_len and current_epoch != last_printed_epoch:
            model.print_status()
        return model, best_model
    
    def save_as_pickle(self, file_name: str = "model.pkl") -> None:
        """Save to ``file_name``, adding ``.pkl`` if absent.

        Uses ``cloudpickle`` to preserve dynamic hard constraints. Loading
        pickle files can execute code; only load trusted files.
        """
        import cloudpickle as pickle
        if file_name[-4:] != '.pkl': file_name += '.pkl'
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)


def load_from_pickle(file_name: str) -> NN:
    """Return a saved model from ``file_name``, adding ``.pkl`` if absent.

    Accepts cloudpickle and legacy pickle files. Only load trusted files;
    loading can execute code.
    """
    import cloudpickle as pickle
    if file_name[-4:] != '.pkl': file_name += '.pkl'
    with open(file_name, 'rb') as f:
        return pickle.load(f)


class _JointRFFEmbedding(nn.Module):

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
    """Fully connected PINN with configurable hidden-layer widths.

    Args:
        input_vars: Coordinate names.
        output_vars: Field names.
        hidden_layer: Width per hidden layer.
        activation: Hidden-layer activation module.
        weight_init: Initialization scheme; see ``NN``.
    """

    def __init__(
        self,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        hidden_layer: List[int] = [50, 50, 50, 50],
        activation: nn.Module = nn.Tanh(),
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        """Initialize and build the feedforward network."""
        super().__init__(input_vars, output_vars, weight_init=weight_init)
        self.activation = activation
        self.hidden_layer = hidden_layer
        self._build_network()
        self.to(get_dtype())

    def _build_network(self) -> None:
        self.layer_list = [self.input_num] + self.hidden_layer + [self.output_num]

        layers = []
        for i in range(len(self.layer_list) - 2):
            layers.append(nn.Linear(self.layer_list[i], self.layer_list[i+1]))
            layers.append(self.activation)

        layers.append(nn.Linear(self.layer_list[-2], self.layer_list[-1]))

        self.net = nn.Sequential(*layers)
        self._init_weights()

class PINN(FNN):
    """Fully connected PINN with equal-width hidden layers.

    Args:
        input_vars: Coordinate names.
        output_vars: Field names.
        width: Hidden-layer width.
        length: Hidden-layer count.
        activation: Hidden-layer activation module.
        weight_init: Initialization scheme; see ``NN``.
    """

    def __init__(
        self,
        input_vars: Optional[List[str]] = None,
        output_vars: Optional[List[str]] = None,
        width: int = 32,
        length: int = 4,
        activation: nn.Module = nn.Tanh(),
        weight_init: Union[str, Callable, None] = 'kaiming',
    ):
        """Initialize a PINN with equally wide hidden layers."""
        super().__init__(
            input_vars, output_vars,
            [width for _ in range(length)],
            activation,
            weight_init=weight_init,
        )


class RFFPINN(FNN):
    """PINN with a fixed random Fourier input embedding.

    Args:
        input_vars: Coordinate names.
        output_vars: Field names.
        width: Hidden-layer width after embedding.
        length: Hidden-layer count after embedding.
        embed_dim: Positive even embedding dimension.
        alpha: Frequency standard-deviation scale.
        activation: Hidden-layer activation module.
        weight_init: Initialization scheme; see ``NN``.
    """

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
        """Initialize a PINN with a fixed random Fourier embedding."""
        self.embed_dim = embed_dim
        self.alpha = alpha
        super().__init__(
            input_vars, output_vars,
            [width for _ in range(length)],
            activation,
            weight_init=weight_init,
        )

    def _build_network(self) -> None:
        self.layer_list = [self.embed_dim] + self.hidden_layer + [self.output_num]

        layers = [_JointRFFEmbedding(self.input_num, self.embed_dim, self.alpha)]
        for i in range(len(self.layer_list) - 2):
            layers.append(nn.Linear(self.layer_list[i], self.layer_list[i+1]))
            layers.append(self.activation)

        layers.append(nn.Linear(self.layer_list[-2], self.layer_list[-1]))

        self.net = nn.Sequential(*layers)
        self._init_weights()
