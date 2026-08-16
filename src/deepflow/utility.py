"""Global configuration, reproducibility, and autograd utilities."""

import random
import numpy as np
import torch
import scipy
from typing import Tuple, List, Union, Generator, Optional

# Module-level seed storage for reproducibility helpers
_GLOBAL_SEED = None
_RNG = None  # Advancing deterministic RNG for repeated sampling calls

def _next_seed() -> Optional[int]:
    """Return the next integer from the advancing module-level RNG, or None if not seeded."""
    global _RNG
    if _RNG is not None:
        return _RNG.randint(0, 2**32 - 1)
    return None

def latin_hypercube_sampling(n_samples: int, n_dimensions: int, lower_lim:list, upper_lim:list, seed:Optional[int]=None) -> torch.Tensor:
    """
    Generates Latin Hypercube Samples scaled to [lower_lim, upper_lim].

    Reproducibility:
        Pass an explicit ``seed``, or set one globally via ``manual_seed``.
        If neither is provided, SciPy's default (``None``) is used and results
        will vary between runs.

    Args:
        n_samples: Number of samples.
        n_dimensions: Number of dimensions.
        lower_lim: Lower bound for each dimension.
        upper_lim: Upper bound for each dimension.
        seed: Optional per-call random seed.

    Returns:
        A tensor of shape ``(n_samples, n_dimensions)``.
    """
    # Resolve seed: explicit > per-call advancing RNG > global fallback > None (non-deterministic)
    seed = seed if seed is not None else (_next_seed() if _RNG is not None else _GLOBAL_SEED)
    lhs = scipy.stats.qmc.LatinHypercube(d=n_dimensions, strength=1, seed=seed)
    sample = lhs.random(n=n_samples)
    sample = scipy.stats.qmc.scale(sample, lower_lim, upper_lim)
    return torch.tensor(sample, dtype=get_dtype())

# Module-level device configuration (mirrored by dtype below)
device = 'cpu' if not torch.cuda.is_available() else 'cuda'
def get_device():
    """Return the configured PyTorch device name."""
    global device
    return device

# Module-level floating-point dtype configuration
_DEFAULT_DTYPE = torch.float32


def get_dtype() -> torch.dtype:
    """Return the current global floating-point dtype used by deepflow."""
    global _DEFAULT_DTYPE
    return _DEFAULT_DTYPE


def set_dtype(value: torch.dtype) -> None:
    """
    Set the global floating-point dtype used by deepflow.

    Args:
        value: ``torch.float32`` or ``torch.float64``. Also accepts the
            aliases ``torch.float`` and ``torch.double``.

    Raises:
        ValueError: If ``value`` is not a supported floating-point dtype.

    Returns:
        None. The global default dtype is updated in place.
    """
    global _DEFAULT_DTYPE
    if value not in (torch.float32, torch.float64):
        raise ValueError(
            f"dtype must be torch.float32 or torch.float64, got {value}"
        )
    _DEFAULT_DTYPE = value
    torch.set_default_dtype(value)
def manual_seed(seed:int, deterministic:bool=False):
    """
    Set all random seeds for reproducible training runs.

    Seeds Python's ``random``, NumPy, PyTorch (CPU & GPU), and configures
    cuDNN for determinism when CUDA is available.

    Args:
        seed: Integer seed passed to all RNGs.
        deterministic: If ``True``, enables PyTorch's deterministic mode via
            ``torch.use_deterministic_algorithms(True)`` (may impact performance).

    Returns:
        None.
    """
    global _GLOBAL_SEED, _RNG
    _GLOBAL_SEED = seed
    _RNG = random.Random(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)        # Multi-GPU coverage
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)

def calc_grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Calculates the gradient of tensor y with respect to tensor x.

    If ``y`` does not depend on ``x``, return a zero tensor with the same
    shape, device, and dtype as ``x``.

    Args:
        y: Tensor whose gradient is requested.
        x: Tensor with respect to which the gradient is taken.

    Returns:
        Gradient tensor with the same shape as ``x``.
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True
    )[0]
    return grad if grad is not None else torch.zeros_like(x)

def calc_grads(y: torch.Tensor, x_list: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    """
    Calculate gradients of one tensor with respect to several tensors.

    Args:
        y: Tensor whose gradients are requested.
        x_list: Coordinate tensors used as differentiation inputs.

    Returns:
        Tuple of gradients aligned with ``x_list``. Unused inputs receive zero
        tensors.
    """
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x_list,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True
    )
    # Handle cases where gradient doesn't depend on input (return zeros)
    return tuple(g if g is not None else torch.zeros_like(x) for g, x in zip(grads, x_list))

def to_require_grad(*tensors: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """
    Clone tensors and enable gradients for PINN training.

    Args:
        *tensors: Tensors to detach, clone, and mark differentiable.

    Returns:
        One tensor for a single input, otherwise a tuple of tensors.
    """
    result = tuple(t.clone().detach().requires_grad_(True) for t in tensors)
    if len(result) == 1:
        return result[0]
    return result

def torch_to_numpy(*tensors: torch.Tensor) -> Union[float, Tuple]:
    """
    Convert CPU or GPU tensors to NumPy arrays.

    Args:
        *tensors: Tensors to detach and move to CPU.

    Returns:
        One NumPy array for a single input, otherwise a tuple of arrays.
    """
    def to_numpy(x):
        return x.detach().cpu().numpy()

    if len(tensors) == 1:
        return to_numpy(tensors[0])
    else:
        return tuple(to_numpy(x) for x in tensors)
