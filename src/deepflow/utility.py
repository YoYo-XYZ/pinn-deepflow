"""Global configuration, reproducibility, and autograd utilities."""

import random
import numpy as np
import torch
from scipy.stats import qmc
from typing import Tuple, List, Union, Generator, Optional

_GLOBAL_SEED = None
_RNG = None

def _next_seed() -> Optional[int]:
    global _RNG
    if _RNG is not None:
        return _RNG.randint(0, 2**32 - 1)
    return None

def latin_hypercube_sampling(n_samples: int, n_dimensions: int, lower_lim:list, upper_lim:list, seed:Optional[int]=None) -> torch.Tensor:
    """Return Latin Hypercube samples within per-axis ``lower_lim`` and
    ``upper_lim``.

    The tensor has shape ``(n_samples, n_dimensions)``. Use ``seed`` or
    ``manual_seed`` for reproducibility; unseeded runs vary.
    """
    seed = seed if seed is not None else (_next_seed() if _RNG is not None else _GLOBAL_SEED)
    lhs = qmc.LatinHypercube(d=n_dimensions, strength=1, seed=seed)
    sample = lhs.random(n=n_samples)
    sample = qmc.scale(sample, lower_lim, upper_lim)
    return torch.tensor(sample, dtype=get_dtype())

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
def get_device():
    """Return the configured PyTorch device name."""
    global device
    return device

_DEFAULT_DTYPE = torch.float32


def get_dtype() -> torch.dtype:
    """Return the current global floating-point dtype used by deepflow."""
    global _DEFAULT_DTYPE
    return _DEFAULT_DTYPE


def set_dtype(value: torch.dtype) -> None:
    """Set DeepFlow and PyTorch's default dtype to ``torch.float32`` or
    ``torch.float64``.

    Aliases ``torch.float`` and ``torch.double`` work too. Other types raise
    ``ValueError``.
    """
    global _DEFAULT_DTYPE
    if value not in (torch.float32, torch.float64):
        raise ValueError(
            f"dtype must be torch.float32 or torch.float64, got {value}"
        )
    _DEFAULT_DTYPE = value
    torch.set_default_dtype(value)
def manual_seed(seed:int, deterministic:bool=False):
    """Seed Python, NumPy, and PyTorch CPU/GPU RNGs; enable deterministic cuDNN.

    ``deterministic=True`` also requires deterministic PyTorch operations,
    which may reduce performance.
    """
    global _GLOBAL_SEED, _RNG
    _GLOBAL_SEED = seed
    _RNG = random.Random(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if deterministic:
        torch.use_deterministic_algorithms(True)

def calc_grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the gradient of ``y`` with respect to ``x``, matching ``x``'s shape.

    Unused inputs receive zeros with the same device and dtype.
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
    """Return gradients of ``y`` as a tuple aligned with ``x_list``; unused inputs
    get zeros.
    """
    grads = torch.autograd.grad(
        outputs=y,
        inputs=x_list,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True
    )
    return tuple(g if g is not None else torch.zeros_like(x) for g, x in zip(grads, x_list))

def to_require_grad(*tensors: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
    """Detach and clone tensors with gradients enabled.

    Return one tensor for a single input, otherwise a tuple.
    """
    result = tuple(t.clone().detach().requires_grad_(True) for t in tensors)
    if len(result) == 1:
        return result[0]
    return result

def torch_to_numpy(*tensors: torch.Tensor) -> Union[float, Tuple]:
    """Detach tensors, move to CPU, and convert to NumPy.

    Return one array for a single input, otherwise a tuple.
    """
    def to_numpy(x):
        return x.detach().cpu().numpy()

    if len(tensors) == 1:
        return to_numpy(tensors[0])
    else:
        return tuple(to_numpy(x) for x in tensors)
