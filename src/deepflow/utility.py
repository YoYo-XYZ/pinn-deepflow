import random
import numpy as np
import torch
import scipy
from typing import Tuple, List, Union, Generator, Optional

# Module-level seed storage for reproducibility helpers
_GLOBAL_SEED = None

def latin_hypercube_sampling(n_samples: int, n_dimensions: int, lower_lim:list, upper_lim:list, seed:Optional[int]=None) -> torch.Tensor:
    """
    Generates Latin Hypercube Samples scaled to [lower_lim, upper_lim].

    Reproducibility:
        Pass an explicit ``seed``, or set one globally via :func:`manual_seed`.
        If neither is provided, SciPy's default (``None``) is used and results
        will vary between runs.
    """
    # Resolve seed: explicit > global fallback > None (non-deterministic)
    seed = seed if seed is not None else _GLOBAL_SEED
    lhs = scipy.stats.qmc.LatinHypercube(d=n_dimensions, strength=1, seed=seed)
    sample = lhs.random(n=n_samples)
    sample = scipy.stats.qmc.scale(sample, lower_lim, upper_lim)
    return torch.tensor(sample, dtype=torch.float32)

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
def get_device():
    global device
    return device
def manual_seed(seed:int, deterministic:bool=False):
    """
    Set all random seeds for reproducible training runs.

    Seeds Python's ``random``, NumPy, PyTorch (CPU & GPU), and configures
    cuDNN for determinism when CUDA is available.

    Args:
        seed: Integer seed passed to all RNGs.
        deterministic: If ``True``, enables PyTorch's deterministic mode via
            ``torch.use_deterministic_algorithms(True)`` (may impact performance).
    """
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed

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
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        allow_unused=True
    )[0]
    return grad

def calc_grads(y: torch.Tensor, x_list: Union[Tuple[torch.Tensor, ...], List[torch.Tensor]]) -> Tuple[torch.Tensor, ...]:
    """
    Calculates gradients of a single tensor y with respect to a list of tensors x_list.
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
    Clones tensors and sets requires_grad=True for PINN training.
    """
    result = tuple(t.clone().detach().requires_grad_(True) for t in tensors)
    if len(result) == 1:
        return result[0]
    return result

def torch_to_numpy(*tensors: torch.Tensor) -> Union[float, Tuple]:
    """
    Helper to convert torch tensors (CPU or GPU) to numpy arrays.
    """
    def to_numpy(x):
        return x.detach().cpu().numpy()

    if len(tensors) == 1:
        return to_numpy(tensors[0])
    else:
        return tuple(to_numpy(x) for x in tensors)