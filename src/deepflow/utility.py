import torch
import scipy
from typing import Tuple, List, Union, Generator

def latin_hypercube_sampling(n_samples: int, n_dimensions: int, lower_lim:list, upper_lim:list) -> torch.Tensor:
    """
    Generates Latin Hypercube Samples in the unit hypercube [0,1]^n_dimensions.
    """
    # Generate LHS using scipy
    lhs = scipy.stats.qmc.LatinHypercube(d=n_dimensions, strength=1)
    sample = lhs.random(n=n_samples)
    sample = scipy.stats.qmc.scale(sample, lower_lim, upper_lim)
    return torch.tensor(sample, dtype=torch.float32)

device = 'cpu' if not torch.cuda.is_available() else 'cuda'
def get_device():
    global device
    return device
def manual_seed(seed:int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # For current GPU

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