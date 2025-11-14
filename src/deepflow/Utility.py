import torch

def calc_grad(y, x):
    """
    Calculates the gradient of a tensor y with respect to a tensor x.

    Returns:
        torch.Tensor: The gradient of y with respect to x.
    """
    grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=torch.ones_like(y),
        retain_graph=True,
        create_graph=True,)[0]
    return grad

def to_require_grad(*tensors):
    if len(tensors) == 1:
        return tensors[0].clone().detach().requires_grad_(True)
    else:
        return (t.clone().detach().requires_grad_(True) for t in tensors)

def torch_to_numpy(*tensors):
    def to_numpy(x):
        try:
            return x.detach().numpy()
        except:
            return x.numpy()

    if len(tensors) == 1:
        return to_numpy(tensors[0])
    else:
        return tuple(to_numpy(x) for x in tensors)