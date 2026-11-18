import random

import matplotlib
import numpy as np
import pytest
import torch

matplotlib.use("Agg")

import deepflow as df
from deepflow import utility


@pytest.fixture
def deterministic_cpu(monkeypatch):
    device, dtype = df.device, df.dtype
    torch_dtype = torch.get_default_dtype()
    python_state, numpy_state = random.getstate(), np.random.get_state()
    torch_state = torch.get_rng_state()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    threads = torch.get_num_threads()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    monkeypatch.setattr(utility, "_RNG", utility._RNG)
    monkeypatch.setattr(utility, "_GLOBAL_SEED", utility._GLOBAL_SEED)
    df.device = "cpu"
    df.dtype = torch.float64
    torch.set_num_threads(1)
    df.manual_seed(1729)
    try:
        yield
    finally:
        df.device, df.dtype = device, dtype
        torch.set_default_dtype(torch_dtype)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.set_num_threads(threads)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


@pytest.fixture
def polynomial_model():
    class PolynomialModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.loss_history = {}

        def forward(self, inputs):
            x, y = inputs["x"], inputs["y"]
            u = x**3 + y**3
            if "t" in inputs:
                u = u + inputs["t"] ** 3
            return {"u": u, "v": x**2 * y**2, "p": x**2 + y**2}

    return PolynomialModel()
