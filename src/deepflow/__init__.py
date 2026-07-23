import sys
import types

import torch

from . import utility as _utility
from .geometry import *
from .domain import *
from .physicsinformed import *
from .pde import *
from .nn import *
try:
    from .qnn import *
except ImportError as exc:
    if exc.name != "pennylane":
        raise
from .evaluation import *
from .utility import (
    get_device,
    get_dtype,
    set_dtype,
    manual_seed,
    latin_hypercube_sampling,
)


class _DeepFlowModule(types.ModuleType):
    """Module wrapper that exposes writable global configuration properties."""

    @property
    def device(self):
        return _utility.get_device()

    @device.setter
    def device(self, value) -> None:
        _utility.device = value

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype()

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        set_dtype(value)


sys.modules[__name__].__class__ = _DeepFlowModule
