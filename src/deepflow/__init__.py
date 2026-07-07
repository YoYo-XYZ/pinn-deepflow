import sys
import types

import torch

from .geometry import *
from .domain import *
from .physicsinformed import *
from .pde import *
from .nn import *
from .qnn import *
from .evaluation import *
from .utility import (
    device,
    get_device,
    get_dtype,
    set_dtype,
    manual_seed,
    latin_hypercube_sampling,
)


class _DeepFlowModule(types.ModuleType):
    """Module wrapper that makes ``df.dtype`` a readable/writable property."""

    @property
    def dtype(self) -> torch.dtype:
        return get_dtype()

    @dtype.setter
    def dtype(self, value: torch.dtype) -> None:
        set_dtype(value)


sys.modules[__name__].__class__ = _DeepFlowModule