from typing import Callable, Dict, Tuple
import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]
Initializer = Callable[[tuple[int, ...]], ArrayF]

_INIT_REG: Dict[str, Initializer] = {}

def register_initializer(name: str):
    def deco(fn: Initializer):
        _INIT_REG[name.lower()] = fn
        return fn
    return deco

def build_initializer(name: str) -> Initializer:
    key = name.lower()
    if key not in _INIT_REG:
        raise ValueError(f"Unknown initializer '{name}'. Available: {list(_INIT_REG)}")
    return _INIT_REG[key]

def available_initializations() -> list[str]: return sorted(_INIT_REG)
from mai.initializations.common import _xavier_normal, _xavier_uniform, _he_normal, _he_uniform, _lecun_normal, _zeroes, _ones