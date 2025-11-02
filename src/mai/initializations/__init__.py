from __future__ import annotations
from typing import Dict, Callable, Tuple, Protocol, runtime_checkable
import numpy as np
from numpy.typing import NDArray

ArrayF = NDArray[np.floating]
InitFn = Callable[[Tuple[int, ...], np.random.Generator], ArrayF] 

@runtime_checkable
class Initializer(Protocol): 
    def __call__(self, rng: np.random.Generator, shape: Tuple[int, ...], dtype: np.dtype = np.float32) -> ArrayF: ...

_INIT_REG: Dict[str, Initializer] = {}

def register_init(name: str): 
    def deco(fn: Initializer): 
        if not isinstance(fn, Initializer):  
            raise TypeError(f"{fn} does not look like an Initializer") 
        _INIT_REG[name.lower()] = fn 
        return fn 
    return deco

def build_init(name: str) -> InitFn:
    key = name.lower() 
    try: 
        return _INIT_REG[key] 
    except KeyError: 
        raise KeyError(f"Unknown initializer '{name}'. Available: {', '.join(sorted(_INIT_REG))}")

def available_inits() -> list[str]: 
    return sorted(_INIT_REG)

from mai.initializations.common import _xavier_normal, _xavier_uniform, _he_normal, _he_uniform, _lecun_normal, _zeroes, _ones