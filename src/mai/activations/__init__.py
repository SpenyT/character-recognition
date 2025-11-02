from __future__ import annotations
from typing import Dict, Callable, Tuple, Optional, Protocol, List
import numpy as np

Array = np.ndarray
Activation = Tuple[Callable[[Array], Array], Callable[[Array], Array]]

class ActivationFactory(Protocol):
    def __call__(self) -> Activation: ...

_ACTIV_REG: Dict[Optional[str], Activation] = {}

def register_activ(name: Optional[str]):
    def deco(factory: ActivationFactory):
        key = None if name is None else name.lower()
        if key in _ACTIV_REG: 
            raise KeyError(f"Activation '{key}' already registered")
        _ACTIV_REG[key] = factory() 
        return factory
    return deco

def build_activ(name: Optional[str]) -> Activation:
    key = None if name is None else name.lower()
    try: 
        return _ACTIV_REG[key] 
    except KeyError: 
        raise KeyError(f"Unknown activation '{name}'. Available: {', '.join(sorted(k for k in _ACTIV_REG if k is not None))}")

def available_activations() -> List[str]: 
    return sorted(k for k in _ACTIV_REG if k is not None)

from mai.activations.default import _none
from mai.activations.relu import _relu
from mai.activations.sigmoid import _sigmoid
from mai.activations.tanh import _tanh