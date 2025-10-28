import numpy as np
from typing import Callable, Tuple
from mai.activations import register_activ

@register_activ("relu")
def _relu() -> Tuple[Callable, Callable]:
    def f(Z: np.ndarray) -> np.ndarray:
        return np.maximum(0, Z)
    def df(Z: np.ndarray) -> np.ndarray:
        return (Z > 0).astype(float)
    
    return f, df