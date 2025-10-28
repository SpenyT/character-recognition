import numpy as np
from typing import Callable, Tuple
from mai.activations import register_activ

@register_activ("sigmoid")
def _sigmoid() -> Tuple[Callable, Callable]:
    def f(Z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-Z))
    def df(Z: np.ndarray) -> np.ndarray:
        s = f(Z)
        return s * (1 - s)
    
    return f, df