import numpy as np
from typing import Callable, Tuple
from mai.activations import register_activ

@register_activ("sigmoid")
def _sigmoid() -> Tuple[Callable, Callable]:
    def f(Z: np.ndarray) -> np.ndarray:
        Zc = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Zc)) # prevents possible overflow. Was using: return 1 / (1 + np.exp(-Z)) 
    def df(Z: np.ndarray) -> np.ndarray:
        s = f(Z)
        return s * (1.0 - s)
    
    return f, df
