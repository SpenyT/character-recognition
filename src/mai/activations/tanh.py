import numpy as np
from typing import Callable, Tuple
from mai.activations import register_activ

@register_activ("tanh")
def _tanh() -> Tuple[Callable, Callable]:
    def f(Z: np.ndarray) -> np.ndarray:
        return np.tanh(Z)
    def df(Z: np.ndarray) -> np.ndarray:
        A = np.tanh(Z)
        return 1.0 - A * A
    
    return f, df