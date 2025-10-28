import numpy as np
from typing import Callable, Tuple
from mai.activations import register_activ

@register_activ(None)
def _none() -> Tuple[Callable, Callable]:
    def f(x: np.ndarray) -> np.ndarray:  return x
    def df(x: np.ndarray)-> np.ndarray: return np.ones_like(x)
    return f, df