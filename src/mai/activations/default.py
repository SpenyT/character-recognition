import numpy as np
from typing import Callable, Tuple
from mai.activations import register_activ

@register_activ(None)
def _none():
    def f(x):  return x
    def df(x): return np.ones_like(x)
    return f, df