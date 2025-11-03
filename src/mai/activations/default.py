import numpy as np
from . import register_activ, Activation, Array

@register_activ(None)
def _none() -> Activation:
    def f(x: Array) -> Array:  return x
    def df(x: Array)-> Array: return np.ones_like(x)
    return f, df