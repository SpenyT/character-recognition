import numpy as np
from . import register_activ, Activation, Array

@register_activ("tanh")
def _tanh() -> Activation:
    def f(Z: Array) -> Array:
        return np.tanh(Z)
    def df(Z: Array) -> Array:
        A = np.tanh(Z)
        return 1.0 - A * A
    
    return f, df