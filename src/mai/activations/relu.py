import numpy as np
from . import register_activ, Activation, Array

@register_activ("relu")
def _relu() -> Activation:
    def f(Z: Array) -> Array:
        return np.maximum(0, Z)
    def df(Z: Array) -> Array:
        return (Z > 0).astype(float)
    return f, df