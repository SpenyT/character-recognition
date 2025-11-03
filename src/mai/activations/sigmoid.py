import numpy as np
from . import register_activ, Activation, Array

@register_activ("sigmoid")
def _sigmoid() -> Activation:
    def f(Z: Array) -> Array:
        Zc = np.clip(Z, -500, 500)
        return 1.0 / (1.0 + np.exp(-Zc)) # prevents possible overflow. Was using: return 1 / (1 + np.exp(-Z)) 
    def df(Z: Array) -> Array:
        s = f(Z)
        return s * (1.0 - s)
    return f, df
