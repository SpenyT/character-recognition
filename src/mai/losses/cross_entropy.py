import numpy as np
from numpy.typing import NDArray
from typing import Callable, Tuple
from mai.losses import register_loss

#numpy typing
ArrayF = NDArray[np.floating]
ArrayI = NDArray[np.integer]

@register_loss("cross_entropy")
def _cross_entropy() -> Tuple[Callable, Callable]:
    def softmax(z: ArrayF) -> ArrayF:
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z); return exp / exp.sum(axis=1, keepdims=True)
    
    def loss(y_idx: ArrayI, logits: ArrayF) -> float:
        y = np.asarray(y_idx).ravel().astype(np.int64)
        p = softmax(logits)
        n = y.shape[0]
        eps = 1e-12
        return float(-np.log(p[np.arange(n), y] + eps).mean())
    
    def grad(y_idx: ArrayI, logits: ArrayF) -> ArrayF:
        y = np.asarray(y_idx).ravel().astype(np.int64)
        p = softmax(logits)
        n = y.shape[0]
        p[np.arange(n), y] -= 1.0
        return p / n
    
    return loss, grad