import numpy as np
from mai.losses import register_loss

@register_loss("cross_entropy")
def _cross_entropy():
    def softmax(z):
        z = z - z.max(axis=1, keepdims=True)
        exp = np.exp(z); return exp / exp.sum(axis=1, keepdims=True)
    def loss(y_true_idx, logits):
        p = softmax(logits)
        n = y_true_idx.shape[0]
        return -np.log(p[np.arange(n), y_true_idx]).mean()
    def grad(y_true_idx, logits):
        p = softmax(logits)
        n = y_true_idx.shape[0]
        p[np.arange(n), y_true_idx] -= 1.0
        return p / n
    return loss, grad