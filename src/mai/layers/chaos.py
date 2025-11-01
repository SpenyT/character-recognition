import numpy as np
from .layers_abstract import Layer

class Chaos(Layer):
    def __init__(self, drop_rate: float, seed: int | None = None, featurewise : bool = False) -> np.ndarray:
        if not (0.0 <= drop_rate < 1.0):
            raise ValueError(f"drop_rate must be in [0, 1), got {drop_rate}.")
        self.p = float(drop_rate)
        self.scale = 1.0 / (1.0 - self.p) if self.p < 1.0 else 0.0
        
        self._rand_num_gen = np.random.default_rng(seed)
        self.featurewise = featurewise
        self._mask = None
        self.is_training = True

    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        if not self.is_training or self.p == 0.0:
            self._mask = None
            return X
        
        mask_shape = (1,) + X.shape[1:] if self.featurewise else X.shape
        self._mask = (self._rand_num_gen(mask_shape) >= self.p).astype(X.dtype)
        return X * self._mask * self.scale

    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self._mask is None or not self.is_training or self.p == 0.0:
            return gradient_output
        else:
            return gradient_output * self._mask * self.scale