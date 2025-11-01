import numpy as np
from . import Layer


class Flatten(Layer):
    def __init__(self):
        self._input_shape = None

    def forward_prop(self, x) -> np.ndarray:
        if x.ndim < 2:
            x = np.expand_dims(x, axis=0)
        self._input_shape = x.shape
        batch = x.shape[0]
        return x.reshape(batch, -1)
    
    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self._input_shape is None:
            raise RuntimeError("Flatten.backward_prop called before forward_prop.")
        return gradient_output.reshape(self._input_shape)