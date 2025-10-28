import numpy as np
from typing import Optional, Callable, Tuple
from mai.initializations import build_initializer
from mai.activations import build_activ
from mai.layers.layers_abstract import Layer

class FCL(Layer):
    def __init__(self, in_features: int, out_features : int, activ: str = None, weight_init: str = "he_uniform", bias_init: str = "zeroes"):
        self.in_features = in_features
        self.out_features = out_features
        self._activation_fn: Callable[[np.ndarray], np.ndarray] = None 
        self._activation_deriv: Callable[[np.ndarray], np.ndarray] = None
        self._activation_fn, self._activation_deriv = build_activ(activ)
        w_init = build_initializer(weight_init)
        b_init = build_initializer(bias_init)
        self.W = w_init((in_features, out_features))
        self.b = b_init((1, out_features))
        self._dW = np.zeros_like(self.W)
        self._db = np.zeros_like(self.b)
        self._input: Optional[np.ndarray] = None

    @property
    def params(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.W, self.b)
    @property
    def grads(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._dW, self._db)
    
    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        self._input = X
        self._Z = X @ self.W + self.b
        return self._activation_fn(self._Z)

    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        local = self._activation_deriv(self._Z)
        if local.shape != gradient_output.shape:
            print("ERROR: derivation shape: ", local.shape, "!= gradient output shape: ", gradient_output.shape)
        dZ = gradient_output * local

        self._dW = self._input.T @ dZ
        self._db = dZ.sum(axis=0, keepdims=True)

        grad_input = dZ @ self.W.T

        self.W -= learning_rate * self._dW
        self.b -= learning_rate * self._db

        return grad_input