import numpy as np
from typing import Optional, Callable, Tuple
from .utils.initializations import resolve_weight_initializer, resolve_bias_initializer
from .utils.activations import resolve_activation
from .layers_abstract import Layer

class FCL(Layer):
    def __init__(self, n_input: int, n_output : int, activation: str = None, weight_initializer: str = "he", bias_initializer: str = "zeros"):
        self._activation_fn: Callable[[np.ndarray], np.ndarray] = None 
        self._activation_deriv: Callable[[np.ndarray], np.ndarray] = None
        self._activation_fn, self._activation_deriv = resolve_activation(activation)
        self.W = np.random.randn(n_output, n_input) * resolve_weight_initializer(weight_initializer, n_input, n_output)
        self.b = resolve_bias_initializer(bias_initializer, n_output)
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
        self._Z = self.W.dot(X) + self.b # save pre-activation input
        return self._activation_fn(self._Z)

    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        dZ = gradient_output.T * self._activation_deriv(self._Z)
        gradient_input = self.W.T @ dZ
        gradient_weights = dZ @ self._input 
        gradient_biases = np.sum(dZ, axis=1, keepdims=True)

        self.W -= learning_rate * gradient_weights
        self.b -= learning_rate * gradient_biases

        return gradient_input.T