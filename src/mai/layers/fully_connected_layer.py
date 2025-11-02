import numpy as np
from typing import Dict, Callable, Tuple, Optional, Iterable, Protocol, List
from mai.initializations import build_init
from mai.activations import build_activ, Array
from mai.layers.layers_abstract import Layer

class FCL(Layer):
    def __init__(self, in_features: int, out_features : int, activ: Optional[str] = None, weight_init: str = "he_uniform", bias_init: str = "zeroes", seed: Optional[int] = None, dtype: np.dtype = np.float32):
        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype
        self._rng = np.random.default_rng(seed)

        self._activation_fn, self._activation_deriv = build_activ(activ)

        w_init = build_init(weight_init)
        b_init = build_init(bias_init)

        self.W: Array = w_init((in_features, out_features), self._rng).astype(dtype, copy=False)
        self.b: Array = b_init((out_features,), self._rng).astype(dtype, copy=False)
        
        self._dW: Array = np.zeros_like(self.W)
        self._db: Array = np.zeros_like(self.b)
        self._input: Optional[Array] = None
        self._Z: Optional[Array] = None

    @property
    def params(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self.W, self.b)
    @property
    def grads(self) -> Tuple[np.ndarray, np.ndarray]:
        return (self._dW, self._db)
    
    def forward_prop(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(self.dtype, copy=False)
        if X.shape[-1] != self.in_features:
            raise ValueError(f"FCL.forward: expected last dim {self.in_features}, got {X.shape[-1]}")
        self._input = X
        Z = X @ self.W + self.b
        self._Z = Z
        return self._activation_fn(Z)

    def backward_prop(self, gradient_output: np.ndarray, learning_rate: float) -> np.ndarray:
        if self._input is None or self._Z is None:
            raise RuntimeError("Backward called before forward")
        act_prime = self._activation_deriv(self._Z)

        if act_prime.shape != gradient_output.shape:
            print("ERROR: derivation shape: ", act_prime.shape, "!= gradient output shape: ", gradient_output.shape)
        
        dZ = gradient_output * act_prime
        self._dW[...] = self._input.T @ dZ
        self._db[...] = dZ.sum(axis=0, keepdims=True)

        grad_input = dZ @ self.W.T

        self.W -= learning_rate * self._dW
        self.b -= learning_rate * self._db
        return grad_input