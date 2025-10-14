from abc import ABC, abstractmethod
from typing import Iterable, Optional, Protocol
import numpy as np

from .initializations import initialization_router

# for now lets make each Layer have a forward and backward propogation function
class Layer(ABC):
    @abstractmethod
    def forward_prop(self, x) -> np.ndarray:
        ...

    @abstractmethod
    def backward_prop(self, gradient_output: np.ndarray, learning_rate) -> np.ndarray:
        ...

class Trainable(Protocol):
    @property
    def params(self) -> Iterable[np.ndarray]: ...

    @property
    def grads(self) -> Iterable[np.ndarray]: ...


class FullyConnectedLayer(Layer):
    def __init__(self, n_inputs: int, n_outputs: int, activ: str = 'xavier'):
        self.W = np.random.randn(n_inputs, n_outputs) * np.sqrt(2. / n_inputs) # could maybe allow for more initialization functions
        self.b = np.zeros((1, n_outputs))
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: Optional[np.ndarray] = None

    @property
    def params(self):
        return (self.W, self.b)
    @property
    def grads(self):
        return (self.dW, self.db)
    
    def forward_prop(self, x):
        self.input = x
        return np.dot(x, self.W) + self.b

    def backward_prop(self, gradient_output, learning_rate):
        gradient_input   = np.dot(gradient_output, self.W.T)
        gradient_weights = np.dot(self.input.T, gradient_output)
        gradient_biases  = np.sum(gradient_output, axis=0, keepdims=True)

        self.W -= learning_rate * gradient_weights
        self.b -= learning_rate * gradient_biases

        return gradient_input