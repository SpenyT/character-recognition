import numpy as np
from .neural_layers import Layer

class ReLU(Layer):
    def __init__(self):
        self._mask = None # tells us which is activated and which isn't

    def forward_prop(self, x) -> np.ndarray:
        self._mask = x > 0
        return x * self._mask
    
    def backward_prop(self, gradient_output: np.ndarray, learning_rate=None) -> np.ndarray:
        return gradient_output * self._mask.astype(gradient_output.dtype)


class LeakyReLU(Layer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def forward_prop(self, x):
        self.input = x
        return np.where(x > 0, x, self.learning_rate * x)
    
    def backward_prop(self, gradient_output, learning_rate=None):
        dx = np.ones_like(self.input)
        dx[self.input <= 0] = self.learning_rate
        return gradient_output * dx


class Sigmoid(Layer):
    def forward_prop(self, x):
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward_prop(self, gradient_output, learning_rate=None):
        return gradient_output * self.out * (1 - self.out)



#----------------------------------------------------#

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward_prop(self, x):
        return np.reshape(x, self.output_shape)

    def backward_prop(self, gradient_output, learning_rate=None):
        return np.reshape(gradient_output, self.input_shape)


