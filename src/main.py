import numpy as np

from models.neural_network import NeuralNetwork
from layers.fully_connected_layer import FCL


X = np.random.randn(5, 10)

net = NeuralNetwork()
net.add(FCL(5, 10, activation="relu", weight_initializer="he", bias_initializer="zeroes"))
net.add(FCL(10, 5, activation="relu", weight_initializer="he", bias_initializer="zeroes"))
output = net.forward_prop(X)
print("Output: ", output)

gradient_output = np.random.randn(5, 10)
net.backward_prop(gradient_output, learning_rate=0.04)