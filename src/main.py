import numpy as np

from utils.neural_network import NeuralNetwork
from utils.neural_layers import *
from utils.activations import ReLU


X = np.random.randn(10, 5)

net = NeuralNetwork()
net.add(FullyConnectedLayer(5, 4))
net.add(ReLU())
net.add(FullyConnectedLayer(4, 3))

output = net.forward_prop(X)
print("Output: ", output)

gradient_output = np.random.randn(10, 3)
net.backward_prop(gradient_output, learning_rate=0.04)