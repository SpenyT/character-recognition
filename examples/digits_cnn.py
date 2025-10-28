import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
X = digits.data.astype(np.float32) / 16.0
Y = digits.target.astype(int)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

from mai.layers import FCL
from mai.models import NeuralNetwork

model = [
    FCL(64, 64, activ='relu', weight_init="xavier_normal", bias_init="ones"), 
    FCL(64, 10, activ=None, weight_init="xavier_uniform", bias_init="zeroes")
]

net = NeuralNetwork(model, loss='cross_entropy')
net.train(X_train, Y_train, epochs=5, learning_rate=0.01)

test_acc = net.evaluate(X_test, Y_test)
print(f"Test Accuracy: {test_acc:.4f}")