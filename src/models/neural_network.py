import numpy as np
from layers import Layer
from models.utils.loss import resolve_loss


class NeuralNetwork:
    def __init__(self, model : np.ndarray[Layer] = [], loss: str = "mse"):
        self.layers = model
        self.loss, self.loss_prime = resolve_loss(loss)

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(
                f"Cannot add object of type {type(layer)}. "
                f"Only objects that inherit from 'Layer' are supported."
            )
        self.layers.append(layer)

    def forward_prop(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_prop(input_data)
        return input_data
    
    def backward_prop(self, loss_grad, learning_rate):
        for layer in reversed(self.layers):
            loss_grad = layer.backward_prop(loss_grad, learning_rate)

    def train(self, x_train, y_train, epochs: int, learning_rate: float, print_output: bool = True):
        x_train = np.asarray(x_train)
        y_train = np.asarray(y_train)

        print("Starting training...")
        for epoch in range(epochs):
            err = 0.0
            correct = 0
            for x, y in zip(x_train, y_train):
                x = x.reshape(1, -1)
                logits = self.forward_prop(x)
                err += self.loss(np.array([y]), logits)
                grad = self.loss_prime(np.array([y]), logits)
                self.backward_prop(grad, learning_rate)

                pred = int(np.argmax(logits, axis=1)[0])
                if pred == int(y): correct += 1

            err /= len(x_train)
            train_acc = correct / len(x_train)
            print(f"Epoch {epoch+1}/{epochs} - loss: {err:.4f} - acc: {train_acc:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        logits = self.forward_prop(np.asarray(X))
        return np.argmax(logits, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float((preds == y).mean())
