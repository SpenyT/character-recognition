import numpy as np

def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z))
    A = Z_exp / Z_exp.sum(axis=0, keepdims=True)
    return A


def mse(y_true, y_pred):
    return np.mean(np.power(y_true -y_pred, 2))


def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size
