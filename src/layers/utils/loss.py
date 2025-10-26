import numpy as np
from typing import Callable, Tuple

def resolve_loss(
    loss_name: str
) -> Tuple[
    Callable[[np.ndarray, np.ndarray], np.ndarray], 
    Callable[[np.ndarray, np.ndarray], np.ndarray]
]:
    match loss_name:
        case "mse":
            return mse, mse_prime
        case "cross_entropy":
            return cross_entropy, cross_entropy_prime
        case _:
            raise ValueError(f"Loss function '{loss_name}' not recognized.")


def softmax(Z):
    Z_exp = np.exp(Z - np.max(Z))
    A = Z_exp / Z_exp.sum(axis=0, keepdims=True)
    return A

def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))
            
def mse_prime(y_true, y_pred):
    return 2 * (y_pred - y_true) / y_true.size

def cross_entropy(y_true, y_pred):
    m = y_true.shape[1]
    p = softmax(y_pred)
    log_likelihood = -np.log(p[y_true, range(m)])
    loss = np.sum(log_likelihood) / m
    return loss
            
def cross_entropy_prime(y_true, y_pred):
    m = y_true.shape[1]
    grad = softmax(y_pred)
    grad[y_true, range(m)] -= 1
    grad = grad / m
    return grad