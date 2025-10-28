import numpy as np
from matplotlib import pyplot as plt
import random

def show_example(net, X: np.ndarray, y: np.ndarray, idx: int, img_size=(28, 28), as_letter=False):
    x1 = X[idx].reshape(1, -1)
    pred = int(net.predict(x1)[0])
    label = int(y[idx])
    
    def to_char(v): return chr(v + 65)
    pred_str  = to_char(pred)  if as_letter else str(pred)
    label_str = to_char(label) if as_letter else str(label)

    plt.imshow(X[idx].reshape(*img_size), cmap='gray', interpolation='nearest')
    plt.axis('off')
    plt.title(f"Pred: {pred_str}  |  Label: {label_str}")
    plt.show()


def show_random_predictions(net, X, y, n=5, img_size=(28, 28), as_letter=False, seed=None):
    rng = random.Random(seed)
    indices = rng.sample(range(len(y)), k=min(n, len(y)))

    for idx in indices:
        show_example(net, X, y, idx, img_size=img_size, as_letter=as_letter)