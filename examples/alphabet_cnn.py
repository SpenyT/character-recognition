import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from mai.utils import kaggle_download_csv
#from matplotlib import pyplot as plt

# import kaggle dataset & randomize
df = kaggle_download_csv("ashishguptajiit/handwritten-az")
df = shuffle(df, random_state=42).reset_index(drop=True)

labels = df.iloc[:, 0].to_numpy(dtype=int)
pixels = df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0

X_dev, Y_dev = pixels[:1000], labels[:1000]
X_train, Y_train = pixels[1000:], labels[1000:]

# plt.imshow(X_dev[1].reshape(28, 28), cmap="gray")
# plt.axis("off")
# plt.show()

from mai.models import NeuralNetwork
from mai.layers import FCL

model = [
    FCL(784, 142, activation='relu',  weight_initializer="he"),
    FCL(142, 142, activation='relu',  weight_initializer="he"),
    FCL(142,  26, activation=None)
]
net = NeuralNetwork(model, loss='cross_entropy')
net.train(X_train, Y_train, epochs=5, learning_rate=0.01)
test_acc = net.evaluate(X_dev, Y_dev)
print(f"Test accuracy: {test_acc:.4f}")

# visualize samples
from mai.utils import show_random_predictions
show_random_predictions(net, X_dev, Y_dev, n=10, img_size=(28, 28), as_letter=True)

# test with completely new data set
df2 = kaggle_download_csv("sachinpatel21/az-handwritten-alphabets-in-csv-format")
df2 = shuffle(df2, random_state=42).reset_index(drop=True)

labels_new = df.iloc[:, 0].to_numpy(dtype=int)
pixels_new = df.iloc[:, 1:].to_numpy(dtype=np.float32) / 255.0

X_new, Y_new = pixels_new[:], labels_new[:]

# plt.imshow(X_new[1].reshape(28, 28), cmap="gray")
# plt.axis("off")
# plt.show()

print("Evaluating Accuracy...")
test_acc = net.evaluate(X_new, Y_new)
test_acc