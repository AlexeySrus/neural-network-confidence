from keras.datasets import mnist, cifar10
from sklearn.utils import shuffle
import numpy as np


class Loader:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def shuffle(self):
        self.x, self.y = shuffle(self.x, self.y)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def one_hot_mnist(values):
    res = []

    for v in values:
        elem = [0]*10
        elem[v] = 1
        res.append(elem)

    return np.array(res).astype('float32')


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape((len(x_train), 1, 28, 28)) / 255.0).astype(
        'float32'
    )
    x_test = (x_test.reshape((len(x_test), 1, 28, 28)) / 255.0).astype(
        'float32'
    )

    return x_train, one_hot_mnist(y_train), x_test, one_hot_mnist(y_test)


def load_cifar10():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = (x_train.reshape((len(x_train), 1, 28, 28)) / 255.0).astype(
        'float32'
    )
    x_test = (x_test.reshape((len(x_test), 1, 28, 28)) / 255.0).astype(
        'float32'
    )

    return x_train, one_hot_mnist(y_train), x_test, one_hot_mnist(y_test)


def load_mnist_for_ae():
    x_train, y_train, x_test, y_test = load_mnist()
    return x_train, x_train, x_test, x_test


def load_cifar10_for_ae():
    x_train, y_train, x_test, y_test = load_cifar10()
    return x_train, x_train, x_test, x_test


def get_loaders(load_data):
    x_train, y_train, x_test, y_test = load_data
    return Loader(x_train, y_train), Loader(x_test, y_test)
