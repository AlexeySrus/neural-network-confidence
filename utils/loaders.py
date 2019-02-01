from keras.datasets import mnist, cifar10
from sklearn.utils import shuffle
import numpy as np
import os
import cv2
from utils.preprocessing import resize_image


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


class NIST19Loader:
    dirs = ['4a', '4b', '4c', '4d', '4e', '4f', '5a', '6a', '6b', '6c', '6d',
            '6e', '6f', '7a', '30', '31', '32', '33',
            '34', '35', '36', '37', '38', '39', '41', '42', '43', '44', '45',
            '46', '47', '48', '49', '50',
            '51', '52', '53', '54', '55', '56', '57', '58', '59', '61', '62',
            '63', '64', '65', '66', '67', '68', '69', '70',
            '71', '72', '73', '74', '75', '76', '77', '78', '79']

    classes = ['J', 'K', 'L', 'M', 'N', 'O', 'Z', 'j', 'k', 'l', 'm', 'n', 'o',
               'z', '0', '1', '2', '3',
               '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G',
               'H', 'I', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'a', 'b', 'c', 'd',
               'e', 'f', 'g', 'h', 'i', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y']

    def __init__(self, path, shape=(72, 72), shuffled=True):
        self.path = path
        self.shape = shape

        self.data = self.generate_paths()

        self.x = [d[1] for d in self.data]
        self.y = [d[0] for d in self.data]

        if shuffled:
            self.x, self.y = shuffle(self.x, self.y)

    def generate_imgs_tuple(self):
        supdirs = list(
            set(['hsf_{}'.format(i) for i in range(8)]) - set(['hsf_5']))

        classes_dirs_tuple = {
            c: [self.path + '/by_class/' + d + '/' + s + '/' for s in supdirs] +
               [self.path + '/by_class/' + d + '/train_' + d + '/']
            for c, d in zip(self.classes, self.dirs)
        }

        classes_imgs_tuple = {
            c: self.flatten_list(
                [[d + img_name for img_name in os.listdir(d)] for d in
                 drs_list])
            for c, drs_list in classes_dirs_tuple.items()
        }

        return classes_imgs_tuple

    def generate_paths(self):
        classes_imgs_tuple = self.generate_imgs_tuple()

        _label_img_store = [(label, img) for label in self.classes for img in
                            classes_imgs_tuple[label]]

        return _label_img_store

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.x[idx], 0)

        if img is None:
            if idx >= len(self.x):
                return self[idx - 1]
            return self[idx + 1]

        img = resize_image(img, self.shape)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32') / 255.0
        return img, self.one_hot_vector(self.y[idx])

    @staticmethod
    def flatten_list(lst):
        res = []
        for l in lst:
            for e in l:
                res.append(e)
        return res

    def one_hot_vector(self, label):
        res = [0] * len(self.classes)
        res[self.classes.index(label)] = 1
        return np.array(res, dtype=np.float32)

    def get_classes_count(self):
        return len(self.classes)


def one_hot_mnist(values):
    res = []

    for v in values:
        elem = [0]*10
        elem[v] = 1
        res.append(elem)

    return np.array(res).astype('float32')


def one_hot_cifar(values):
    res = []

    for v in values:
        elem = [0]*10
        elem[v[0]] = 1
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
    x_train = (x_train.reshape((len(x_train), 32, 32, 3)) / 255.0).astype(
        'float32'
    ).transpose(0, 3, 1, 2)
    x_test = (x_test.reshape((len(x_test), 32, 32, 3)) / 255.0).astype(
        'float32'
    ).transpose(0, 3, 1, 2)

    return x_train, one_hot_cifar(y_train), x_test, one_hot_cifar(y_test)


def load_mnist_for_ae():
    x_train, y_train, x_test, y_test = load_mnist()
    return x_train, x_train, x_test, x_test


def load_cifar10_for_ae():
    x_train, y_train, x_test, y_test = load_cifar10()
    return x_train, x_train, x_test, x_test


def get_loaders(load_data):
    x_train, y_train, x_test, y_test = load_data
    return Loader(x_train, y_train), Loader(x_test, y_test)
