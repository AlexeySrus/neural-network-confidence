import torch
from torch.nn.functional import binary_cross_entropy
import numpy as np


def cross_entropy(predictions, targets, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce


def SoftmaxL1(predictions, targets):
    return np.abs(predictions - targets).sum() / 2


def classification_with_confidence(x, basic_net, ae_net, conf_f=SoftmaxL1):
    y1 = basic_net(x)[0]
    x_gen = ae_net(x)
    y2 = basic_net(x_gen)[0]
    bn = np.abs(conf_f(
        y1.detach().to('cpu').numpy(),
        y2.detach().to('cpu').numpy()
    ))
    return y1, y2, 1 - bn, x_gen
