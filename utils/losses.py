import torch


def l2(y_pred, y_true):
    return ((y_pred - y_true) ** 2).mean()


def acc(y_pred, y_true):
    return (y_pred.argmax(dim=1) == y_true.argmax(dim=1)).sum().type(
        torch.FloatTensor
    ) / y_true.size(0)
