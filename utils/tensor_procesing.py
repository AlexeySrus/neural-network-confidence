import torch


def flatten(x):
    n = 1
    for d in x.shape[1:]:
        n *= d
    return x.view(x.size(0), n)


