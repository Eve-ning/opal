import torch


# SEE Desmos: https://www.desmos.com/calculator/f7tlysna7c
def adj_inv_sigmoid(x):
    """ Adjusts the accuracy such that it's more linear for smoother learning """
    return -torch.log(1 / ((x / 2.5) + 0.5) - 1)


def adj_sigmoid(x):
    """ Inverses the accuracy adjustment """
    return -(0.5 * torch.exp(-x) - 0.5) / (0.4 * (torch.exp(-x) + 1))


# assert adj_inv_sigmoid(adj_sigmoid(torch.ones([1]))).item() - 1 < 1e-5
