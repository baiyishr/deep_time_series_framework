import torch.nn as nn


def mse_loss(y_pred, y_true):
    return nn.MSELoss()(y_pred, y_true)


def cross_entropy_loss(y_pred, y_true):
    return nn.CrossEntropyLoss()(y_pred, y_true)
