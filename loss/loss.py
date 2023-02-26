import torch.nn as nn
import torch


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        assert preds.shape==targets.shape, "preds tensor shape is different with targets shape"
        loss = nn.MSELoss()(preds.float(), targets.float())
        return loss.float()


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        loss = nn.CrossEntropyLoss()(preds.float(), targets.float())
        return loss.float()
