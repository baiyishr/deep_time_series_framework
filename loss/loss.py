import torch.nn as nn


class MSE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        loss = nn.MSELoss()(preds.float(), targets.float())
        return loss.float()


class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        loss = nn.CrossEntropyLoss()(preds.float(), targets.float())
        return loss.float()
