from typing import Optional
import torch
import torch.nn as nn


class CustomRNN(nn.Module):
    def __init__(self, feature_size, num_classes, hidden_size=128, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(feature_size, hidden_size,
                          num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), self.rnn.hidden_size)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
