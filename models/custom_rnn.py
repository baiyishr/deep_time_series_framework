import torch
import torch.nn as nn
import pytorch_lightning as pl


class CustomRNN(pl.LightningModule):
    def __init__(self, **params):
        super().__init__()
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']
        self.num_layers = params['num_layers']
        self.rnn = nn.RNN(self.input_size,
                          self.hidden_size,
                          self.num_layers,
                          batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size)
        out, hidden = self.rnn(x, hidden)
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)
