import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.init as init

class CustomRNN(pl.LightningModule):
    def __init__(self, device, **params):
        super().__init__()

        self.to_device = device
        self.input_size = params['input_size']
        self.hidden_size = params['hidden_size']
        self.output_size = params['output_size']
        self.num_layers = params['num_layers']

        self.rnn = nn.LSTM(self.input_size,
                          self.hidden_size,
                          self.num_layers,
                          batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(self.hidden_size, self.output_size)

        # Initialize LSTM weights using Xavier initialization
        for name, param in self.rnn.named_parameters():
            if 'weight' in name:
                init.xavier_uniform_(param)

    def forward(self, x):
        out, _ = self.rnn(x.float())
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out
