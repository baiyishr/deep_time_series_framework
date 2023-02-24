import pytorch_lightning as pl
import torch


class TSTModel(pl.LightningModule):
    def __init__(self, num_labels, lr, model_name):
        super().__init__()

        self.lr = lr

    def forward(self, inputs):
        pass
