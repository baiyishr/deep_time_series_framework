import torch
import torch.nn as nn
import sys
from models.custom_rnn import CustomRNN
from models.time_series_transformer import TSTModel
import pytorch_lightning as pl


class DTSModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        self.model_name = config.model_name
        self.model_params = model_params

        self.loss_fn_type = config.loss_fn_type
        self.loss_fn_params = config.loss_params

        self.model = self.get_model()
        self.loss_fn = self.get_loss_fn()

    def get_model(self):
        model_class = getattr(models, self.model_type)
        return model_class(**self.model_params)

    def get_loss_fn(self):
        loss_fn_class = getattr(losses, self.loss_fn_type)
        return loss_fn_class(**self.loss_fn_params, num_classes=self.num_classes)

    def forward(self, inputs):
        return self.model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs.logits, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
