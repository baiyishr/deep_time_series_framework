import torch
import torch.nn as nn
import sys
from .model import CustomRNN
from .model import TSTModel


class DTSModel(nn.Module):
    def __init__(self, model_type, model_params, loss_fn_type, loss_fn_params):
        super().__init__()

        self.model_type = model_type
        self.model_params = model_params

        self.loss_fn_type = loss_fn_type
        self.loss_fn_params = loss_fn_params

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
