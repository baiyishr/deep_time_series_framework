# pylint: disable=too-many-ancestors
import importlib
import torch
import pytorch_lightning as pl
from utils import model_dictionary, loss_dictionary


class DTSModel(pl.LightningModule):
    def __init__(self, config, device):
        super().__init__()

        self.to_device = device
        self.model_name = config['model_name']
        self.model_params = config['model_params']

        self.loss_fn_type = config['loss_fn_type']
        self.loss_fn_params = config['loss_params']

        self.model = self.get_model()
        self.loss_fn = self.get_loss_fn()
        self.float()

    def get_model(self):
        module = importlib.import_module(model_dictionary[self.model_name][0])
        model_class = getattr(module, model_dictionary[self.model_name][1])
        return model_class(self.to_device, **self.model_params)

    def get_loss_fn(self):
        module = importlib.import_module(loss_dictionary[self.loss_fn_type][0])
        loss_fn_class = getattr(module, loss_dictionary[self.loss_fn_type][1])
        return loss_fn_class(**self.loss_fn_params)

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('val_loss', loss)
    
    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, labels)
        self.log('test_loss', loss)
        
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),lr=self.model_params['lr'])
        return optimizer
