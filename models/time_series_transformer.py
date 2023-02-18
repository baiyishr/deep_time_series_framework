import pytorch_lightning as pl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class TSTModel(pl.LightningModule):
    def __init__(self, num_labels, lr, model_name):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels)
        self.lr = lr

    def forward(self, inputs):
        return self.model(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        outputs = self(inputs)
        loss = torch.nn.functional.cross_entropy(outputs.logits, labels)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=args.patience, factor=args.factor, mode=args.mode, verbose=True)
        return [optimizer], [scheduler]
