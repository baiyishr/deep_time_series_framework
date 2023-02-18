import argparse
import json
import pytorch_lightning as pl
from argparse import Namespace
from model import DTSModel
from datamodule import DataModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


def train(config):
    pl.seed_everything(42, workers=True)
    logger = TensorBoardLogger('logs/', name=config.model.model_name)

    # Create LightningDataModule
    data_module = DataModule(**config.data)

    # Create LightningModule
    model = DTSModel(config.model)

    # Callback to save the model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        period=100  # Save checkpoint every 100 steps
    )

    # Create Trainer
    trainer = pl.Trainer(
        accelerator=config.train.accelerator,
        devices=config.train.devices,
        max_epochs=config.max_epochs,
        callbacks=[
            EarlyStopping(monitor='val_loss'),
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback
        ],
        logger=logger,
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON configuration file')
    args = parser.parse_args()

    # Load configuration from JSON file
    with open(args.config) as f:
        config_dict = json.load(f)
    config = Namespace(**config_dict)

    # Train the model
    train(config)
