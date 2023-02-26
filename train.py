import argparse
import json
import torch
import pytorch_lightning as pl
from argparse import Namespace
from model import DTSModel
from datamodules.csvdatamodule import CsvDataModule
from datamodules.hivedatamodule import HiveDataModule
from datamodules.sqldatamodule import SqlDataModule
from datamodules.s3datamodule import S3DataModule
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import importlib
from utils import data_source_dictionary

def train(config):
    pl.seed_everything(42, workers=True)
    logger = TensorBoardLogger('logs/', name=config.model['model_name'])
    device = torch.device('cuda') if config.train['accelerator']=='gpu' else torch.device('cpu')

    # Create LightningDataModule
    data_source_module = importlib.import_module(data_source_dictionary[config.data['data']][0])
    data_source = getattr(data_source_module, data_source_dictionary[config.data['data']][1])
    data_module = data_source(config.data['data_params'])

    # Create LightningModule
    model = DTSModel(config.model, device)
    model.to(device)

    # Callback to save the model checkpoint
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='./checkpoints/',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
        save_last=True,
        every_n_epochs=1  # Save checkpoint every 100 steps
    )

    # Create Trainer
    trainer = pl.Trainer(
        accelerator=config.train['accelerator'],
        devices=config.train['devices'],
        strategy=config.train['strategy'],
        max_epochs=config.train['max_epochs'],
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

"""
Command:

python train.py --config test_train_config.json

"""
