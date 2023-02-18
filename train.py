import argparse
import json
import pytorch_lightning as pl
from argparse import Namespace
from model import DTSModel
from .models.datamodule import DataModule


def train(config):
    # Create LightningModule
    model = DTSModel(num_labels=config.num_labels,
                     lr=config.lr, model_name=config.model_name)

    # Create LightningDataModule
    data_module = DataModule(
        batch_size=config.batch_size,
        model_name=config.model_name,
        train_path=config.train_path,
        val_path=config.val_path,
        test_path=config.test_path)

    # Create Trainer
    trainer = pl.Trainer(gpus=config.gpus, max_epochs=config.max_epochs)

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
