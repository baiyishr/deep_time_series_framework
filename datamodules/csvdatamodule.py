import importlib
from typing import Optional
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from utils import dataset_map


class CsvDataModule(LightningDataModule):
    def __init__(self, config):
        self.config = config
        self.train_data = pd.read_csv(self.config["train_path"]).values
        self.val_data = pd.read_csv(self.config["val_path"]).values
        self.test_data = pd.read_csv(self.config["test_path"]).values

        # Define dataset
        module = importlib.import_module(dataset_map[config["dataset"]][0])
        self.dataset = getattr(module, dataset_map[config["dataset"]][1])

        # normalization
        self.input_scaler = StandardScaler() #MinMaxScaler(feature_range=(-1, 1))
        self.input_scaler.fit(self.train_data[:, 1:6])
        self.target_scaler = StandardScaler() #MinMaxScaler(feature_range=(-1, 1))
        self.target_scaler.fit(self.train_data[:, 4].reshape(-1, 1))

    def train_dataloader(self):
        # Create PyTorch dataset for training data
        train_dataset = self.dataset(
            self.train_data, self.config, 
            self.input_normalize,self.target_normalize)

        # Create PyTorch dataloader for training data
        return DataLoader(
            train_dataset, batch_size=self.config["batch_size"], num_workers=4, shuffle=False)

    def val_dataloader(self):
        # Create PyTorch dataset for val data
        val_dataset = self.dataset(
            self.val_data, self.config, 
            self.input_normalize,self.target_normalize)

        # Create PyTorch dataloader for test data
        return DataLoader(
            val_dataset, batch_size=self.config["batch_size"], num_workers=4, shuffle=False)

    def test_dataloader(self):
        # Create PyTorch dataset for test data
        test_dataset = self.dataset(
            self.test_data, self.config, 
            self.input_normalize,self.target_normalize)

        # Create PyTorch dataloader for test data
        return DataLoader(
            test_dataset, batch_size=self.config["batch_size"], num_workers=4, shuffle=False)
    
    def _log_hyperparams(self, *args, **kwargs):
        pass  # do nothing

    def input_normalize(self, x):
        # Normalize input data using fitted scaler
        x_norm = self.input_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return x_norm

    def target_normalize(self, x):
        # Normalize input data using fitted scaler
        x_norm = self.target_scaler.transform(x.reshape(-1, x.shape[-1])).reshape(x.shape)
        return x_norm
    
    def input_denormalize(self, x_norm):
        # Denormalize input data using fitted scaler
        x = self.input_scaler.inverse_transform(x_norm.reshape(-1, x_norm.shape[-1])).reshape(x_norm.shape)
        return x

    def target_denormalize(self, x_norm):
        # Denormalize input data using fitted scaler
        x = self.target_scaler.inverse_transform(x_norm.reshape(-1, x_norm.shape[-1])).reshape(x_norm.shape)
        return x


