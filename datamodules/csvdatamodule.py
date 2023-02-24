import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class CsvDataModule(LightningDataModule):
    def __init__(self, config):
        self.config = config
        self.train_data = pd.read_csv(self.config["train_path"]).values
        self.val_data = pd.read_csv(self.config["val_path"]).values
        self.test_data = pd.read_csv(self.config["test_path"]).values

        # Define data transforms
        self.transform = None

    def train_dataloader(self):
        # Create PyTorch dataset for training data
        train_dataset = CsvDataset(
            self.train_data, self.config["seq_len"], self.config["tgt_len"])

        # Create PyTorch dataloader for training data
        return DataLoader(
            train_dataset, batch_size=self.config["batch_size"], shuffle=False)

    def val_dataloader(self):
        # Create PyTorch dataset for val data
        val_dataset = CsvDataset(
            self.val_data, self.config["seq_len"], self.config["tgt_len"])

        # Create PyTorch dataloader for test data
        return DataLoader(
            val_dataset, batch_size=self.config["batch_size"], shuffle=False)

    def test_dataloader(self):
        # Create PyTorch dataset for test data
        test_dataset = CsvDataset(
            self.test_data, self.config["seq_len"], self.config["tgt_len"])

        # Create PyTorch dataloader for test data
        return DataLoader(
            test_dataset, batch_size=self.config["batch_size"], shuffle=False)


class CsvDataset(Dataset):
    def __init__(self, data, seq_len, tgt_len, transform=None):
        self.data = data
        self.seq_len = seq_len
        self.tgt_len = tgt_len
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        # [seq_len, num_features]
        x = self.data[idx: idx+self.seq_len, 1:6].astype(float)
        # [tgt_len, close price]
        y = self.data[idx+self.seq_len: idx +
                      self.seq_len+self.tgt_len, 4].astype(float)

        if self.transform:
            x = self.transform(x)

        return x, y
