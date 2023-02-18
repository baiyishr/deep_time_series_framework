import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.transforms import Compose


class CsvDataModule:
    def __init__(self, config):
        self.config = config

    def setup(self, stage=None):
        # Load data from CSV files
        self.train_data = pd.read_csv(self.config["train_csv_file"])
        self.val_data = pd.read_csv(self.config["val_csv_file"])
        self.test_data = pd.read_csv(self.config["test_csv_file"])

        # Define data transforms
        self.transform = Compose([])

    def train_dataloader(self):
        # Create PyTorch dataset for training data
        train_dataset = CsvDataset(
            self.train_data, self.config["features"], self.config["target"], self.transform)

        # Split training data into train and validation sets
        train_size = int(len(train_dataset) * self.config["train_val_split"])
        val_size = len(train_dataset) - train_size
        train_set, val_set = random_split(
            train_dataset, [train_size, val_size])

        # Create PyTorch dataloader for training data
        train_loader = DataLoader(
            train_set, batch_size=self.config["batch_size"], shuffle=True)

        # Create PyTorch dataloader for validation data
        val_loader = DataLoader(
            val_set, batch_size=self.config["batch_size"], shuffle=False, sampler=SubsetRandomSampler(range(val_size)))

        return {"train": train_loader, "val": val_loader}

    def test_dataloader(self):
        # Create PyTorch dataset for test data
        test_dataset = CsvDataset(
            self.test_data, self.config["features"], self.config["target"], self.transform)

        # Create PyTorch dataloader for test data
        test_loader = DataLoader(
            test_dataset, batch_size=self.config["batch_size"], shuffle=False)

        return test_loader


class CsvDataset(Dataset):
    def __init__(self, data, features, target, transform=None):
        self.data = data
        self.features = features
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.loc[idx, self.features]
        y = self.data.loc[idx, self.target]

        if self.transform:
            x = self.transform(x)

        return x, y
