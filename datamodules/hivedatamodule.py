from pyhive import hive
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class HiveDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        features = row.drop('target').values.astype(np.float32)
        target = row['target'].astype(np.float32)
        return features, target


class HiveDataModule(LightningDataModule):
    def __init__(self, config):
        self.config = config

    def prepare_data(self):
        conn = hive.Connection(host=self.config['host'],
                               port=self.config['port'],
                               database=self.config['database'],
                               username=self.config['username'],
                               password=self.config['password'])

        query = f"""
            SELECT {', '.join(self.config['features'])}, {self.config['target']} 
            FROM {self.config['table']}
            WHERE ds >= '{self.config['train_start_date']}' AND ds < '{self.config['train_end_date']}'
        """

        self.train_df = pd.read_sql(query, conn)

        query = f"""
            SELECT {', '.join(self.config['features'])}, {self.config['target']} 
            FROM {self.config['table']}
            WHERE ds >= '{self.config['val_start_date']}' AND ds < '{self.config['val_end_date']}'
        """

        self.val_df = pd.read_sql(query, conn)

        query = f"""
            SELECT {', '.join(self.config['features'])}, {self.config['target']} 
            FROM {self.config['table']}
            WHERE ds >= '{self.config['test_start_date']}' AND ds < '{self.config['test_end_date']}'
        """

        self.test_df = pd.read_sql(query, conn)

    def setup(self):
        self.train_dataset = HiveDataset(self.train_df)
        self.val_dataset = HiveDataset(self.val_df)
        self.test_dataset = HiveDataset(self.test_df)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config['batch_size'], num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.config['batch_size'], num_workers=4)
