from typing import Optional
from pyhive import hive
from hive_helper import HiveHelper
from torch.utils.data import DataLoader, Dataset as TorchDataset
import pytorch_lightning as pl
import json


class DataModule(pl.LightningDataModule):
    def __init__(self, config_file: str):
        super().__init__()
        with open(config_file, 'r') as f:
            config = json.load(f)
        self.config = config

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        hh = HiveHelper(
            host=self.config['hive']['host'],
            port=self.config['hive']['port'],
            database=self.config['hive']['database'],
            username=self.config['hive']['username'],
            password=self.config['hive']['password']
        )

        # Load training data from Hive table
        train_query = f"""
            SELECT {', '.join(self.config['features'])}
            FROM {self.config['hive']['table']}
            WHERE ds BETWEEN '{self.config['train_start_date']}' AND '{self.config['train_end_date']}'
        """
        train_data = hh.execute_query(train_query)
        self.train_data = MyDataset(train_data)

        # Load validation data from Hive table
        val_query = f"""
            SELECT {', '.join(self.config['features'])}
            FROM {self.config['hive']['table']}
            WHERE ds BETWEEN '{self.config['val_start_date']}' AND '{self.config['val_end_date']}'
        """
        val_data = hh.execute_query(val_query)
        self.val_data = MyDataset(val_data)

        # Load test data from Hive table
        test_query = f"""
            SELECT {', '.join(self.config['features'])}
            FROM {self.config['hive']['table']}
            WHERE ds BETWEEN '{self.config['test_start_date']}' AND '{self.config['test_end_date']}'
        """
        test_data = hh.execute_query(test_query)
        self.test_data = MyDataset(test_data)

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config['batch_size'])

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.config['batch_size'])

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.config['batch_size'])
