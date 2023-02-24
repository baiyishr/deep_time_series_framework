import mysql.connector
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class SqlDataModule(LightningDataModule):
    def __init__(self, config):
        self.config = config

        # Set up database connection
        self.cnx = mysql.connector.connect(
            user=config['user'],
            password=config['password'],
            host=config['host'],
            database=config['database']
        )

        # Set up query to retrieve data
        self.query = config['query']

        # Set up column names
        self.column_names = config['column_names']

    def __del__(self):
        # Close database connection when done
        self.cnx.close()

    def get_dataset(self, split):
        # Add WHERE clause to query based on split
        split_column = self.config['split_column']
        split_value = self.config['split_values'][split]
        split_filter = f"{split_column} = '{split_value}'"
        query = f"{self.query} WHERE {split_filter}"

        # Retrieve data from database and return as PyTorch Dataset
        df = pd.read_sql_query(query, self.cnx)
        return SqlDataset(df, self.column_names)


class SqlDataset(Dataset):
    def __init__(self, df, column_names):
        self.df = df[column_names]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx].to_dict()

    def get_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)


# config
# config = {
#     'user': 'your_username',
#     'password': 'your_password',
#     'host': 'your_host',
#     'database': 'your_database',
#     'query': 'SELECT * FROM your_table',
#     'column_names': ['column1', 'column2', 'column3'],
#     'split_column': 'split_column',
#     'split_values': {
#         'train': 'train_value',
#         'val': 'val_value',
#         'test': 'test_value'
#     }
# }
