import os
import torch
import boto3
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule


class S3Dataset(Dataset):
    def __init__(self, bucket_name, s3_prefix):
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.s3 = boto3.resource('s3')
        self.bucket = self.s3.Bucket(bucket_name)
        self.file_list = [obj.key for obj in self.bucket.objects.filter(
            Prefix=s3_prefix) if obj.key.endswith('.pt')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        key = self.file_list[idx]
        obj = self.bucket.Object(key)
        response = obj.get()
        body = response['Body'].read()
        tensor = torch.load(io.BytesIO(body))
        return tensor


class S3DataModule(LightningDataModule):
    def __init__(self, bucket_name, s3_prefix, batch_size=32):
        super().__init__()
        self.bucket_name = bucket_name
        self.s3_prefix = s3_prefix
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = S3Dataset(self.bucket_name, self.s3_prefix)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
