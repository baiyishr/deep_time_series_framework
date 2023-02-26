from torch.utils.data import Dataset

class RNNDataset(Dataset):
    def __init__(self, data, data_config, input_transform=None, target_transform=None):
        self.data = data
        self.seq_len = data_config['seq_len']
        self.tgt_len = data_config['tgt_len']
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data) - self.seq_len - self.tgt_len

    def __getitem__(self, idx):
        # [seq_len, num_features]
        x = self.data[idx: idx+self.seq_len, 1:6].astype(float)
        # [tgt_len, close price]
        y = self.data[idx+self.seq_len: idx +
                      self.seq_len+self.tgt_len, 4].astype(float)
        

        if self.input_transform:
            x = self.input_transform(x)
        if self.target_transform:
            y = self.target_transform(y.reshape(-1, 1)).squeeze(1)

        return x, y
