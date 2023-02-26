from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Tuple

class TSTDataset(Dataset):
    """
    Dataset class used for TST models
    """

    def __init__(self, data, data_config, input_transform=None, target_transform=None):
        """
        Args:

        """
        super().__init__()
        self.data = data
        self.enc_seq_len = data_config['enc_seq_len']
        self.dec_seq_len = data_config['dec_seq_len']
        self.tgt_seq_len = data_config['tgt_seq_len']
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.data) -self.enc_seq_len-self.dec_seq_len-1

    def __getitem__(self, idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) dec (the decoder input)
        3) tgt (the target)
        """
        # # For example: enc[0:30,1:6], dec[29:34,1:6], tgt[30:35,4]
        # enc = self.data[idx: idx+self.enc_seq_len, 
        #                 1:6].astype(float)
        # dec = self.data[idx+self.enc_seq_len-1 : 
        #                 idx+self.enc_seq_len+self.dec_seq_len-1, 
        #                 1:6].astype(float)
        # tgt = self.data[idx+self.enc_seq_len+self.dec_seq_len-self.tgt_seq_len : 
        #                 idx+self.enc_seq_len+self.tgt_seq_len
        #                 , 4].astype(float)

        # try more overlap: enc[0:30,1:6], dec[26:31,1:6], tgt[27:32,4]
        enc = self.data[idx: idx+self.enc_seq_len, 
                        1:6].astype(float)
        dec = self.data[idx+self.enc_seq_len-self.dec_seq_len+1 : 
                        idx+self.enc_seq_len+1, 
                        1:6].astype(float)
        tgt = self.data[idx+self.enc_seq_len+2-self.tgt_seq_len : 
                        idx+self.enc_seq_len+2
                        , 4].astype(float)

        if self.input_transform:
            enc = self.input_transform(enc)
            dec = self.input_transform(dec)
        if self.target_transform:
            tgt = self.target_transform(tgt.reshape(-1, 1)).squeeze(1)


        return (enc, dec), tgt
