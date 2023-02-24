from torch.utils.data import Dataset
import pandas as pd
import torch
from typing import Tuple

class TransformerDataset(Dataset):
    """
    Dataset class used for TST models
    Modified from https://github.com/KasperGroesLudvigsen/influenza_transformer/
    """

    def __init__(self,
                 data: torch.tensor,
                 indices: list,
                 enc_seq_len: int,
                 dec_seq_len: int,
                 target_seq_len: int
                 ) -> None:
        """
        Args:
            data: [number of samples, number of variables]
            indices: list[(start_index, end_index)]  
            enc_seq_len: length of the input sequence 
            tgt_seq_len: length of the target sequence 
            tgt_idx: The index position of the target variable in data
        """
        super().__init__()
        self.indices = indices
        self.data = data
        self.enc_seq_len = enc_seq_len
        self.dec_seq_len = dec_seq_len
        self.tgt_seq_len = target_seq_len

    def __len__(self):

        return len(self.indices)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) dec (the decoder input)
        3) tgt (the target)
        """
        start_idx = self.indices[index][0]
        end_idx = self.indices[index][1]
        sequence = self.data[start_idx:end_idx]

        src, dec, tgt = self.get_src_tgt(
            sequence=sequence,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            tgt_seq_len=self.tgt_seq_len
        )

        return src, dec, tgt

    def get_src_tgt(
        self,
        sequence: torch.Tensor,
        enc_seq_len: int,
        dec_seq_len: int,
        tgt_seq_len: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Generate the src (encoder input), dec (decoder input) and tgt (the target)
        sequences from a sequence. 
        Args:
            sequence: tensor, a 1D tensor of length n where 
                    n = encoder input length + target sequence length  
            enc_seq_len: int, the desired length of the input to the transformer encoder
            tgt_seq_len: int, the desired length of the target sequence (the 
                            one against which the model output is compared)
        Return: 
            src: tensor, 1D, used as input to the transformer model
            dec: tensor, 1D, used as input to the transformer model
            tgt: tensor, 1D, the target sequence against which the model output
                is compared when computing loss. 

        """
        assert len(sequence) == enc_seq_len + tgt_seq_len, \
            "Sequence length does not equal (input length + target length)"

        # encoder input
        src = sequence[:enc_seq_len]

        # decoder input. As per the paper, it must have the same dimension as the
        # target sequence, and it must contain the last value of src, and all
        # values of tgt except the last (i.e. it must be shifted right by 1)
        dec = sequence[enc_seq_len-1:len(sequence)-1]

        assert len(
            dec) == dec_seq_len, "Length of dec does not match decoder input sequence length"

        # The target sequence against which the model output will be compared to compute loss
        tgt = sequence[-tgt_seq_len:]

        assert len(
            tgt) == tgt_seq_len, "Length of tgt does not match target sequence length"

        # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len]
        return src, dec, tgt.squeeze(-1)
