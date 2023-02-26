import math
import pytorch_lightning as pl
import torch.nn as nn 
from torch import nn, Tensor
import torch.nn.functional as F
import torch


class TSTModel(pl.LightningModule):
    """
    Time Series Transformer
    Modified from: https://github.com/KasperGroesLudvigsen/influenza_transformer
    """
    def __init__(self, device, **params):
        """
        Args:
            input_size: int, number of input variables. 1 if univariate.
            dec_seq_len: int, the length of the input sequence fed to the decoder
            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val
            n_encoder_layers: int, number of stacked encoder layers in the encoder
            n_decoder_layers: int, number of stacked encoder layers in the decoder
            n_heads: int, the number of attention heads (aka parallel attention layers)
            dropout_encoder: float, the dropout rate of the encoder
            dropout_decoder: float, the dropout rate of the decoder
            dropout_pos_enc: float, the dropout rate of the positional encoder
            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder
            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder
            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__() 

        self.dec_seq_len = params['dec_seq_len']

        # Creating the three linear layers needed for the model
        self.encoder_input_layer = nn.Linear(
            in_features=params['input_size'], 
            out_features=params['dim_val'] 
            )

        self.decoder_input_layer = nn.Linear(
            in_features=params['input_size'],  #['num_predicted_features'],
            out_features=params['dim_val']
            )  
        
        self.linear_mapping = nn.Linear(
            in_features=params['dim_val'], 
            out_features=params['out_seq_len']   #['num_predicted_features']
            )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=params['dim_val'],
            dropout=params['dropout_pos_enc']
            )

        # The encoder layer used in the paper is identical to the one used by
        # Vaswani et al (2017) on which the PyTorch module is based.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=params['dim_val'], 
            nhead=params['n_heads'],
            dim_feedforward=params['dim_feedforward_encoder'],
            dropout=params['dropout_encoder'],
            batch_first=True
            )

        # Stack the encoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerEncoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=params['n_encoder_layers'], 
            norm=None
            )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=params['dim_val'],
            nhead=params['n_heads'],
            dim_feedforward=params['dim_feedforward_decoder'],
            dropout=params['dropout_decoder'],
            batch_first=True
            )

        # Stack the decoder layers in nn.TransformerDecoder
        # It seems the option of passing a normalization instance is redundant
        # in my case, because nn.TransformerDecoderLayer per default normalizes
        # after each sub-layer
        # (https://github.com/pytorch/pytorch/issues/24930).
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=params['n_decoder_layers'], 
            norm=None
            )

    def forward(self, inputs):
        """
        Returns a tensor of shape:
        [batch_size, target_sequence_length, num_predicted_features]
        
        Args:
            enc: [batch_size, enc_seq_len, num_features]
            dec: [batch_size, tgt_seq_len, num_features]
            enc_mask: the mask for the enc sequence to prevent the model from 
                      using data points from the target sequence
            dec_mask: the mask for the dec sequence to prevent the model from
                      using data points from the target sequence
        """
        enc, dec,  = inputs 

        # Pass throguh the input layer right before the encoder
        enc = self.encoder_input_layer(enc) # [batch_size, enc_seq_len, dim_val] regardless of number of input features

        # Pass through the positional encoding layer
        enc = self.positional_encoding_layer(enc) # shape: [batch_size, enc_seq_len, dim_val] regardless of number of input features

        # Pass through all the stacked encoder layers in the encoder
        # Masking is only needed in the encoder if input sequences are padded
        # which they are not in this time series use case, because all my
        # input sequences are naturally of the same length. 
        # (https://github.com/huggingface/transformers/issues/4083)
        enc = self.encoder( # shape: [batch_size, enc_seq_len, dim_val]
            src=enc
            )

        # Pass decoder input through decoder input layer
        decoder_output = self.decoder_input_layer(dec) # [batch_size, dec_seq_len, dim_val] regardless of number of input features

        #if src_mask is not None:
            #print("From model.forward(): Size of src_mask: {}".format(src_mask.size()))
        #if tgt_mask is not None:
            #print("From model.forward(): Size of tgt_mask: {}".format(tgt_mask.size()))

        # Pass throguh decoder - output shape: [batch_size, dec seq len, dim_val]
        decoder_output = self.decoder(
            tgt=decoder_output,
            memory=enc,
            tgt_mask=None, #dec_mask,
            memory_mask=None, #enc_mask
            )

        # Pass through linear mapping
        decoder_output = self.linear_mapping(decoder_output) # shape [batch_size, out_seq_len], 1 feature

        return decoder_output


class PositionalEncoder(nn.Module):
    """
    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=True
        ):

        """
        Parameters:
            dropout: the dropout rate
            max_seq_len: the maximum length of the input sequences
            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        # adapted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        if self.batch_first:
            pe = torch.zeros(1, max_seq_len, d_model)
            
            pe[0, :, 0::2] = torch.sin(position * div_term)
            
            pe[0, :, 1::2] = torch.cos(position * div_term)
        else:
            pe = torch.zeros(max_seq_len, 1, d_model)
        
            pe[:, 0, 0::2] = torch.sin(position * div_term)
        
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        if self.batch_first:
            x = x + self.pe[:,:x.size(1)]
        else:
            x = x + self.pe[:x.size(0)]

        return self.dropout(x)