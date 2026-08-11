import numpy as np
import pandas as pd
import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size if getattr(args, 'hidden_size', None) is not None else 64
        self.input_size = args.input_size if getattr(args, 'input_size', None) is not None else 158
        self.dropout = args.dropout if getattr(args, 'dropout', None) is not None else 0.1
        self.num_layers = args.num_layers if getattr(args, 'num_layers', None) is not None else 2
        self.rnn_type = args.rnn_type if getattr(args, 'rnn_type', None) is not None else "GRU"
        self.nhead = args.nhead if getattr(args, 'nhead', None) is not None else 4
        self.feature_layer = nn.Linear(self.input_size, self.hidden_size)
        self.pos_encoder = PositionalEncoding(self.hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.hidden_size, nhead=self.nhead, dropout=self.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.num_layers)
        self.decoder_layer = nn.Linear(self.hidden_size, 1)
        
    def forward(self, src):
        # src B T D 
        src = self.feature_layer(src)  
        src = src.transpose(1, 0) 
        mask = None
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, mask)  
        output = output.transpose(1, 0)[:, -1, :]  
        output = self.decoder_layer(output)  
        return output.squeeze()
