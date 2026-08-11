import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size if getattr(args, 'hidden_size', None) is not None else 790
        self.input_size = args.input_size if getattr(args, 'input_size', None) is not None else 158
        self.dropout = args.dropout if getattr(args, 'dropout', None) is not None else 0.1
        self.num_layers = args.num_layers if getattr(args, 'num_layers', None) is not None else 2

        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(self.hidden_size, 1)

    def forward(self, x): # x: B * T * D
        out, _ = self.rnn(x) # B * T * hidden_size
        out = self.fc_out(out[:,-1,:]) # B * 1
        return out.squeeze()