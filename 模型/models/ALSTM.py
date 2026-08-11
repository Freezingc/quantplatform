import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size if getattr(args, 'hidden_size', None) is not None else 256
        self.input_size = args.input_size if getattr(args, 'input_size', None) is not None else 158
        self.dropout = args.dropout if getattr(args, 'dropout', None) is not None else 0.1
        self.num_layers = args.num_layers if getattr(args, 'num_layers', None) is not None else 2
        self.rnn_type = args.rnn_type if getattr(args, 'rnn_type', None) is not None else "GRU"
        self._build_model()

    def _build_model(self):
        try:
            klass = getattr(nn, self.rnn_type.upper())
        except Exception as e:
            raise ValueError("unknown rnn_type `%s`" % self.rnn_type) from e
        self.net = nn.Sequential()
        self.net.add_module("fc_in", nn.Linear(in_features=self.input_size, out_features=self.hidden_size))
        self.net.add_module("act", nn.Tanh())
        self.rnn = klass(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.fc_out = nn.Linear(in_features=self.hidden_size * 2, out_features=1)
        self.att_net = nn.Sequential()
        self.att_net.add_module(
            "att_fc_in",
            nn.Linear(in_features=self.hidden_size, out_features=int(self.hidden_size / 2)),
        )
        self.att_net.add_module("att_dropout", torch.nn.Dropout(self.dropout))
        self.att_net.add_module("att_act", nn.Tanh())
        self.att_net.add_module(
            "att_fc_out",
            nn.Linear(in_features=int(self.hidden_size / 2), out_features=1, bias=False),
        )
        self.att_net.add_module("att_softmax", nn.Softmax(dim=1))

    def forward(self, inputs):
        rnn_out, _ = self.rnn(self.net(inputs)) # inputs: B T hidden_size
        rnn_out = rnn_out[:,-1,:] # B 1 H
        attention_score = self.att_net(rnn_out) # B 1
        out_att = torch.mul(rnn_out, attention_score)
        out = torch.cat((rnn_out, out_att), dim=1) 
        out = self.fc_out(out)  
        return out.squeeze()