import numpy as np
import pandas as pd
import copy
import torch
import torch.nn as nn
import torch.optim as optim


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.hidden_size = args.hidden_size if getattr(args, 'hidden_size', None) is not None else 64
        self.input_size = args.input_size if getattr(args, 'input_size', None) is not None else 1
        self.dropout = args.dropout if getattr(args, 'dropout', None) is not None else 0.1
        self.num_layers = args.num_layers if getattr(args, 'num_layers', None) is not None else 2
        self.rnn = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
        )
        self.transformation = nn.Linear(self.hidden_size, self.hidden_size)
        self.a = nn.Parameter(torch.randn(self.hidden_size * 2, 1))
        self.a.requires_grad = True
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU()
        self.softmax = nn.Softmax(dim=1)

    def cal_attention(self, x, y):
        x = self.transformation(x)
        y = self.transformation(y)

        sample_num = x.shape[0]
        dim = x.shape[1]
        e_x = x.expand(sample_num, sample_num, dim)
        e_y = torch.transpose(e_x, 0, 1)
        attention_in = torch.cat((e_x, e_y), 2).view(-1, dim * 2)
        self.a_t = torch.t(self.a)
        attention_out = self.a_t.mm(torch.t(attention_in)).view(sample_num, sample_num)
        attention_out = self.leaky_relu(attention_out)
        att_weight = self.softmax(attention_out)
        return att_weight

    def forward(self, x):
        out, _ = self.rnn(x) # x: B * T * D
        hidden = out[:, -1, :] #batch_size * hidden_size
        att_weight = self.cal_attention(hidden, hidden)
        hidden = att_weight.mm(hidden) + hidden
        hidden = self.fc(hidden) 
        hidden = self.leaky_relu(hidden) #batch_size * hidden_size
        hidden = self.fc_out(hidden) #batch_size * 1
        return hidden.squeeze()
