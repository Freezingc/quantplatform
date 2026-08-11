import numpy as np
import pandas as pd
from typing import Text, Union
import copy
import torch
import torch.nn as nn
import torch.optim as optim

class CNNEncoderBase(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        """Build a basic CNN encoder

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        kernel_size : int
            The size of convolutional kernels
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size

        # set padding to ensure the same length
        # it is correct only when kernel_size is odd, dilation is 1, stride is 1
        self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, padding=(kernel_size - 1) // 2)

    def forward(self, x):
        x = x.view(x.shape[0], -1, self.input_dim).permute(0, 2, 1)
        y = self.conv(x)  
        y = y.permute(0, 2, 1)  

        return y


class KRNNEncoderBase(nn.Module):
    def __init__(self, input_dim, output_dim, dup_num, rnn_layers, dropout):
        """Build K parallel RNNs

        Parameters
        ----------
        input_dim : int
            The input dimension
        output_dim : int
            The output dimension
        dup_num : int
            The number of parallel RNNs
        rnn_layers: int
            The number of RNN layers
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dup_num = dup_num
        self.rnn_layers = rnn_layers
        self.dropout = dropout

        self.rnn_modules = nn.ModuleList()
        for _ in range(dup_num):
            self.rnn_modules.append(nn.GRU(input_dim, output_dim, num_layers=self.rnn_layers, dropout=dropout))

    def forward(self, x):
        batch_size, seq_len, input_dim = x.shape
        x = x.permute(1, 0, 2)

        hids = []
        for rnn in self.rnn_modules:
            h, _ = rnn(x)  
            hids.append(h)
        hids = torch.stack(hids, dim=-1)
        hids = hids.view(seq_len, batch_size, self.output_dim, self.dup_num)
        hids = hids.mean(dim=3)
        hids = hids.permute(1, 0, 2)

        return hids


class CNNKRNNEncoder(nn.Module):
    def __init__(
        self, cnn_input_dim, cnn_output_dim, cnn_kernel_size, rnn_output_dim, rnn_dup_num, rnn_layers, dropout):
        """Build an encoder composed of CNN and KRNN

        Parameters
        ----------
        cnn_input_dim : int
            The input dimension of CNN
        cnn_output_dim : int
            The output dimension of CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_output_dim : int
            The output dimension of KRNN
        rnn_dup_num : int
            The number of parallel duplicates for KRNN
        rnn_layers : int
            The number of RNN layers
        """
        super().__init__()

        self.cnn_encoder = CNNEncoderBase(cnn_input_dim, cnn_output_dim, cnn_kernel_size)
        self.krnn_encoder = KRNNEncoderBase(cnn_output_dim, rnn_output_dim, rnn_dup_num, rnn_layers, dropout)

    def forward(self, x):
        cnn_out = self.cnn_encoder(x)
        krnn_out = self.krnn_encoder(cnn_out)

        return krnn_out


class Model(nn.Module):
    def __init__(self, args):
        """Build a KRNN model

        Parameters
        ----------
        fea_dim : int
            The feature dimension
        cnn_dim : int
            The hidden dimension of CNN
        cnn_kernel_size : int
            The size of convolutional kernels
        rnn_dim : int
            The hidden dimension of KRNN
        rnn_dups : int
            The number of parallel duplicates
        rnn_layers: int
            The number of RNN layers
        """
        super().__init__()
        self.cnn_input_dim = args.cnn_input_dim if getattr(args, 'cnn_input_dim', None) is not None else 1
        self.cnn_output_dim = args.cnn_output_dim if getattr(args, 'cnn_output_dim', None) is not None else 10
        self.cnn_kernel_size = args.cnn_kernel_size if getattr(args, 'cnn_kernel_size', None) is not None else 1
        self.rnn_output_dim = args.rnn_output_dim if getattr(args, 'rnn_output_dim', None) is not None else 1
        self.rnn_dup_num = args.rnn_dup_num if getattr(args, 'rnn_dup_num', None) is not None else 1
        self.rnn_layers = args.rnn_layers if getattr(args, 'rnn_layers', None) is not None else 1
        self.dropout = args.dropout if getattr(args, 'dropout', None) is not None else 0.0
        self.encoder = CNNKRNNEncoder(
            cnn_input_dim=self.cnn_input_dim,
            cnn_output_dim=self.cnn_output_dim,
            cnn_kernel_size=self.cnn_kernel_size,
            rnn_output_dim=self.rnn_output_dim,
            rnn_dup_num=self.rnn_dup_num,
            rnn_layers=self.rnn_layers,
            dropout=self.dropout,
        )

        self.out_fc = nn.Linear(self.rnn_output_dim, 1)

    def forward(self, x):
        encode = self.encoder(x)
        out = self.out_fc(encode[:, -1, :]).squeeze()

        return out