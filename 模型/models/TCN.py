import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, input_size, outputs_size, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # 第一个卷积层：输入通道数是input_size，输出通道数是outputs_size
        self.conv1 = weight_norm(
            nn.Conv1d(input_size, outputs_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        # 第二个卷积层：输入通道数是outputs_size，输出通道数是outputs_size
        self.conv2 = weight_norm(
            nn.Conv1d(outputs_size, outputs_size, kernel_size, stride=stride, padding=padding, dilation=dilation)
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net1 = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1
        )
        self.net2 = nn.Sequential(
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x的形状: (batch_size, features, seq_len)
        # print(f"TemporalBlock input: {x.shape}")
        x = self.net1(x)
        # print(f"After net1: {x.shape}")
        x = self.net2(x)
        # print(f"After net2: {x.shape}")
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.seq_len = args.seq_len if getattr(args, 'seq_len', None) is not None else 30
        self.input_size = args.input_size if getattr(args, 'input_size', None) is not None else 158
        self.outputs_size = args.outputs_size if getattr(args, 'outputs_size', None) is not None else 256
        self.dilation = args.dilation if getattr(args, 'dilation', None) is not None else 4
        self.kernel_size = args.kernel_size if getattr(args, 'kernel_size', None) is not None else 2
        self.stride = args.stride if getattr(args, 'stride', None) is not None else 1
        self.padding = (self.kernel_size - 1) * self.dilation
        self.dropout = args.dropout if getattr(args, 'dropout', None) is not None else 0.1

        # 创建多个TemporalBlock层
        self.tcn_layers = nn.ModuleList([
            TemporalBlock(
                self.input_size if i == 0 else self.outputs_size,
                self.outputs_size,
                self.kernel_size,
                stride=self.stride,
                dilation=self.dilation * (2 ** i),
                padding=self.padding,
                dropout=self.dropout,
            ) for i in range(1)  # 使用2个TemporalBlock
        ])
        
        # 最终的线性层，输出单个值
        self.final_linear = nn.Linear(self.outputs_size, 1)

    def forward(self, x):
        # 输入形状: (stock_num, days, features)
        # print(f"Model input shape: {x.shape}")
        
        # 转换为 (stock_num, features, days) 给Conv1d使用
        x = x.transpose(1, 2)  # (stock_num, features, days)
        # print(f"After transpose: {x.shape}")
        
        # 通过多个TemporalBlock
        for i, tcn_layer in enumerate(self.tcn_layers):
            x = tcn_layer(x)
            # print(f"After TCN layer {i+1}: {x.shape}")
        
        # 转换为 (stock_num, days, outputs_size)
        x = x.transpose(1, 2)  # (stock_num, days, outputs_size)
        # print(f"After final transpose: {x.shape}")
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]  # (stock_num, outputs_size)
        # print(f"After taking last timestep: {x.shape}")
        
        # 通过最终的线性层
        x = self.final_linear(x)  # (stock_num, 1)
        # print(f"Final output shape: {x.shape}")
        
        # 压缩为1维数组 (stock_num,)
        x = x.squeeze(-1)
        # print(f"Final squeezed output shape: {x.shape}")
        
        return x