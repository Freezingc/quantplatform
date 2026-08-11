import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1D, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class BlockDiagonal(nn.Module):
    def __init__(self, in_features, out_features, num_blocks):
        super(BlockDiagonal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_blocks = num_blocks
        
        assert in_features % num_blocks == 0
        assert out_features % num_blocks == 0
        
        block_in_features = in_features // num_blocks
        block_out_features = out_features // num_blocks
        
        self.blocks = nn.ModuleList([
            nn.Linear(block_in_features, block_out_features)
            for _ in range(num_blocks)
        ])
     
    def forward(self, x):
        x = x.chunk(self.num_blocks, dim=-1)
        x = [block(x_i) for block, x_i in zip(self.blocks, x)]
        x = torch.cat(x, dim=-1)
        return x

class mLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, proj_factor=2):
        super(mLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.proj_factor = proj_factor
        
        assert hidden_size % num_heads == 0
        assert proj_factor > 0
        
        self.layer_norm = nn.LayerNorm(input_size)
        self.up_proj_left = nn.Linear(input_size, int(input_size * proj_factor))
        self.up_proj_right = nn.Linear(input_size, hidden_size)
        self.down_proj = nn.Linear(hidden_size, input_size)
        
        self.causal_conv = CausalConv1D(1, 1, 4)
        self.skip_connection = nn.Linear(int(input_size * proj_factor), hidden_size)
        
        self.Wq = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wk = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wv = BlockDiagonal(int(input_size * proj_factor), hidden_size, num_heads)
        self.Wi = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wf = nn.Linear(int(input_size * proj_factor), hidden_size)
        self.Wo = nn.Linear(int(input_size * proj_factor), hidden_size)
        
        self.group_norm = nn.GroupNorm(num_heads, hidden_size)

    def forward(self, x, prev_state):
        h_prev, c_prev, n_prev, m_prev = prev_state
        assert x.size(-1) == self.input_size
        x_norm = self.layer_norm(x)
        x_up_left = self.up_proj_left(x_norm)
        x_up_right = self.up_proj_right(x_norm)
        
        x_conv = F.silu(self.causal_conv(x_up_left.unsqueeze(1)).squeeze(1))
        x_skip = self.skip_connection(x_conv)
        
        q = self.Wq(x_conv)
        k = self.Wk(x_conv) / (self.head_size ** 0.5)
        v = self.Wv(x_up_left)
        
        i_tilde = self.Wi(x_conv)
        f_tilde = self.Wf(x_conv)
        o = torch.sigmoid(self.Wo(x_up_left))
        
        m_t = torch.max(f_tilde + m_prev, i_tilde)
        i = torch.exp(i_tilde - m_t)
        f = torch.exp(f_tilde + m_prev - m_t)
        
        c_t = f * c_prev + i * (v * k) # v @ k.T
        n_t = f * n_prev + i * k
        h_t = o * (c_t * q) / torch.max(torch.abs(n_t.T @ q), 1)[0] # o * (c @ q) / max{|n.T @ q|, 1}
        
        output = h_t
        output_norm = self.group_norm(output)
        output = output_norm + x_skip
        output = output * F.silu(x_up_right)
        output = self.down_proj(output)
        final_output = output + x
        
        return final_output, (h_t, c_t, n_t, m_t)
 
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size if getattr(args, 'input_size', None) is not None else 1
        self.hidden_size = args.hidden_size if getattr(args, 'hidden_size', None) is not None else 64
        self.nhead = args.nhead if getattr(args, 'nhead', None) is not None else 1
        self.batch_first = args.batch_first if getattr(args, 'batch_first', None) is not None else True
        self.proj_factor = args.proj_factor if getattr(args, 'proj_factor', None) is not None else 2
        self.layers = mLSTMBlock(self.input_size, self.hidden_size, self.nhead, self.proj_factor)

    def forward(self, x, state=None):
        assert x.ndim == 3
        if self.batch_first: 
            x = x.transpose(0, 1)
        seq_len, batch_size, _ = x.size()
        state = torch.zeros(4, batch_size, self.hidden_size)
        
        output = []
        for t in range(seq_len):
            x_t = x[t]
            x_t, state_tuple = self.layers(x_t, tuple(state.clone()))
            state = torch.stack(list(state_tuple))
            output.append(x_t)
        
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output[:,-1,:].squeeze()