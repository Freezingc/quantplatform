import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Dense multi-head graph attention with adjacency mask."""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, adj_mask):
        # x: [N, Din], adj_mask: [N, N] bool
        n = x.size(0)
        h = self.proj(x).view(n, self.num_heads, self.out_dim)  # [N, H, D]

        src = (h * self.att_src.unsqueeze(0)).sum(dim=-1)  # [N, H]
        dst = (h * self.att_dst.unsqueeze(0)).sum(dim=-1)  # [N, H]
        e = self.leaky_relu(src.unsqueeze(1) + dst.unsqueeze(0))  # [N, N, H]

        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(~adj_mask.unsqueeze(-1), neg_inf)
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)

        out = torch.einsum("ijh,jhd->ihd", alpha, h)  # [N, H, D]
        if self.concat:
            out = out.reshape(n, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out


class Model(nn.Module):
    """
    STGAT (framework-compatible):
    input:  [B, T, F]  (B=stocks as graph nodes)
    output: [B]
    """

    def __init__(self, args):
        super().__init__()
        self.input_size = args.input_size if getattr(args, "input_size", None) is not None else 158
        self.hidden_size = args.hidden_size if getattr(args, "hidden_size", None) is not None else 128
        self.num_layers = args.num_layers if getattr(args, "num_layers", None) is not None else 1
        self.dropout_rate = args.dropout if getattr(args, "dropout", None) is not None else 0.1

        self.gat_hidden = getattr(args, "gat_hidden_size", 32)
        self.gat_heads = getattr(args, "gat_heads", 4)
        self.gat_layers = getattr(args, "gat_layers", 2)
        self.gat_topk = getattr(args, "gat_topk", 20)
        self.kernel_size = max(3, int(getattr(args, "stgat_kernel_size", 3)))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1

        # Temporal branch (TCN-like gated conv over time)
        self.t_conv_p = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.t_conv_q = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        self.t_proj = nn.Linear(self.hidden_size, self.gat_hidden)

        # Sequence branch (GRU captures longer temporal dependency)
        rnn_dropout = self.dropout_rate if self.num_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )
        self.s_proj = nn.Linear(self.hidden_size, self.gat_hidden)

        # Spatial graph branch
        self.gat_blocks = nn.ModuleList(
            [
                GraphAttentionLayer(
                    in_dim=self.gat_hidden if i == 0 else self.gat_hidden * self.gat_heads,
                    out_dim=self.gat_hidden,
                    num_heads=self.gat_heads,
                    dropout=self.dropout_rate,
                    concat=True,
                )
                for i in range(self.gat_layers)
            ]
        )

        # Fusion + prediction head
        fusion_dim = self.gat_hidden * self.gat_heads + self.gat_hidden
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, 1),
        )

    def _build_graph(self, node_state):
        # node_state: [N, D]
        n = node_state.size(0)
        if n <= 1:
            return torch.eye(n, device=node_state.device, dtype=torch.bool)

        k = max(1, min(self.gat_topk, n - 1))
        with torch.no_grad():
            z = F.normalize(node_state, p=2, dim=-1)
            sim = z @ z.t()
            sim.fill_diagonal_(-1e9)
            knn_idx = torch.topk(sim, k=k, dim=1).indices
            adj = torch.zeros((n, n), device=node_state.device, dtype=torch.bool)
            adj.scatter_(1, knn_idx, True)
            adj.fill_diagonal_(True)
        return adj

    def forward(self, x):
        # x: [B, T, F]
        # ----- temporal gated conv branch -----
        xt = x.permute(0, 2, 1)  # [B, F, T]
        p = self.t_conv_p(xt)
        q = torch.sigmoid(self.t_conv_q(xt))
        t_feat = (p * q).permute(0, 2, 1)  # [B, T, H]
        t_last = t_feat[:, -1, :]  # [B, H]
        t_node = F.elu(self.t_proj(t_last))  # [B, gat_hidden]

        # ----- sequence branch -----
        rnn_out, _ = self.rnn(x)
        s_last = rnn_out[:, -1, :]  # [B, H]
        s_node = F.elu(self.s_proj(s_last))  # [B, gat_hidden]

        # ----- spatial graph attention -----
        adj = self._build_graph(s_node)
        h = s_node
        for gat in self.gat_blocks:
            h_new = gat(h, adj)
            h = F.elu(h_new + h if h_new.shape == h.shape else h_new)

        # ----- fusion -----
        fusion = torch.cat([h, t_node], dim=-1)
        score = self.fuse(fusion).squeeze(-1)  # [B]
        return score