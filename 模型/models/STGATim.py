import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== STL =====================
class SimpleSTL(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def moving_avg(self, x):
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.padding, self.padding), mode='replicate')
        x = F.avg_pool1d(x, kernel_size=self.kernel_size, stride=1)
        return x.permute(0, 2, 1)

    def forward(self, x):
        trend = self.moving_avg(x)

        detrended = x - trend
        seasonal = self.moving_avg(detrended)

        residual = x - trend - seasonal

        return trend, seasonal, residual

# ===================== GAT =====================
class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.att_dst = nn.Parameter(torch.Tensor(num_heads, out_dim))
        self.leaky_relu = nn.LeakyReLU(0.2)

        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x, adj_weight, adj_mask):
        # x: [N, D], adj_weight: [N, N], adj_mask: [N, N] bool
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        n = x.size(0)
        h = self.proj(x).view(n, self.num_heads, self.out_dim)

        src = (h * self.att_src.unsqueeze(0)).sum(-1)
        dst = (h * self.att_dst.unsqueeze(0)).sum(-1)
        e = self.leaky_relu(src.unsqueeze(1) + dst.unsqueeze(0))

        # 仅对图边加入权重，并在softmax前做邻接掩码
        edge_bias = torch.where(adj_mask, adj_weight, torch.zeros_like(adj_weight))
        e = e + edge_bias.unsqueeze(-1)

        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(~adj_mask.unsqueeze(-1), neg_inf)
        e = torch.nan_to_num(e, nan=0.0, posinf=1e4, neginf=-1e4)

        alpha = F.softmax(e, dim=1)
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1e-12)
        alpha = self.dropout(alpha)

        out = torch.einsum("ijh,jhd->ihd", alpha, h)
        out = out.reshape(n, self.num_heads * self.out_dim)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
        return out


# ===================== TCN =====================
class TemporalConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=4, dilation=4)

    def forward(self, x):
        # x: [B, T, F]
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        x = x.permute(0, 2, 1)
        out = self.conv1(x) + self.conv2(x)
        out = F.relu(out)
        return out.permute(0, 2, 1)


# ===================== Model =====================
class Model(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.input_size = getattr(args, "input_size", 158)
        self.hidden_size = getattr(args, "hidden_size", 128)
        self.gat_hidden = getattr(args, "gat_hidden_size", 32)
        self.gat_heads = getattr(args, "gat_heads", 4)
        self.gat_layers = getattr(args, "gat_layers", 2)
        self.topk = getattr(args, "gat_topk", 20)

        # STL
        self.stl = SimpleSTL()

        # Temporal
        self.tcn = TemporalConv(self.input_size, self.hidden_size)
        self.t_proj = nn.Linear(self.hidden_size, self.gat_hidden)

        # Residual branch
        self.res_mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.gat_hidden)
        )

        # GAT
        self.gat_layers_list = nn.ModuleList([
            GraphAttentionLayer(
                in_dim=self.gat_hidden if i == 0 else self.gat_hidden * self.gat_heads,
                out_dim=self.gat_hidden,
                num_heads=self.gat_heads
            )
            for i in range(self.gat_layers)
        ])

        # Fusion
        fusion_dim = self.gat_hidden * self.gat_heads + self.gat_hidden + self.gat_hidden
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

    # ===== 图构建（带权重）=====
    def build_graph(self, x):
        # x: [N, D]
        n = x.size(0)
        if n <= 1:
            adj_mask = torch.eye(n, device=x.device, dtype=torch.bool)
            adj_weight = torch.zeros((n, n), device=x.device, dtype=x.dtype)
            return adj_mask, adj_weight

        with torch.no_grad():
            z = F.normalize(x, p=2, dim=-1, eps=1e-12)
            z = torch.nan_to_num(z, nan=0.0, posinf=1e4, neginf=-1e4)
            sim = z @ z.t()  # cosine
            sim = torch.nan_to_num(sim, nan=0.0, posinf=1.0, neginf=-1.0)
            sim.fill_diagonal_(-1e9)

            # TopK mask
            k = max(1, min(self.topk, n - 1))
            idx = torch.topk(sim, k=k, dim=1).indices

            adj_mask = torch.zeros((n, n), device=x.device, dtype=torch.bool)
            adj_mask.scatter_(1, idx, True)
            adj_mask = adj_mask | adj_mask.t()
            adj_mask.fill_diagonal_(True)

        # 权重
        adj_weight = torch.where(adj_mask, sim, torch.zeros_like(sim))
        adj_weight = torch.nan_to_num(adj_weight, nan=0.0, posinf=1.0, neginf=-1.0)

        return adj_mask, adj_weight

    # ===== forward =====
    def forward(self, x):
        # x: [B, T, F]

        # ===== STL =====
        trend, seasonal, residual = self.stl(x)

        # ===== Temporal =====
        ts = trend + seasonal
        t_feat = self.tcn(ts)
        t_feat = torch.nan_to_num(t_feat, nan=0.0, posinf=1e4, neginf=-1e4)
        t_node = self.t_proj(t_feat[:, -1, :])
        t_node = torch.nan_to_num(t_node, nan=0.0, posinf=1e4, neginf=-1e4)

        # ===== Residual =====
        res_node = self.res_mlp(residual[:, -1, :])
        res_node = torch.nan_to_num(res_node, nan=0.0, posinf=1e4, neginf=-1e4)

        # ===== Graph =====
        node_state = t_node
        adj_mask, adj_weight = self.build_graph(node_state)

        h = node_state
        for gat in self.gat_layers_list:
            h_new = gat(h, adj_weight, adj_mask)
            h = F.elu(h_new + h if h_new.shape == h.shape else h_new)
            h = torch.nan_to_num(h, nan=0.0, posinf=1e4, neginf=-1e4)

        # ===== Fusion =====
        fusion = torch.cat([h, t_node, res_node], dim=-1)
        fusion = torch.nan_to_num(fusion, nan=0.0, posinf=1e4, neginf=-1e4)
        out = self.fuse(fusion).squeeze(-1)
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)

        return out