# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class GraphAttentionLayer(nn.Module):
#     """Dense GAT layer with adjacency mask."""

#     def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
#         super().__init__()
#         self.in_dim = in_dim
#         self.out_dim = out_dim
#         self.num_heads = num_heads
#         self.concat = concat
#         self.dropout = nn.Dropout(dropout)

#         self.proj = nn.Linear(in_dim, out_dim * num_heads, bias=False)
#         self.att_src = nn.Parameter(torch.Tensor(num_heads, out_dim))
#         self.att_dst = nn.Parameter(torch.Tensor(num_heads, out_dim))
#         self.leaky_relu = nn.LeakyReLU(0.2)
#         self.reset_parameters()

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.proj.weight)
#         nn.init.xavier_uniform_(self.att_src)
#         nn.init.xavier_uniform_(self.att_dst)

#     def forward(self, x, adj_mask):
#         # x: [N, in_dim], adj_mask: [N, N] bool
#         num_nodes = x.size(0)
#         h = self.proj(x).view(num_nodes, self.num_heads, self.out_dim)  # [N, H, D]

#         # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])，拆成源/目标两部分可向量化
#         src_score = (h * self.att_src.unsqueeze(0)).sum(dim=-1)  # [N, H]
#         dst_score = (h * self.att_dst.unsqueeze(0)).sum(dim=-1)  # [N, H]
#         e = self.leaky_relu(src_score.unsqueeze(1) + dst_score.unsqueeze(0))  # [N, N, H]

#         # 非邻接边屏蔽，softmax 仅在邻居上归一化
#         neg_inf = torch.finfo(e.dtype).min
#         e = e.masked_fill(~adj_mask.unsqueeze(-1), neg_inf)
#         alpha = F.softmax(e, dim=1)
#         alpha = self.dropout(alpha)

#         out = torch.einsum("ijh,jhd->ihd", alpha, h)  # [N, H, D]
#         if self.concat:
#             out = out.reshape(num_nodes, self.num_heads * self.out_dim)
#         else:
#             out = out.mean(dim=1)
#         return out


# class Model(nn.Module):
#     """
#     改进版 GAT：
#     1) 先用 GRU 编码每只股票的时序特征
#     2) 再在当日股票截面上构图并执行图注意力消息传递
#     """

#     def __init__(self, args):
#         super().__init__()
#         self.input_size = args.input_size if getattr(args, "input_size", None) is not None else 158
#         self.temporal_hidden = args.hidden_size if getattr(args, "hidden_size", None) is not None else 128
#         self.temporal_layers = args.num_layers if getattr(args, "num_layers", None) is not None else 1
#         self.dropout_rate = args.dropout if getattr(args, "dropout", None) is not None else 0.1

#         self.gat_hidden = getattr(args, "gat_hidden_size", 32)
#         self.gat_heads = getattr(args, "gat_heads", 4)
#         self.gat_layers = getattr(args, "gat_layers", 2)
#         self.gat_topk = getattr(args, "gat_topk", 20)

#         rnn_dropout = self.dropout_rate if self.temporal_layers > 1 else 0.0
#         self.rnn = nn.GRU(
#             input_size=self.input_size,
#             hidden_size=self.temporal_hidden,
#             num_layers=self.temporal_layers,
#             batch_first=True,
#             dropout=rnn_dropout,
#         )

#         self.pre_gat = nn.Linear(self.temporal_hidden, self.gat_hidden)
#         self.gat_blocks = nn.ModuleList(
#             [
#                 GraphAttentionLayer(
#                     in_dim=self.gat_hidden if i == 0 else self.gat_hidden * self.gat_heads,
#                     out_dim=self.gat_hidden,
#                     num_heads=self.gat_heads,
#                     dropout=self.dropout_rate,
#                     concat=True,
#                 )
#                 for i in range(self.gat_layers)
#             ]
#         )

#         self.post_gat = nn.Linear(self.gat_hidden * self.gat_heads, self.gat_hidden)
#         self.head = nn.Linear(self.gat_hidden, 1)
#         self.dropout = nn.Dropout(self.dropout_rate)

#     def _build_graph(self, node_state):
#         """
#         用节点 embedding 的余弦相似度动态构图，保留每个节点 top-k 邻居。
#         node_state: [N, D]
#         """
#         num_nodes = node_state.size(0)
#         if num_nodes <= 1:
#             return torch.eye(num_nodes, device=node_state.device, dtype=torch.bool)

#         k = max(1, min(self.gat_topk, num_nodes - 1))
#         with torch.no_grad():
#             norm_state = F.normalize(node_state, p=2, dim=-1)
#             sim = norm_state @ norm_state.t()  # [N, N]
#             sim.fill_diagonal_(-1e9)
#             knn_idx = torch.topk(sim, k=k, dim=1).indices  # [N, k]

#             adj = torch.zeros((num_nodes, num_nodes), device=node_state.device, dtype=torch.bool)
#             adj.scatter_(1, knn_idx, True)
#             adj.fill_diagonal_(True)
#         return adj

#     def forward(self, x):
#         # x: [B, T, F]；B 在这个任务里是截面股票数（图节点）
#         out, _ = self.rnn(x)
#         node_state = out[:, -1, :]  # [B, temporal_hidden]
#         node_state = F.elu(self.pre_gat(node_state))

#         adj = self._build_graph(node_state)
#         h = node_state
#         for gat in self.gat_blocks:
#             h_new = gat(h, adj)
#             # 统一使用残差，增强深层稳定性
#             h = h_new + h if h_new.shape == h.shape else h_new
#             h = F.elu(h)

#         h = F.elu(self.post_gat(self.dropout(h)))
#         score = self.head(self.dropout(h)).squeeze(-1)
#         return score




import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """Dense GAT layer with adjacency mask."""

    def __init__(self, in_dim, out_dim, num_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.in_dim = in_dim
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
        # x: [N, in_dim], adj_mask: [N, N] bool
        num_nodes = x.size(0)
        h = self.proj(x).view(num_nodes, self.num_heads, self.out_dim)  # [N, H, D]

        # e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        src_score = (h * self.att_src.unsqueeze(0)).sum(dim=-1)  # [N, H]
        dst_score = (h * self.att_dst.unsqueeze(0)).sum(dim=-1)  # [N, H]
        e = self.leaky_relu(src_score.unsqueeze(1) + dst_score.unsqueeze(0))  # [N, N, H]

        # 屏蔽非邻接边
        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(~adj_mask.unsqueeze(-1), neg_inf)
        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)

        out = torch.einsum("ijh,jhd->ihd", alpha, h)  # [N, H, D]
        if self.concat:
            out = out.reshape(num_nodes, self.num_heads * self.out_dim)
        else:
            out = out.mean(dim=1)
        return out


class Model(nn.Module):
    """
    改进版 GRU + Dynamic Symmetric KNN-GAT 用于股票截面预测
    关键改进：
    1. GAT 维度处理更稳定（第一层 concat，后续 mean）
    2. _build_graph 改为无向图 + 可选相似度阈值过滤
    3. 残差连接更可靠
    """

    def __init__(self, args):
        super().__init__()
        self.input_size = getattr(args, "input_size", 158)
        self.temporal_hidden = getattr(args, "hidden_size", 128)
        self.temporal_layers = getattr(args, "num_layers", 1)
        self.dropout_rate = getattr(args, "dropout", 0.1)

        self.gat_hidden = getattr(args, "gat_hidden_size", 32)
        self.gat_heads = getattr(args, "gat_heads", 4)
        self.gat_layers = getattr(args, "gat_layers", 2)
        self.gat_topk = getattr(args, "gat_topk", 20)
        self.sim_threshold = getattr(args, "sim_threshold", 0.0)   # 新增：相似度阈值

        # GRU 时序编码
        rnn_dropout = self.dropout_rate if self.temporal_layers > 1 else 0.0
        self.rnn = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.temporal_hidden,
            num_layers=self.temporal_layers,
            batch_first=True,
            dropout=rnn_dropout,
        )

        self.pre_gat = nn.Linear(self.temporal_hidden, self.gat_hidden)

        # GAT blocks：只有第一层 concat=True
        self.gat_blocks = nn.ModuleList()
        current_in_dim = self.gat_hidden
        for i in range(self.gat_layers):
            concat = (i == 0)  # 第一层多头拼接，后续平均
            self.gat_blocks.append(
                GraphAttentionLayer(
                    in_dim=current_in_dim,
                    out_dim=self.gat_hidden,
                    num_heads=self.gat_heads,
                    dropout=self.dropout_rate,
                    concat=concat,
                )
            )
            current_in_dim = self.gat_hidden * self.gat_heads if concat else self.gat_hidden

        # post_gat 自动适配最终维度
        self.post_gat = nn.Linear(current_in_dim, self.gat_hidden)

        # 可选 LayerNorm（推荐开启，提升稳定性）
        # self.norm_layers = nn.ModuleList([nn.LayerNorm(self.gat_hidden) for _ in range(self.gat_layers)])

        self.head = nn.Linear(self.gat_hidden, 1)
        self.dropout = nn.Dropout(self.dropout_rate)

    def _build_graph(self, node_state):
        """
        改进版动态构图：
        - 使用余弦相似度 + L2 归一化
        - Top-k 候选邻居
        - 强制对称（无向图）→ 更符合股票相关性
        - 可选相似度阈值过滤噪声
        - 保留自环
        """
        num_nodes = node_state.size(0)
        if num_nodes <= 1:
            return torch.eye(num_nodes, device=node_state.device, dtype=torch.bool)

        k = max(1, min(self.gat_topk, num_nodes - 1))

        with torch.no_grad():
            norm_state = F.normalize(node_state, p=2, dim=-1)          # [N, D]
            sim = norm_state @ norm_state.t()                           # [N, N] 余弦相似度

            # 去掉自相似
            sim.fill_diagonal_(-1e9)

            # Top-k（有向）
            knn_idx = torch.topk(sim, k=k, dim=1).indices               # [N, k]

            # 构造初始邻接矩阵
            adj = torch.zeros((num_nodes, num_nodes),
                              device=node_state.device,
                              dtype=torch.bool)
            adj.scatter_(1, knn_idx, True)

            # 关键改进1：转为无向图（对称）
            adj = adj | adj.t()

            # 关键改进2：相似度阈值过滤（可选）
            if self.sim_threshold > -1.0:
                adj = adj & (sim > self.sim_threshold)

            # 加上自环
            adj.fill_diagonal_(True)

        return adj

    def forward(self, x):
        # x: [B, T, F]，B = 当天股票数（节点数）
        out, _ = self.rnn(x)
        node_state = out[:, -1, :]                      # [B, temporal_hidden]
        node_state = F.elu(self.pre_gat(node_state))

        adj = self._build_graph(node_state)

        h = node_state
        for i, gat in enumerate(self.gat_blocks):
            h_new = gat(h, adj)

            # 残差连接（第一层形状会变化，后续保持一致）
            if h_new.shape == h.shape:
                h = h_new + h
            else:
                h = h_new

            h = F.elu(h)

            # 可选：LayerNorm
            # h = self.norm_layers[i](h)

        h = F.elu(self.post_gat(self.dropout(h)))
        score = self.head(self.dropout(h)).squeeze(-1)   # [B]
        return score