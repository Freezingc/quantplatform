import torch
import torch.nn as nn
import torch.nn.functional as F


# ===================== VMD =====================
class VMD(nn.Module):
    """
    Variational Mode Decomposition (Dragomiretskiy & Zosso, 2014).
    Batched FFT-based ADMM solver in PyTorch.

    将每条 1D 信号分解为 K 个固有模态函数 (IMF)，
    每个 IMF 围绕自适应确定的中心频率。

    Args:
        num_modes: 分解模态数 K（默认 3: 低频趋势 / 中频周期 / 高频噪声）
        alpha: 带宽约束参数（越大频带越窄，默认 2000）
        tau: 拉格朗日乘子更新步长（0 = 严格重构约束）
        n_iter: ADMM 迭代次数
    """
    def __init__(self, num_modes=3, alpha=2000, tau=0.0, n_iter=30, signal_clip=5.0, mode_clip=5.0):
        super().__init__()
        self.K = num_modes
        self.alpha = alpha
        self.tau = tau
        self.n_iter = n_iter
        self.signal_clip = signal_clip
        self.mode_clip = mode_clip
        self.eps = 1e-4

    @torch.no_grad()
    def forward(self, x):
        # x: [B, T, F]
        B, T, F = x.shape
        # 将每只股票的每个特征视为独立 1D 信号
        signal = x.permute(0, 2, 1).reshape(B * F, T)  # [N, T]
        signal_mean = signal.mean(dim=-1, keepdim=True)
        signal_std = signal.std(dim=-1, unbiased=False, keepdim=True).clamp(min=self.eps)
        signal = ((signal - signal_mean) / signal_std).clamp(-self.signal_clip, self.signal_clip)
        N = signal.shape[0]

        # FFT
        f_hat = torch.fft.rfft(signal, dim=-1)  # [N, L], L = T//2+1
        L = f_hat.shape[-1]
        freqs = torch.arange(L, device=x.device, dtype=x.dtype) / T  # [L]

        # 初始化：各模态均分频谱
        u_hat = f_hat.unsqueeze(1).expand(-1, self.K, -1).clone() / self.K
        # 初始化中心频率：均匀分布在 [0, 0.5]
        omega = torch.zeros(N, self.K, device=x.device, dtype=x.dtype)
        for k in range(self.K):
            omega[:, k] = (0.5 / self.K) * (k + 0.5)
        # 对偶变量
        lambda_hat = torch.zeros_like(f_hat)  # [N, L]

        # ADMM 迭代
        freqs_expanded = freqs.unsqueeze(0)  # [1, L]
        for _ in range(self.n_iter):
            for k in range(self.K):
                residual = f_hat - (u_hat.sum(1) - u_hat[:, k]) + lambda_hat / 2
                bw = self.alpha * (freqs_expanded - omega[:, k:k+1]) ** 2
                u_hat[:, k] = residual / (1 + bw)

                power = u_hat[:, k].abs().square()
                denom = power.sum(-1).clamp(min=1e-10)
                omega[:, k] = (power * freqs).sum(-1) / denom

            # 更新对偶变量
            if self.tau > 0:
                u_hat_total = u_hat.sum(1)
                lambda_hat = lambda_hat + self.tau * (f_hat - u_hat_total)

        # IFFT 重建各模态的时域信号
        modes = []
        for k in range(self.K):
            mk = torch.fft.irfft(u_hat[:, k], n=T, dim=-1).clamp(-self.mode_clip, self.mode_clip)  # [N, T]
            modes.append(mk.reshape(B, F, T).permute(0, 2, 1))  # [B, T, F]

        return modes  # list of K tensors, each [B, T, F]


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

    def forward(self, x, adj_weight):
        # x: [N, D], adj_weight: [N, N]
        n = x.size(0)
        h = self.proj(x).view(n, self.num_heads, self.out_dim)

        src = (h * self.att_src.unsqueeze(0)).sum(-1)
        dst = (h * self.att_dst.unsqueeze(0)).sum(-1)
        e = self.leaky_relu(src.unsqueeze(1) + dst.unsqueeze(0))

        # 融合边权重
        e = e + adj_weight.unsqueeze(-1)

        alpha = F.softmax(e, dim=1)
        alpha = self.dropout(alpha)

        out = torch.einsum("ijh,jhd->ihd", alpha, h)
        return out.reshape(n, self.num_heads * self.out_dim)


# ===================== TCN =====================
class TemporalConv(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=2, dilation=2)
        self.conv2 = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding=4, dilation=4)

    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)
        out = self.conv1(x) + self.conv2(x)
        return F.relu(out).permute(0, 2, 1)


# ===================== Model =====================
class Model(nn.Module):
    """
    STGAT_VMD: Spatio-Temporal GAT with Variational Mode Decomposition.

    Pipeline:
      x → VMD → K modes (low / mid / high freq)
      low + mid → TCN → node features → TopK Graph → Multi-layer GAT
      high freq → MLP → residual features
      Fusion(GAT_out, temporal, residual) → score
    """
    def __init__(self, args):
        super().__init__()

        self.input_size = getattr(args, "input_size", 158)
        self.hidden_size = getattr(args, "hidden_size", 128)
        self.gat_hidden = getattr(args, "gat_hidden_size", 32)
        self.gat_heads = getattr(args, "gat_heads", 4)
        self.num_gat_layers = getattr(args, "gat_layers", 2)
        self.topk = getattr(args, "gat_topk", 20)
        self.dropout_rate = getattr(args, "dropout", 0.1)
        self.output_scale = getattr(args, "output_scale", 3.0)

        # VMD
        vmd_modes = getattr(args, "vmd_modes", 3)
        vmd_alpha = getattr(args, "vmd_alpha", 2000)
        vmd_iter = getattr(args, "vmd_iter", 30)
        vmd_signal_clip = getattr(args, "vmd_signal_clip", 5.0)
        vmd_mode_clip = getattr(args, "vmd_mode_clip", 5.0)
        self.vmd = VMD(
            num_modes=vmd_modes,
            alpha=vmd_alpha,
            n_iter=vmd_iter,
            signal_clip=vmd_signal_clip,
            mode_clip=vmd_mode_clip,
        )
        self.num_modes = vmd_modes

        # Temporal branch（处理低频 + 中频模态）
        self.tcn = TemporalConv(self.input_size, self.hidden_size)
        self.t_proj = nn.Linear(self.hidden_size, self.gat_hidden)
        self.temporal_norm = nn.LayerNorm(self.gat_hidden, elementwise_affine=False)

        # Residual branch（处理高频模态）
        self.res_mlp = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, self.gat_hidden)
        )
        self.residual_norm = nn.LayerNorm(self.gat_hidden, elementwise_affine=False)

        # Multi-layer GAT
        self.gat_layers_list = nn.ModuleList([
            GraphAttentionLayer(
                in_dim=self.gat_hidden if i == 0 else self.gat_hidden * self.gat_heads,
                out_dim=self.gat_hidden,
                num_heads=self.gat_heads,
                dropout=self.dropout_rate
            )
            for i in range(self.num_gat_layers)
        ])
        self.gat_norms = nn.ModuleList([
            nn.LayerNorm(self.gat_hidden * self.gat_heads, elementwise_affine=False)
            for _ in range(self.num_gat_layers)
        ])

        # Fusion + prediction head
        fusion_dim = self.gat_hidden * self.gat_heads + self.gat_hidden + self.gat_hidden
        self.fusion_norm = nn.LayerNorm(fusion_dim, elementwise_affine=False)
        self.fuse = nn.Sequential(
            nn.Linear(fusion_dim, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.hidden_size, 1)
        )

    def _normalize_sequence(self, x):
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, unbiased=False, keepdim=True).clamp(min=1e-4)
        return (x - mean) / std

    def _ensure_finite(self, tensor, name):
        if torch.isfinite(tensor).all():
            return tensor
        total = tensor.numel()
        finite = torch.isfinite(tensor).sum().item()
        nan_count = torch.isnan(tensor).sum().item()
        posinf_count = torch.isposinf(tensor).sum().item()
        neginf_count = torch.isneginf(tensor).sum().item()
        raise RuntimeError(
            f"STGAT_VMD produced non-finite values at {name}: "
            f"shape={tuple(tensor.shape)}, finite={finite}/{total}, "
            f"nan={nan_count}, posinf={posinf_count}, neginf={neginf_count}"
        )

    def build_graph(self, x):
        # x: [N, D]
        n = x.size(0)
        z = F.normalize(x, dim=-1, eps=1e-6)
        sim = z @ z.t()  # [N, N]

        # TopK 稀疏化
        k = min(self.topk, max(n - 1, 1))
        topk_idx = torch.topk(sim, k=k, dim=1).indices
        mask = torch.zeros_like(sim)
        mask.scatter_(1, topk_idx, 1.0)

        # 加权邻接矩阵
        adj_weight = sim * mask
        return adj_weight

    def forward(self, x):
        # x: [B, T, F]
        x = self._ensure_finite(x, 'input')

        # ===== VMD 分解 =====
        modes = self.vmd(x)  # list of K tensors [B, T, F]
        for idx, mode in enumerate(modes):
            self._ensure_finite(mode, f'vmd_mode_{idx}')

        # 低频 + 中频 → TCN
        if self.num_modes >= 2:
            ts = modes[0] + modes[1]
        else:
            ts = modes[0]

        ts = self._normalize_sequence(ts)
        ts = self._ensure_finite(ts, 'temporal_input')
        t_feat = self.tcn(ts)
        t_feat = self._ensure_finite(t_feat, 'temporal_conv')
        t_node = self.temporal_norm(self.t_proj(t_feat[:, -1, :]))  # [B, gat_hidden]
        t_node = self._ensure_finite(t_node, 'temporal_node')

        # 高频 → MLP
        high = modes[-1] if self.num_modes >= 3 else (x - modes[0])
        high = self._normalize_sequence(high)
        high = self._ensure_finite(high, 'residual_input')
        res_node = self.residual_norm(self.res_mlp(high[:, -1, :]))  # [B, gat_hidden]
        res_node = self._ensure_finite(res_node, 'residual_node')

        # ===== TopK 图 + 多层 GAT =====
        adj_weight = self.build_graph(t_node)
        adj_weight = self._ensure_finite(adj_weight, 'adj_weight')
        h = t_node
        for layer_idx, (gat, norm) in enumerate(zip(self.gat_layers_list, self.gat_norms)):
            h_new = gat(h, adj_weight)
            h_new = self._ensure_finite(h_new, f'gat_layer_{layer_idx}_output')
            h = h_new + h if h_new.shape == h.shape else h_new
            h = F.elu(norm(h))
            h = self._ensure_finite(h, f'gat_layer_{layer_idx}_post_norm')

        # ===== Fusion =====
        fusion = self.fusion_norm(torch.cat([h, t_node, res_node], dim=-1))
        fusion = self._ensure_finite(fusion, 'fusion')
        out = self.fuse(fusion).squeeze(-1)  # [B]
        out = self._ensure_finite(out, 'score')

        return out