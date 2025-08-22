"""
model.py
--------
高精度版本：SuperGlue 风格的「自注意 + 交叉注意 + Sinkhorn」。
- 初始特征：k 近邻距离签名（SE(2)+尺度稳健）
- 编码器：多层 Transformer（自注意 + 交叉注意），带 RBF 距离几何偏置
- 打分：缩放点积 + |差| 的 MLP 融合
- dustbin：可学习（matchability 头），优雅处理删点/不等基数/杂点
保持与旧版一致的接口：MatchingModel(X,Y)->P[(N+1)x(M+1)]
"""
from __future__ import annotations
from typing import Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ot_utils import sinkhorn  # 仍复用你的 Sinkhorn

# ------------------------- 节点距离签名（稳健初始化） -------------------------
def node_distance_signature(coords: torch.Tensor, k: int = 5) -> torch.Tensor:
    """
    距离签名：对每个点取到其它点的最小 k 个距离（升序）。
    若 N-1<k，自动收缩并用“最后一个值重复”右填到长度 k，保证 [N,k] 形状稳定。
    """
    device = coords.device
    dtype = coords.dtype
    N = coords.shape[0]
    if N == 0:
        return torch.zeros((0, k), dtype=dtype, device=device)
    if N == 1:
        return torch.zeros((1, k), dtype=dtype, device=device)

    diff = coords[:, None, :] - coords[None, :, :]           # [N,N,2]
    dist = torch.sqrt(torch.sum(diff * diff, dim=-1) + 1e-12)  # [N,N]
    dist = dist + torch.diag_embed(torch.full((N,), float('inf'), device=device, dtype=dtype))

    k_eff = int(min(k, N - 1))
    smallest = torch.topk(dist, k_eff, dim=1, largest=False, sorted=True).values  # [N,k_eff]
    sig_small = torch.sort(smallest, dim=1).values

    if k_eff < k:
        if k_eff > 0:
            last = sig_small[:, -1:].expand(N, k - k_eff)
            sig = torch.cat([sig_small, last], dim=1)
        else:
            sig = torch.zeros((N, k), dtype=dtype, device=device)
    else:
        sig = sig_small
    return sig

# ------------------------- 几何偏置：RBF 距离核 -------------------------
class RBFBias(nn.Module):
    """
    将 pairwise 距离 d_ij 映射为注意力的加性 bias。
    bias_ij = <w_head, phi(d_ij)>，其中 phi 是 RBF 展开（高斯核）。
    """
    def __init__(self, n_kernels: int = 16, n_heads: int = 4):
        super().__init__()
        self.n_kernels = n_kernels
        self.n_heads = n_heads
        # 核心参数：RBF 的中心与宽度（可学习）
        self.centers = nn.Parameter(torch.linspace(0.0, 2.0, n_kernels)[None, :])  # [1,K]
        self.log_sigmas = nn.Parameter(torch.zeros(1, n_kernels))                  # [1,K]
        # 每个 head 一组线性组合权重
        self.w = nn.Parameter(torch.zeros(n_heads, n_kernels))                     # [H,K]
        nn.init.normal_(self.w, std=0.2)

    def forward(self, D: torch.Tensor) -> torch.Tensor:
        """
        输入:
          D : [N,M] 的欧氏距离矩阵
        输出:
          bias: [H,N,M] 可直接加到注意力 logits 上（对每个 head 不同）
        """
        # RBF 展开
        sigma2 = torch.exp(self.log_sigmas) ** 2 + 1e-8  # [1,K]
        feat = torch.exp(- (D[..., None] - self.centers) ** 2 / (2.0 * sigma2))  # [N,M,K]
        # head-wise 线性组合
        # bias[h,i,j] = <w[h,:], feat[i,j,:]>
        bias = torch.einsum('hk,nmk->hnm', self.w, feat)  # [H,N,M]
        return bias

# ------------------------- Multi-Head Attention（带几何偏置） -------------------------
class MHAttnWithBias(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.dk = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        # x:[N,d] -> [H,N,dk]
        N, d = x.shape
        x = x.view(N, self.nhead, self.dk).permute(1, 0, 2)
        return x

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        # x:[H,N,dk] -> [N,d]
        H, N, dk = x.shape
        x = x.permute(1, 0, 2).contiguous().view(N, H * dk)
        return x

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Q:[Nq,d], K:[Nk,d], V:[Nk,d]; bias:[H,Nq,Nk]（可为 None）
        单样本、非 batch（N 一般 < 128）
        """
        Qh = self._split(self.q_proj(Q))  # [H,Nq,dk]
        Kh = self._split(self.k_proj(K))  # [H,Nk,dk]
        Vh = self._split(self.v_proj(V))  # [H,Nk,dk]

        # 注意力 logits: [H,Nq,Nk]
        logits = torch.einsum('hqd,hkd->hqk', Qh, Kh) / math.sqrt(self.dk)
        if bias is not None:
            logits = logits + bias  # 加性几何偏置

        attn = F.softmax(logits, dim=-1)
        attn = self.dropout(attn)

        out = torch.einsum('hqk,hkd->hqd', attn, Vh)  # [H,Nq,dk]
        out = self._merge(out)                        # [Nq,d]
        out = self.o_proj(out)                        # [Nq,d]
        return out

# ------------------------- Transformer Block -------------------------
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MHAttnWithBias(d_model, nhead, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-Attn
        h = self.attn(X, X, X, bias=bias)
        X = X + self.drop(h)
        X = self.ln1(X)
        # FFN
        h = self.ffn(X)
        X = X + self.drop(h)
        X = self.ln2(X)
        return X

class CrossBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn_xy = MHAttnWithBias(d_model, nhead, dropout)
        self.ln_xy = nn.LayerNorm(d_model)
        self.ffn_xy = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2_xy = nn.LayerNorm(d_model)

        self.attn_yx = MHAttnWithBias(d_model, nhead, dropout)
        self.ln_yx = nn.LayerNorm(d_model)
        self.ffn_yx = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.ln2_yx = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, HX: torch.Tensor, HY: torch.Tensor,
                bias_xy: Optional[torch.Tensor], bias_yx: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        # X attends to Y
        h_xy = self.attn_xy(HX, HY, HY, bias=bias_xy)
        HX = HX + self.drop(h_xy); HX = self.ln_xy(HX)
        h = self.ffn_xy(HX); HX = HX + self.drop(h); HX = self.ln2_xy(HX)
        # Y attends to X
        h_yx = self.attn_yx(HY, HX, HX, bias=bias_yx)
        HY = HY + self.drop(h_yx); HY = self.ln_yx(HY)
        h = self.ffn_yx(HY); HY = HY + self.drop(h); HY = self.ln2_yx(HY)
        return HX, HY

# ------------------------- 匹配头：点积 + |差| MLP -------------------------
class PairwiseHead(nn.Module):
    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(5.0))
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 点积权重
        self.beta  = nn.Parameter(torch.tensor(1.0))  # 差异 MLP 权重
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, HX: torch.Tensor, HY: torch.Tensor) -> torch.Tensor:
        # 点积相似度
        sim = (HX @ HY.t()) / (HX.shape[-1] ** 0.5)          # [N,M]
        # |差| 的 MLP（逐对计算）
        N, M, d = HX.shape[0], HY.shape[0], HX.shape[1]
        # 构造 |HX_i - HY_j|
        diff = torch.abs(HX[:, None, :] - HY[None, :, :])    # [N,M,d]
        mlp_score = self.mlp(diff).squeeze(-1)               # [N,M]
        logits = self.alpha * sim + self.beta * mlp_score
        return logits * self.scale

# ------------------------- 主模型 -------------------------
class MatchingModel(nn.Module):
    """
    - 输入：两组坐标 X[N,2], Y[M,2]
    - 编码：距离签名 -> 线性投影 -> (自注意 + 交叉注意)*L（带 RBF 几何偏置）
    - 打分：点积 + |差| MLP 融合
    - dustbin：可学习（matchability 头）
    - 输出：经 Sinkhorn 的 (N+1)x(M+1) 双随机矩阵
    """
    def __init__(
        self,
        k: int = 5,               # 距离签名长度
        d_model: int = 128,       # token 维度
        nhead: int = 4,
        num_layers: int = 4,      # 每层包含：Self-X、Self-Y、Cross
        dropout: float = 0.1,
        rbf_kernels: int = 16,    # 几何偏置 RBF 核数
        learnable_dustbin: bool = True,
        dustbin_bias: float = -2.0  # dustbin 的初始偏置（越小越难选）
    ):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.learnable_dustbin = learnable_dustbin

        # 节点初始特征：距离签名 -> 线性升维到 d_model
        self.enc_in = nn.Sequential(
            nn.Linear(k, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # 几何偏置（RBF），自注意使用 D_xx / D_yy；交叉注意使用 D_xy / D_yx
        self.bias_xx = RBFBias(n_kernels=rbf_kernels, n_heads=nhead)
        self.bias_yy = RBFBias(n_kernels=rbf_kernels, n_heads=nhead)
        self.bias_xy = RBFBias(n_kernels=rbf_kernels, n_heads=nhead)
        self.bias_yx = RBFBias(n_kernels=rbf_kernels, n_heads=nhead)

        # 编码堆叠
        self.self_blocks_x = nn.ModuleList([TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.self_blocks_y = nn.ModuleList([TransformerBlock(d_model, nhead, dropout) for _ in range(num_layers)])
        self.cross_blocks   = nn.ModuleList([CrossBlock(d_model, nhead, dropout) for _ in range(num_layers)])

        # 匹配打分头
        self.pair_head = PairwiseHead(d_model, hidden=128)

        # 可学习 dustbin（matchability 头）
        if self.learnable_dustbin:
            self.matchability_x = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
            )
            self.matchability_y = nn.Sequential(
                nn.Linear(d_model, d_model // 2), nn.GELU(), nn.Linear(d_model // 2, 1)
            )
            # 让初始 dustbin 比较难被选（负偏置）
            with torch.no_grad():
                for m in (self.matchability_x, self.matchability_y):
                    last = [m[-1]]
                    last[0].bias.fill_(dustbin_bias)

    # ---------- 几何工具 ----------
    @staticmethod
    def pairwise_dist(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        # A:[Na,2], B:[Nb,2] -> D:[Na,Nb]
        diff = A[:, None, :] - B[None, :, :]
        return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-12)

    def encode_nodes(self, coords: torch.Tensor) -> torch.Tensor:
        sig = node_distance_signature(coords, k=self.k)  # [N,k]
        H = self.enc_in(sig)                              # [N,d]
        return H

    # ---------- 组装 learnable dustbin ----------
    @staticmethod
    def _add_learnable_dustbin(logits: torch.Tensor,
                               dust_x: torch.Tensor,
                               dust_y: torch.Tensor,
                               corner: float = 0.0) -> torch.Tensor:
        """
        logits:[N,M], dust_x:[N,1]（X->dustbin 列），dust_y:[M,1]（Y->dustbin 行）
        返回 [N+1,M+1]
        """
        N, M = logits.shape
        out = logits.new_full((N + 1, M + 1), -1e9)
        out[:N, :M] = logits
        out[:N, M:M+1] = dust_x           # 最后一列
        out[N:N+1, :M] = dust_y.t()       # 最后一行
        out[N, M] = corner
        return out

    def forward(self, X: torch.Tensor, Y: torch.Tensor, sinkhorn_iters: int = 50, tau: float = 0.5) -> torch.Tensor:
        """
        X:[N,2], Y:[M,2] -> P:[N+1,M+1]
        """
        N, M = X.shape[0], Y.shape[0]

        # 1) 初始 token
        HX = self.encode_nodes(X)   # [N,d]
        HY = self.encode_nodes(Y)   # [M,d]

        # 2) 预计算几何距离（用于注意力偏置）
        D_xx = self.pairwise_dist(X, X)   # [N,N]
        D_yy = self.pairwise_dist(Y, Y)   # [M,M]
        D_xy = self.pairwise_dist(X, Y)   # [N,M]
        D_yx = D_xy.t().contiguous()      # [M,N]

        # 3) 多层（Self-X，Self-Y，Cross）
        for l in range(self.num_layers):
            # Self
            bias_xx = self.bias_xx(D_xx)  # [H,N,N]
            bias_yy = self.bias_yy(D_yy)  # [H,M,M]
            HX = self.self_blocks_x[l](HX, bias=bias_xx)
            HY = self.self_blocks_y[l](HY, bias=bias_yy)
            # Cross
            bias_xy = self.bias_xy(D_xy)  # [H,N,M]
            bias_yx = self.bias_yx(D_yx)  # [H,M,N]
            HX, HY = self.cross_blocks[l](HX, HY, bias_xy, bias_yx)

        # 4) 两两打分
        logits_nm = self.pair_head(HX, HY)  # [N,M]

        # 5) dustbin（learnable）
        if self.learnable_dustbin:
            dust_x = self.matchability_x(HX)  # [N,1]  -> 最后一列
            dust_y = self.matchability_y(HY)  # [M,1]  -> 最后一行
            logits_ext = self._add_learnable_dustbin(logits_nm, dust_x, dust_y, corner=0.0)  # [N+1,M+1]
        else:
            # 回退：常数 dustbin
            dust_val = -4.0
            logits_ext = logits_nm.new_full((N + 1, M + 1), dust_val)
            logits_ext[:N, :M] = logits_nm
            logits_ext[N, M] = 0.0  # corner

        # 6) Sinkhorn -> 近似双随机
        P = sinkhorn(logits_ext / tau, n_iters=sinkhorn_iters)
        return P
