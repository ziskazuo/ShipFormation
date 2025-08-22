# model_traditional_v2.py
from __future__ import annotations
import numpy as np
from typing import Tuple

# ---------- 工具：距离签名（与旧版兼容，做尺度归一化） ----------
def k_neigh_signature(P: np.ndarray, idx: int, k: int) -> np.ndarray:
    N = P.shape[0]
    if N <= 1:
        return np.zeros((k,), dtype=np.float32)
    d = np.sqrt(((P[idx][None, :] - P) ** 2).sum(axis=1) + 1e-12)
    d[idx] = np.inf
    k_eff = int(min(k, N - 1))
    d_small = np.partition(d, k_eff - 1)[:k_eff]
    d_small.sort()
    denom = max(d_small[-1], 1e-6) if k_eff > 0 else 1.0
    sig_small = (d_small / denom).astype(np.float32)
    if k_eff < k:
        if k_eff == 0:
            return np.zeros((k,), dtype=np.float32)
        pad = np.full((k - k_eff,), sig_small[-1], dtype=np.float32)
        return np.concatenate([sig_small, pad], axis=0)
    return sig_small

# ---------- Shape Context（旋转/尺度稳健的结构直方图） ----------
def _pairwise(P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """返回每对点的向量差 dx,dy（形状 [N,N]），用于快速算角度和半径。"""
    dx = P[:, 0][:, None] - P[:, 0][None, :]
    dy = P[:, 1][:, None] - P[:, 1][None, :]
    return dx, dy

def shape_context(P: np.ndarray,
                  nbins_r: int = 5,
                  nbins_theta: int = 12,
                  r_inner: float = 0.125,
                  r_outer: float = 2.0) -> np.ndarray:
    """
    计算每个点的 Shape Context 描述子（直方图），返回 [N, nbins_r, nbins_theta]。
    - 半径分箱为对数刻度（尺度稳健）：半径以点到其它点距离的中位数/均值为基准
    - 角度范围 (-pi, pi] 均分为 nbins_theta
    参考：Belongie et al., "Shape Matching and Object Recognition Using Shape Contexts"
    """
    N = P.shape[0]
    if N == 0:
        return np.zeros((0, nbins_r, nbins_theta), dtype=np.float32)
    dx, dy = _pairwise(P)
    r = np.sqrt(dx * dx + dy * dy) + 1e-12  # [N,N]
    theta = np.arctan2(dy, dx)  # (-pi, pi]

    # 尺度基准：每个点的半径用全局“中位互距”归一化更稳
    # 避免极端值影响
    r_valid = r[np.isfinite(r) & (r > 1e-9)]
    med = np.median(r_valid) if r_valid.size > 0 else 1.0
    r_norm = r / max(med, 1e-6)

    # 对数半径边界
    r_bin_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r + 1)
    # 角度边界
    theta_edges = np.linspace(-np.pi, np.pi, nbins_theta + 1)

    H = np.zeros((N, nbins_r, nbins_theta), dtype=np.float32)
    for i in range(N):
        # 去掉自身
        mask = np.ones(N, dtype=bool); mask[i] = False
        ri = r_norm[i, mask]
        ti = theta[i, mask]
        # 半径/角度量化
        r_idx = np.clip(np.searchsorted(r_bin_edges, ri, side='right') - 1, 0, nbins_r - 1)
        t_idx = np.clip(np.searchsorted(theta_edges, ti, side='right') - 1, 0, nbins_theta - 1)
        # 累积计数
        for rb, tb in zip(r_idx, t_idx):
            H[i, rb, tb] += 1.0

        # 归一化（使不同点/不同样本的计数尺度一致）
        s = H[i].sum()
        if s > 0:
            H[i] /= s

    return H  # [N, R, T]

def chi2_cost(h1: np.ndarray, h2: np.ndarray, eps: float = 1e-8) -> float:
    """Chi-square 直方图距离。h1/h2 形状相同。"""
    num = (h1 - h2) ** 2
    den = h1 + h2 + eps
    return float(0.5 * np.sum(num / den))

def sc_cost_row(desc_q: np.ndarray, desc_Y: np.ndarray,
                rotation_invariant: bool = True) -> np.ndarray:
    """
    计算 q 的 SC 与 Y 每个点的 SC 的匹配代价（越小越相似）。
    - desc_q: [R,T]
    - desc_Y: [M,R,T]
    若 rotation_invariant=True，则在角度维做循环移位，取最小值（相当于最佳旋转对齐）。
    返回：costs[M]
    """
    R, T = desc_q.shape
    M = desc_Y.shape[0]
    costs = np.zeros((M,), dtype=np.float32)
    if not rotation_invariant:
        for j in range(M):
            costs[j] = chi2_cost(desc_q, desc_Y[j])
        return costs

    # 旋转不变：在角度维度做循环移位取最小 Chi2
    for j in range(M):
        hj = desc_Y[j]
        best = np.inf
        for s in range(T):
            hj_shift = np.roll(hj, shift=s, axis=1)  # 沿角度维旋转
            c = chi2_cost(desc_q, hj_shift)
            if c < best:
                best = c
        costs[j] = best
    return costs

# ---------- 融合预测：SC + 距离签名 ----------
def predict_query_match(P1: np.ndarray, q_idx: int, P2: np.ndarray,
                        k_sig: int = 7,
                        nbins_r: int = 5, nbins_theta: int = 12,
                        w_sc: float = 1.0, w_sig: float = 0.30,
                        rotation_invariant: bool = True) -> tuple[int, np.ndarray, dict]:
    """
    只预测一个查询点的对应：
      1) 计算两侧的 Shape Context；
      2) 计算 q 的 SC 与 Y 所有点的 SC 代价（可旋转对齐）；
      3) 计算 q 的距离签名 与 Y 的距离签名 代价（L2）；
      4) 融合：C = w_sc * C_sc + w_sig * C_sig；
      5) 取 C 最小的列为预测；把 softmax(-C) 作为“置信度分布”返回。
    返回：(pred_j, probs[M], debug_info)
    """
    N1 = P1.shape[0]; N2 = P2.shape[0]
    if N1 == 0 or N2 == 0 or q_idx < 0 or q_idx >= N1:
        return -1, np.zeros((N2,), dtype=np.float32), {}

    # 1) Shape Context
    SC1 = shape_context(P1, nbins_r=nbins_r, nbins_theta=nbins_theta)   # [N1,R,T]
    SC2 = shape_context(P2, nbins_r=nbins_r, nbins_theta=nbins_theta)   # [N2,R,T]
    desc_q = SC1[q_idx]                                                 # [R,T]
    C_sc = sc_cost_row(desc_q, SC2, rotation_invariant=rotation_invariant)  # [M]

    # 2) 距离签名代价（用较大的 k，增强可分性）
    k_eff = int(min(k_sig, N1 - 1, N2 - 1))
    if k_eff <= 0:
        # 无法构造签名时，只用 SC
        C = C_sc
    else:
        sig_q = k_neigh_signature(P1, q_idx, k_eff)
        C_sig = np.zeros((N2,), dtype=np.float32)
        for j in range(N2):
            sig_j = k_neigh_signature(P2, j, k_eff)
            C_sig[j] = np.linalg.norm(sig_q - sig_j, ord=2)
        # 归一化两种代价的量纲（鲁棒 min-max）
        def _norm(v):
            v = v - v.min()
            m = v.max()
            return v / (m + 1e-6)
        C = w_sc * _norm(C_sc) + w_sig * _norm(C_sig)

    # 3) softmax(-C) -> 概率
    scores = -C
    exp = np.exp(scores - scores.max())
    probs = exp / max(exp.sum(), 1e-12)
    pred_j = int(probs.argmax()) if N2 > 0 else -1

    debug = {
        "C_sc": C_sc,  # 原始 SC 代价
        "C": C,        # 融合后的总代价
    }
    return pred_j, probs.astype(np.float32), debug
