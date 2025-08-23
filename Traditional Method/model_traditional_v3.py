# model_traditional_v3.py
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

# ----------------- 通用工具 -----------------
def _pairwise(P: np.ndarray) -> np.ndarray:
    """返回两两欧氏距离矩阵 [N,N]"""
    if P.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = P[:, None, :] - P[None, :, :]
    D = np.sqrt((diff ** 2).sum(axis=2) + 1e-12)
    return D

def _robust_scale(P: np.ndarray) -> float:
    """以点到质心的中位数半径作为稳健尺度"""
    if P.size == 0:
        return 1.0
    c = P.mean(axis=0)
    r = np.sqrt(((P - c) ** 2).sum(axis=1) + 1e-12)
    med = np.median(r) if r.size > 0 else 1.0
    return float(max(med, 1e-6))

def _minmax_norm(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    if v.size == 0:
        return v
    mn, mx = float(v.min()), float(v.max())
    if mx - mn < 1e-9:
        return np.zeros_like(v, dtype=np.float32)
    return (v - mn) / (mx - mn)

def _quantile_sig(dist_vec: np.ndarray, q: int = 8) -> np.ndarray:
    """把距离向量压成 q 个分位数签名，增强鲁棒性"""
    if dist_vec.size == 0:
        return np.zeros((q,), dtype=np.float32)
    qs = np.linspace(0.0, 1.0, q, endpoint=True)
    return np.quantile(dist_vec, qs).astype(np.float32)

# ----------------- 局部特征：定向 Shape Context -----------------
def shape_context_oriented(P: np.ndarray,
                           nbins_r: int = 5,
                           nbins_theta: int = 12,
                           r_inner: float = 0.125,
                           r_outer: float = 2.0) -> np.ndarray:
    """
    定向（不做角度循环）的 Shape Context: [N, nbins_r, nbins_theta]
    - 半径对数刻度 + 全局中位半径归一化（尺度稳健）
    - 角度直接量化（不旋转对齐，因你数据角度基本不变）
    """
    N = P.shape[0]
    if N == 0:
        return np.zeros((0, nbins_r, nbins_theta), dtype=np.float32)

    dx = P[:, 0][:, None] - P[:, 0][None, :]
    dy = P[:, 1][:, None] - P[:, 1][None, :]
    r = np.sqrt(dx * dx + dy * dy) + 1e-12
    theta = np.arctan2(dy, dx)  # (-pi, pi]

    # 全局稳健尺度
    r_valid = r[np.isfinite(r) & (r > 1e-9)]
    med = np.median(r_valid) if r_valid.size > 0 else 1.0
    r_norm = r / max(med, 1e-6)

    # 分箱
    r_edges = np.logspace(np.log10(r_inner), np.log10(r_outer), nbins_r + 1)
    t_edges = np.linspace(-np.pi, np.pi, nbins_theta + 1)

    H = np.zeros((N, nbins_r, nbins_theta), dtype=np.float32)
    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        ri = r_norm[i, mask]; ti = theta[i, mask]
        r_idx = np.clip(np.searchsorted(r_edges, ri, side='right') - 1, 0, nbins_r - 1)
        t_idx = np.clip(np.searchsorted(t_edges, ti, side='right') - 1, 0, nbins_theta - 1)
        for rb, tb in zip(r_idx, t_idx):
            H[i, rb, tb] += 1.0
        s = H[i].sum()
        if s > 0:
            H[i] /= s
    return H

def chi2_cost(h1: np.ndarray, h2: np.ndarray, eps: float = 1e-8) -> float:
    num = (h1 - h2) ** 2
    den = h1 + h2 + eps
    return float(0.5 * np.sum(num / den))

# ----------------- 局部特征：最近邻距离签名 -----------------
def k_nn_signature(P: np.ndarray, idx: int, k: int) -> np.ndarray:
    N = P.shape[0]
    if N <= 1:
        return np.zeros((k,), dtype=np.float32)
    d = np.sqrt(((P[idx][None, :] - P) ** 2).sum(axis=1) + 1e-12)
    d[idx] = np.inf
    k_eff = int(min(k, N - 1))
    d_small = np.partition(d, k_eff - 1)[:k_eff]
    d_small.sort()
    # 为了与“全局尺度”兼容，这里做“按该点 k邻里最大值”归一化
    denom = max(d_small[-1], 1e-6)
    sig_small = (d_small / denom).astype(np.float32)
    if k_eff < k:
        if k_eff == 0:
            return np.zeros((k,), dtype=np.float32)
        pad = np.full((k - k_eff,), sig_small[-1], dtype=np.float32)
        return np.concatenate([sig_small, pad], axis=0)
    return sig_small

# ----------------- 全局对齐与强约束 -----------------
def global_align_predict_q(P1: np.ndarray, q_idx: int, P2: np.ndarray,
                           rotation_grid_deg: Tuple[float, ...] = (0.0,),
                           use_scale: bool = True) -> Tuple[np.ndarray, float, float]:
    """
    用“中心 + 稳健尺度 + 小角度搜索(可选)”把 q 从 P1 映射到 P2，返回：
      (q_hat_in_P2, s, theta_best)
    - 因你数据角度基本不变，rotation_grid 默认只有 0 度；若想微调可传 (-5,0,5) 等。
    """
    c1 = P1.mean(axis=0) if P1.size else np.zeros(2)
    c2 = P2.mean(axis=0) if P2.size else np.zeros(2)
    s = (_robust_scale(P2) / _robust_scale(P1)) if use_scale else 1.0

    v = P1[q_idx] - c1  # 向量
    best_theta = 0.0
    best_qhat = c2 + s * v
    best_score = np.inf

    for deg in rotation_grid_deg:
        th = np.deg2rad(deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
        qhat = c2 + (R @ (s * v).reshape(2, 1)).reshape(2)
        # 用“到全体点的分位数距离签名”去衡量 qhat 对应局部与 P2 的匹配度：选最小者
        dists = np.sqrt(((P2 - qhat[None, :]) ** 2).sum(axis=1) + 1e-12)
        score = np.median(dists)  # 稳健：中位残差
        if score < best_score:
            best_score = score
            best_theta = float(th)
            best_qhat = qhat.astype(np.float32)

    return best_qhat, float(s), float(best_theta)

# ----------------- 主预测函数 -----------------
def predict_query_match(P1: np.ndarray, q_idx: int, P2: np.ndarray,
                        # 局部特征
                        nbins_r: int = 5, nbins_theta: int = 12,
                        k_sig: int = 8,
                        # 代价权重（全局&局部高权）
                        w_global_pos: float = 1.30,
                        w_sc: float = 1.20,
                        w_sig: float = 0.60,
                        w_centroid: float = 0.80,
                        w_quant: float = 0.80,
                        # 其它
                        rotation_grid_deg: Tuple[float, ...] = (0.0,),
                        tau: float = 0.35) -> Tuple[int, np.ndarray, Dict[str, np.ndarray]]:
    """
    结合“全局对齐 + 定向SC + 多一致性”的匹配器。
    返回：(pred_j, probs[M], debug)
    """
    N1, N2 = P1.shape[0], P2.shape[0]
    if N1 == 0 or N2 == 0 or q_idx < 0 or q_idx >= N1:
        return -1, np.zeros((N2,), dtype=np.float32), {}

    # 1) 全局对齐：把 q 从 P1 映射至 P2 坐标
    q_hat, s12, theta = global_align_predict_q(P1, q_idx, P2, rotation_grid_deg=rotation_grid_deg, use_scale=True)

    # 2) 构造各项代价
    # 2.1 全局位置残差（强约束）：||P2_j - q_hat||
    C_global = np.sqrt(((P2 - q_hat[None, :]) ** 2).sum(axis=1) + 1e-12)

    # 2.2 局部：定向 Shape Context 的卡方距离
    SC1 = shape_context_oriented(P1, nbins_r=nbins_r, nbins_theta=nbins_theta)  # [N1,R,T]
    SC2 = shape_context_oriented(P2, nbins_r=nbins_r, nbins_theta=nbins_theta)  # [N2,R,T]
    hq = SC1[q_idx]
    C_sc = np.zeros((N2,), dtype=np.float32)
    for j in range(N2):
        C_sc[j] = chi2_cost(hq, SC2[j])

    # 2.3 局部：最近邻距离签名的 L2 距离
    k_eff = int(min(k_sig, N1 - 1, N2 - 1))
    if k_eff <= 0:
        C_sig = np.zeros((N2,), dtype=np.float32)
    else:
        sig_q = k_nn_signature(P1, q_idx, k_eff)
        C_sig = np.zeros((N2,), dtype=np.float32)
        for j in range(N2):
            sig_j = k_nn_signature(P2, j, k_eff)
            C_sig[j] = np.linalg.norm(sig_q - sig_j, ord=2)

    # 2.4 全局：到质心距离一致性 | ||x_q-c1|| - ||y_j-c2|| |
    c1, c2 = P1.mean(axis=0), P2.mean(axis=0)
    rq = np.linalg.norm(P1[q_idx] - c1)
    r2 = np.linalg.norm(P2 - c2, axis=1)
    C_centroid = np.abs(rq * s12 - r2)

    # 2.5 全局+局部：与全体点的“分位数距离签名”一致性
    Dx = _pairwise(P1); Dy = _pairwise(P2)
    dx_q = Dx[q_idx]  # [N1]
    qsig_x = _quantile_sig(dx_q[dx_q > 0], q=10)
    C_quant = np.zeros((N2,), dtype=np.float32)
    for j in range(N2):
        dy_j = Dy[j]
        jsig_y = _quantile_sig(dy_j[dy_j > 0], q=10)
        # L2 代价（签名已受尺度影响；我们用全局 s12 去近似校正）
        # 先把 y 侧签名按 1/s12 缩回到 x 的尺度再比较
        jsig_y_scaled = jsig_y / max(s12, 1e-6)
        C_quant[j] = np.linalg.norm(qsig_x - jsig_y_scaled, ord=2)

    # 3) 代价融合（越小越好），全局与局部项赋予高权
    C = (
        w_global_pos * _minmax_norm(C_global) +
        w_sc         * _minmax_norm(C_sc) +
        w_sig        * _minmax_norm(C_sig) +
        w_centroid   * _minmax_norm(C_centroid) +
        w_quant      * _minmax_norm(C_quant)
    ).astype(np.float32)

    # 4) softmax(-C/τ) 得到“置信度”
    scores = -C / max(tau, 1e-6)
    exp = np.exp(scores - scores.max())
    probs = (exp / max(exp.sum(), 1e-12)).astype(np.float32)
    pred_j = int(probs.argmax()) if N2 > 0 else -1

    debug = {
        "C_global": C_global, "C_sc": C_sc, "C_sig": C_sig,
        "C_centroid": C_centroid, "C_quant": C_quant,
        "C_total": C, "q_hat": q_hat, "s12": np.array([s12], dtype=np.float32), "theta": np.array([theta], dtype=np.float32)
    }
    return pred_j, probs, debug
