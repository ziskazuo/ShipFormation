# model_traditional_v3.py  (v3.1 patch)
from __future__ import annotations
import numpy as np
from typing import Tuple, Dict

# ----------------- 基础工具 -----------------
def _pairwise(P: np.ndarray) -> np.ndarray:
    if P.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=2) + 1e-12)

def _robust_scale(P: np.ndarray) -> float:
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

def _quantile_sig(dist_vec: np.ndarray, q: int = 10) -> np.ndarray:
    if dist_vec.size == 0:
        return np.zeros((q,), dtype=np.float32)
    qs = np.linspace(0.0, 1.0, q, endpoint=True)
    return np.quantile(dist_vec, qs).astype(np.float32)

# ----------------- 邻域与Kabsch对齐 -----------------
def _k_nearest_indices(P: np.ndarray, idx: int, k: int) -> np.ndarray:
    N = P.shape[0]
    if N <= 1 or k <= 0:
        return np.zeros((0,), dtype=int)
    d = np.sqrt(((P - P[idx][None, :]) ** 2).sum(axis=1) + 1e-12)
    d[idx] = np.inf
    k_eff = int(min(k, N - 1))
    idxs = np.argpartition(d, k_eff)[:k_eff]
    # 稳定一点：按角度排序（相对于中心）
    v = P[idxs] - P[idx]
    ang = np.arctan2(v[:, 1], v[:, 0])
    order = np.argsort(ang)
    return idxs[order]

def _similarity_align_residual(X: np.ndarray, Y: np.ndarray) -> float:
    """
    给定成对的点集 X[m,2] -> Y[m,2]，估计相似变换（s,R,t），返回平均残差。
    m>=2 才有意义；m<2 返回一个较大的残差。
    """
    m = X.shape[0]
    if m < 2 or Y.shape[0] != m:
        return 1e6
    Xc = X - X.mean(axis=0, keepdims=True)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    # Kabsch 旋转
    H = Xc.T @ Yc
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    # 尺度（相似变换）
    denom = (Xc**2).sum()
    s = float(S.sum() / max(denom, 1e-9))
    # 重建并求残差
    X_hat = (s * (Xc @ R)) + Y.mean(axis=0, keepdims=True)
    res = np.sqrt(((X_hat - Y)**2).sum(axis=1) + 1e-12).mean()
    return float(res)

def _local_align_residual(P1: np.ndarray, q_idx: int, P2: np.ndarray, j_idx: int, k_local: int = 4) -> float:
    """
    取两侧各自的 k_local 邻域，按角度排序后一一配对，做一次相似对齐残差。
    """
    idx1 = _k_nearest_indices(P1, q_idx, k_local)
    idx2 = _k_nearest_indices(P2, j_idx, k_local)
    m = int(min(idx1.size, idx2.size))
    if m < 2:
        return 1e6
    # 组装有中心的配对：中心 + m个邻居
    X = np.vstack([P1[q_idx][None, :], P1[idx1[:m]]])
    Y = np.vstack([P2[j_idx][None, :], P2[idx2[:m]]])
    return _similarity_align_residual(X, Y)

# ----------------- 局部特征：定向 SC（软分箱可选） -----------------
def _soft_bin_1d(pos: float, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    给定连续位置 pos（0..nbins），返回相邻两个bin索引与权重（线性权重）。
    pos 可在 [0, nbins) 内；溢出时会被夹到边界。
    """
    if pos <= 0:
        return np.array([0]), np.array([1.0], dtype=np.float32)
    if pos >= nbins - 1:
        return np.array([nbins-1]), np.array([1.0], dtype=np.float32)
    i0 = int(np.floor(pos))
    i1 = i0 + 1
    w1 = pos - i0
    w0 = 1.0 - w1
    return np.array([i0, i1]), np.array([w0, w1], dtype=np.float32)

def shape_context_oriented(P: np.ndarray,
                           nbins_r: int = 5,
                           nbins_theta: int = 12,
                           r_inner: float = 0.125,
                           r_outer: float = 2.0,
                           soft: bool = True) -> np.ndarray:
    """
    定向（不循环角度）的 Shape Context: [N, R, T]
    - 半径对数刻度 + 全局中位半径归一化（尺度稳健）
    - 角度直接量化（不旋转对齐）
    - soft=True 时使用双线性软分箱，降低量化抖动
    """
    N = P.shape[0]
    H = np.zeros((N, nbins_r, nbins_theta), dtype=np.float32)
    if N == 0:
        return H

    dx = P[:, 0][:, None] - P[:, 0][None, :]
    dy = P[:, 1][:, None] - P[:, 1][None, :]
    r = np.sqrt(dx * dx + dy * dy) + 1e-12
    theta = np.arctan2(dy, dx)  # (-pi, pi]

    # 全局稳健尺度
    r_valid = r[(r > 1e-9)]
    med = np.median(r_valid) if r_valid.size > 0 else 1.0
    r_norm = r / max(med, 1e-6)

    # 预计算：半径/角度位置到“连续bin坐标”
    log_r = np.log10(r_norm + 1e-12)
    lo, hi = np.log10(r_inner), np.log10(r_outer)
    # 映射到 [0, nbins_r-1]
    r_pos = (log_r - lo) / max(hi - lo, 1e-6) * (nbins_r - 1)
    # 角度映射到 [0, nbins_theta)
    theta2 = (theta + np.pi) % (2 * np.pi)
    t_pos = theta2 / (2 * np.pi) * nbins_theta

    for i in range(N):
        mask = np.ones(N, dtype=bool); mask[i] = False
        if not soft:
            # 硬分箱
            r_idx = np.clip(np.floor(r_pos[i, mask]).astype(int), 0, nbins_r - 1)
            t_idx = np.clip(np.floor(t_pos[i, mask]).astype(int), 0, nbins_theta - 1)
            for rb, tb in zip(r_idx, t_idx):
                H[i, rb, tb] += 1.0
        else:
            # 软分箱：半径与角度分别按相邻两个箱做线性插值
            r_cont = r_pos[i, mask]
            t_cont = t_pos[i, mask]
            for rc, tc in zip(r_cont, t_cont):
                # 半径越界：夹到边界
                rc = float(np.clip(rc, 0.0, nbins_r - 1.0))
                tc = float(tc % nbins_theta)
                r_bins, r_ws = _soft_bin_1d(rc, nbins_r)
                t_bins, t_ws = _soft_bin_1d(tc, nbins_theta)
                # 双线性组合
                for rb, rw in zip(r_bins, r_ws):
                    for tb, tw in zip(t_bins, t_ws):
                        H[i, rb, tb] += rw * tw

        s = H[i].sum()
        if s > 0:
            H[i] /= s
    return H

def chi2_cost(h1: np.ndarray, h2: np.ndarray, eps: float = 1e-8) -> float:
    num = (h1 - h2) ** 2
    den = h1 + h2 + eps
    return float(0.5 * np.sum(num / den))

# ----------------- 距离签名 -----------------
def k_nn_signature(P: np.ndarray, idx: int, k: int) -> np.ndarray:
    N = P.shape[0]
    if N <= 1:
        return np.zeros((k,), dtype=np.float32)
    d = np.sqrt(((P[idx][None, :] - P) ** 2).sum(axis=1) + 1e-12)
    d[idx] = np.inf
    k_eff = int(min(k, N - 1))
    d_small = np.partition(d, k_eff - 1)[:k_eff]
    d_small.sort()
    denom = max(d_small[-1], 1e-6)
    sig_small = (d_small / denom).astype(np.float32)
    if k_eff < k:
        if k_eff == 0:
            return np.zeros((k,), dtype=np.float32)
        pad = np.full((k - k_eff,), sig_small[-1], dtype=np.float32)
        return np.concatenate([sig_small, pad], axis=0)
    return sig_small

# ----------------- 全局对齐（小角度网格） -----------------
def global_align_predict_q(P1: np.ndarray, q_idx: int, P2: np.ndarray,
                           rotation_grid_deg: Tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0),
                           use_scale: bool = True) -> Tuple[np.ndarray, float, float]:
    c1 = P1.mean(axis=0) if P1.size else np.zeros(2)
    c2 = P2.mean(axis=0) if P2.size else np.zeros(2)
    s = (_robust_scale(P2) / _robust_scale(P1)) if use_scale else 1.0

    v = P1[q_idx] - c1
    best_theta, best_qhat, best_score = 0.0, c2 + s * v, np.inf
    for deg in rotation_grid_deg:
        th = np.deg2rad(deg)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], dtype=np.float32)
        qhat = c2 + (R @ (s * v).reshape(2, 1)).reshape(2)
        d = np.sqrt(((P2 - qhat[None, :]) ** 2).sum(axis=1) + 1e-12)
        score = np.median(d)  # 稳健
        if score < best_score:
            best_score, best_theta, best_qhat = score, float(th), qhat.astype(np.float32)
    return best_qhat, float(s), float(best_theta)

# ----------------- 主预测（v3.1） -----------------
def predict_query_match(P1: np.ndarray, q_idx: int, P2: np.ndarray,
                        nbins_r: int = 5, nbins_theta: int = 12,
                        k_sig: int = 8,
                        # 代价权重
                        w_global_pos: float = 1.30,
                        w_sc: float = 1.20,
                        w_sig: float = 0.60,
                        w_centroid: float = 0.80,
                        w_quant: float = 0.80,
                        w_align: float = 1.20,
                        # 其它
                        rotation_grid_deg: Tuple[float, ...] = (-8.0, -4.0, 0.0, 4.0, 8.0),
                        tau: float = 0.35,
                        sc_soft: bool = True,
                        topk_global: int = 3,
                        k_local_align: int = 4,
                        dynamic_weight: bool = True) -> Tuple[int, np.ndarray, Dict[str, np.ndarray]]:
    """
    v3.1：全局小角度搜索 + TopK局部对齐 + SC软分箱 + 退化自适应
    返回：(pred_j, probs[M], debug)
    """
    N1, N2 = P1.shape[0], P2.shape[0]
    if N1 == 0 or N2 == 0 or q_idx < 0 or q_idx >= N1:
        return -1, np.zeros((N2,), dtype=np.float32), {}

    # 1) 全局对齐
    q_hat, s12, theta = global_align_predict_q(P1, q_idx, P2, rotation_grid_deg=rotation_grid_deg, use_scale=True)

    # 2) 各项代价
    # 2.1 全局位置
    C_global = np.sqrt(((P2 - q_hat[None, :]) ** 2).sum(axis=1) + 1e-12)

    # 2.2 定向 SC（可软分箱）
    SC1 = shape_context_oriented(P1, nbins_r=nbins_r, nbins_theta=nbins_theta, soft=sc_soft)
    SC2 = shape_context_oriented(P2, nbins_r=nbins_r, nbins_theta=nbins_theta, soft=sc_soft)
    hq = SC1[q_idx]
    C_sc = np.array([chi2_cost(hq, SC2[j]) for j in range(N2)], dtype=np.float32)

    # 2.3 距离签名
    k_eff = int(min(k_sig, N1 - 1, N2 - 1))
    if k_eff <= 0:
        C_sig = np.zeros((N2,), dtype=np.float32)
    else:
        sig_q = k_nn_signature(P1, q_idx, k_eff)
        C_sig = np.zeros((N2,), dtype=np.float32)
        for j in range(N2):
            C_sig[j] = np.linalg.norm(sig_q - k_nn_signature(P2, j, k_eff), ord=2)

    # 2.4 到质心半径一致性
    c1, c2 = P1.mean(axis=0), P2.mean(axis=0)
    rq = np.linalg.norm(P1[q_idx] - c1)
    r2 = np.linalg.norm(P2 - c2, axis=1)
    C_centroid = np.abs(rq * s12 - r2)

    # 2.5 分位数距离签名一致性（全局分布）
    Dx, Dy = _pairwise(P1), _pairwise(P2)
    qsig_x = _quantile_sig(Dx[q_idx][Dx[q_idx] > 0], q=10)
    C_quant = np.zeros((N2,), dtype=np.float32)
    for j in range(N2):
        jsig_y = _quantile_sig(Dy[j][Dy[j] > 0], q=10)
        C_quant[j] = np.linalg.norm(qsig_x - jsig_y / max(s12, 1e-6), ord=2)

    # 2.6 Top-K 全局候选的局部相似对齐残差
    K = int(min(max(topk_global, 0), N2))
    C_align = np.full((N2,), np.median(C_global) if N2 > 0 else 0.0, dtype=np.float32)  # 基线
    if K > 0:
        cand = np.argsort(C_global)[:K]
        for j in cand:
            C_align[j] = _local_align_residual(P1, q_idx, P2, j, k_local=k_local_align)

    # 3) 退化自适应权重
    if dynamic_weight:
        med_r1 = _robust_scale(P1)
        near_center = (rq <= 0.12 * med_r1)
        small_sample = (N1 <= 4 or N2 <= 4)
        if near_center or small_sample:
            w_global_pos *= 0.70
            w_sc *= 0.75
            w_sig *= 1.25
            w_quant *= 1.25
            w_align *= 1.20

    # 4) 融合
    C = (
        w_global_pos * _minmax_norm(C_global) +
        w_sc         * _minmax_norm(C_sc) +
        w_sig        * _minmax_norm(C_sig) +
        w_centroid   * _minmax_norm(C_centroid) +
        w_quant      * _minmax_norm(C_quant) +
        w_align      * _minmax_norm(C_align)
    ).astype(np.float32)

    # 5) softmax(-C / tau)
    scores = -C / max(tau, 1e-6)
    exp = np.exp(scores - scores.max())
    probs = (exp / max(exp.sum(), 1e-12)).astype(np.float32)
    pred_j = int(probs.argmax()) if N2 > 0 else -1

    debug = {
        "C_global": C_global, "C_sc": C_sc, "C_sig": C_sig,
        "C_centroid": C_centroid, "C_quant": C_quant, "C_align": C_align,
        "C_total": C, "q_hat": q_hat, "s12": np.array([s12], dtype=np.float32), "theta": np.array([theta], dtype=np.float32)
    }
    return pred_j, probs, debug
