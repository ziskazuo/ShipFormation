
"""

-------------
几何与点集相关的 numpy 工具函数。此文件**不依赖 PyTorch**，
因此可独立被 `visualization.py` 的可视化主函数调用。
"""
from __future__ import annotations
import numpy as np

def set_seed(seed: int = 42) -> None:
    """设置 numpy 的随机种子，便于复现实验。"""
    np.random.seed(seed)

def normalize_points(P: np.ndarray) -> np.ndarray:
    """
    将点集做“中心化 + 尺度归一化（RMS半径）”。
    输入: P [N,2]
    输出: 归一化后的点集 [N,2]
    """
    assert P.ndim == 2 and P.shape[1] == 2, "P 必须是 [N,2]"
    C = P.mean(axis=0, keepdims=True)
    Q = P - C
    rms = np.sqrt((Q**2).sum(axis=1).mean()) + 1e-8
    return Q / rms

def random_rotation_matrix(theta: float | None = None) -> np.ndarray:
    """
    生成二维旋转矩阵。若 theta=None，则从 [0, 2π) 均匀采样。
    """
    if theta is None:
        theta = np.random.rand() * 2.0 * np.pi
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)

def apply_se2(P: np.ndarray, R: np.ndarray | None = None,
              scale: float | None = None, t: np.ndarray | None = None) -> np.ndarray:
    """
    对点集施加 SE(2) 仿射：旋转+尺度+平移。
    任意一项为 None 时取默认：R=单位旋转, scale=1.0, t=0。
    """
    if R is None:
        R = np.eye(2, dtype=float)
    if scale is None:
        scale = 1.0
    if t is None:
        t = np.zeros((1, 2), dtype=float)
    return (P @ R.T) * scale + t

def pairwise_distances(P: np.ndarray) -> np.ndarray:
    """返回点集的欧氏距离矩阵 [N,N]。"""
    diff = P[:, None, :] - P[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1) + 1e-12)

def knn_graph(P: np.ndarray, k: int = 3) -> np.ndarray:
    """
    基于欧氏距离构建 kNN 图的邻接矩阵（无向、去自环）
    返回 A [N,N]，A[i,j]∈{0,1}。
    """
    D = pairwise_distances(P)
    N = P.shape[0]
    A = np.zeros((N, N), dtype=np.float32)
    # 对每个点取最近的 k 个（不含自身）
    idx = np.argsort(D, axis=1)[:, 1:k+1]
    for i in range(N):
        for j in idx[i]:
            A[i, j] = 1.0
            A[j, i] = 1.0
    return A

def add_gaussian_noise(P: np.ndarray, sigma: float = 0.01) -> np.ndarray:
    """对点集加入各向同性高斯噪声。"""
    return P + np.random.randn(*P.shape) * sigma
