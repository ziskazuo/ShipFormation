"""
visualization.py
--------------
参数化海面编队生成器。包含常见编队（V/单纵/双纵/单横/双横/梯队/包围）
以及"混合编队"的生成；并提供可视化函数。
本文件仅依赖 numpy 与 matplotlib，不依赖 PyTorch。
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from pathlib import Path

from Data.geom_utils import (
    set_seed, normalize_points, random_rotation_matrix, apply_se2,
    add_gaussian_noise
)

# ------------------------- 基本形态生成 -------------------------
def gen_V(n: int, spacing: float = 1.0, angle_deg: float = 60.0) -> np.ndarray:
    """V 型队列"""
    n_left = n // 2
    n_right = n - n_left - 1  # 留一个 apex
    theta = np.deg2rad(angle_deg / 2.0)
    pts = [np.array([0.0, 0.0])]
    for i in range(1, n_left + 1):  # 左臂
        dx = -i * spacing * np.cos(theta)
        dy = -i * spacing * np.sin(theta)
        pts.append(np.array([dx, dy]))
    for i in range(1, n_right + 1):  # 右臂
        dx =  i * spacing * np.cos(theta)
        dy = -i * spacing * np.sin(theta)
        pts.append(np.array([dx, dy]))
    P = np.stack(pts, axis=0)
    return P - P.mean(axis=0, keepdims=True)

def gen_single_column(n: int, spacing: float = 1.0) -> np.ndarray:
    y = np.arange(n, dtype=float) * -spacing
    x = np.zeros_like(y)
    P = np.stack([x, y], axis=1)
    return P - P.mean(axis=0, keepdims=True)

def gen_double_column(n: int, spacing: float = 1.0, lane_gap: float = 1.0) -> np.ndarray:
    n_left = n // 2
    n_right = n - n_left
    y_left  = np.arange(n_left, dtype=float)  * -spacing
    y_right = np.arange(n_right, dtype=float) * -spacing
    x_left  = np.full_like(y_left,  -lane_gap/2.0)
    x_right = np.full_like(y_right,  lane_gap/2.0)
    P = np.concatenate([np.stack([x_left, y_left], 1),
                        np.stack([x_right, y_right], 1)], axis=0)
    return P - P.mean(axis=0, keepdims=True)

def gen_single_row(n: int, spacing: float = 1.0) -> np.ndarray:
    x = np.arange(n, dtype=float) * spacing
    y = np.zeros_like(x)
    P = np.stack([x, y], axis=1)
    return P - P.mean(axis=0, keepdims=True)

def gen_double_row(n: int, spacing: float = 1.0, lane_gap: float = 1.0) -> np.ndarray:
    n_top = n // 2
    n_bottom = n - n_top
    x_top = np.arange(n_top, dtype=float) * spacing
    x_bot = np.arange(n_bottom, dtype=float) * spacing
    y_top = np.full_like(x_top,  lane_gap/2.0)
    y_bot = np.full_like(x_bot, -lane_gap/2.0)
    P = np.concatenate([np.stack([x_top, y_top], 1),
                        np.stack([x_bot, y_bot], 1)], axis=0)
    return P - P.mean(axis=0, keepdims=True)

def gen_echelon(n: int, spacing: float = 1.0, left: bool = True) -> np.ndarray:
    """梯队：left=True 左梯队，False 右梯队"""
    i = np.arange(n, dtype=float)
    x = (-i if left else i) * spacing
    y = -i * spacing * 0.8
    P = np.stack([x, y], axis=1)
    return P - P.mean(axis=0, keepdims=True)

def gen_circular(n: int, radius: float = 3.0) -> np.ndarray:
    angles = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    P = np.stack([x, y], axis=1)
    return P - P.mean(axis=0, keepdims=True)

def generate_formation(kind: str, n: int, **kwargs) -> np.ndarray:
    """统一的编队生成接口。"""
    kind = kind.lower()
    if kind in ["v", "v型", "vshape", "v-form", "vformation"]:
        return gen_V(n, spacing=kwargs.get("spacing", 1.0), angle_deg=kwargs.get("angle_deg", 60.0))
    if kind in ["single_column", "单纵", "单纵队"]:
        return gen_single_column(n, spacing=kwargs.get("spacing", 1.0))
    if kind in ["double_column", "双纵", "双纵队"]:
        return gen_double_column(n, spacing=kwargs.get("spacing", 1.0), lane_gap=kwargs.get("lane_gap", 1.0))
    if kind in ["single_row", "单横", "单横队"]:
        return gen_single_row(n, spacing=kwargs.get("spacing", 1.0))
    if kind in ["double_row", "双横", "双横队"]:
        return gen_double_row(n, spacing=kwargs.get("spacing", 1.0), lane_gap=kwargs.get("lane_gap", 1.0))
    if kind in ["echelon_left", "左梯队"]:
        return gen_echelon(n, spacing=kwargs.get("spacing", 1.0), left=True)
    if kind in ["echelon_right", "右梯队"]:
        return gen_echelon(n, spacing=kwargs.get("spacing", 1.0), left=False)
    if kind in ["circular", "包围", "包围型"]:
        return gen_circular(n, radius=kwargs.get("radius", 3.0))
    raise ValueError(f"未知编队类型: {kind}")

# ------------------------- 混合编队与样本对 -------------------------
def mix_formations(kinds: List[str], counts: List[int],
                   offsets: List[Tuple[float, float]] | None = None,
                   **kwargs) -> np.ndarray:
    """将若干子编队平移后合并为一个"混合编队"""
    assert len(kinds) == len(counts), "kinds 与 counts 长度需一致"
    if offsets is None:
        offsets = [(0.0, 0.0) for _ in kinds]
    assert len(offsets) == len(kinds), "offsets 长度需与 kinds 相同"
    parts = []
    for kind, n, ofs in zip(kinds, counts, offsets):
        P = generate_formation(kind, n, **kwargs)
        P = P + np.array(ofs, dtype=float)[None, :]
        parts.append(P)
    if not parts:
        return np.zeros((0, 2), dtype=float)
    P = np.concatenate(parts, axis=0)
    return P - P.mean(axis=0, keepdims=True)

def sample_single_formation_pair(
    kind: str,
    n: int,
    noise_sigma: float = 0.02,
    drop_rate: float = 0.1,  # 默认不丢弃点
    add_clutter_rate: float = 0.1,  # 默认不添加杂点
    point_offset_range: float = 0.2,
    point_offset_prob: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, dict, int, int]:
    """为单种编队生成一个 (X,Y) 样本对"""
    # 生成原始编队
    X = generate_formation(kind, n)

    # 应用随机变换
    R = random_rotation_matrix()
    scale = np.random.uniform(0.7, 1.3)
    t = (np.random.randn(1, 2) * 0.5)
    Y_clean = apply_se2(X, R=R, scale=scale, t=t)
    Y_noisy = add_gaussian_noise(Y_clean, sigma=noise_sigma)

    # 添加点偏移
    if point_offset_range > 0 and point_offset_prob > 0:
        offset_mask = np.random.rand(Y_noisy.shape[0]) < point_offset_prob
        point_offsets = np.random.uniform(-point_offset_range, point_offset_range, size=(Y_noisy.shape[0], 2))
        Y_noisy[offset_mask] = Y_noisy[offset_mask] + point_offsets[offset_mask]

    # 随机丢弃点
    N = X.shape[0]
    keep_mask = np.random.rand(N) > drop_rate
    q_idx = int(np.random.randint(0, N))
    keep_mask[q_idx] = True  # 确保查询点不被丢弃

    X_kept_idx = np.where(keep_mask)[0].tolist()
    Y = Y_noisy[keep_mask]
    mapping = {int(old_i): int(new_j) for new_j, old_i in enumerate(X_kept_idx)}

    # 添加杂点
    m_add = int(np.round(Y.shape[0] * add_clutter_rate))
    if m_add > 0:
        min_xy = Y.min(axis=0)
        max_xy = Y.max(axis=0)
        rnd = np.random.rand(m_add, 2) * (max_xy - min_xy)[None, :] + min_xy[None, :]
        Y = np.concatenate([Y, rnd], axis=0)

    y_idx = mapping.get(int(q_idx), -1)
    X_norm = normalize_points(X)
    Y_norm = normalize_points(Y)
    return X_norm, Y_norm, mapping, int(q_idx), int(y_idx)

def sample_pair(
    kinds_pool: List[str] = ("v", "single_column", "double_column", "single_row", "double_row",
                             "echelon_left", "echelon_right", "circular"),
    n_subforms: Tuple[int, int] = (1, 3),
    n_range: Tuple[int, int] = (4, 12),
    noise_sigma: float = 0.02,
    drop_rate: float = 0.1,
    add_clutter_rate: float = 0.1,
    point_offset_range: float = 0.2,
    point_offset_prob: float = 0.7,
) -> Tuple[np.ndarray, np.ndarray, dict, int, int]:
    """生成一个 (X,Y) 混合编队样本对"""
    s = np.random.randint(n_subforms[0], n_subforms[1] + 1)
    kinds = list(np.random.choice(kinds_pool, size=s, replace=False))
    counts = [np.random.randint(n_range[0], n_range[1] + 1) for _ in range(s)]
    offsets = list((np.random.randn(2) * 3.0).tolist() for _ in range(s))
    X = mix_formations(kinds, counts, offsets)

    R = random_rotation_matrix()
    scale = np.random.uniform(0.7, 1.3)
    t = (np.random.randn(1, 2) * 0.5)
    Y_clean = apply_se2(X, R=R, scale=scale, t=t)
    Y_noisy = add_gaussian_noise(Y_clean, sigma=noise_sigma)

    if point_offset_range > 0 and point_offset_prob > 0:
        offset_mask = np.random.rand(Y_noisy.shape[0]) < point_offset_prob
        point_offsets = np.random.uniform(-point_offset_range, point_offset_range, size=(Y_noisy.shape[0], 2))
        Y_noisy[offset_mask] = Y_noisy[offset_mask] + point_offsets[offset_mask]

    N = X.shape[0]
    keep_mask = np.random.rand(N) > drop_rate
    q_idx = int(np.random.randint(0, N))
    keep_mask[q_idx] = True

    X_kept_idx = np.where(keep_mask)[0].tolist()
    Y = Y_noisy[keep_mask]
    mapping = {int(old_i): int(new_j) for new_j, old_i in enumerate(X_kept_idx)}

    m_add = int(np.round(Y.shape[0] * add_clutter_rate))
    if m_add > 0:
        min_xy = Y.min(axis=0)
        max_xy = Y.max(axis=0)
        rnd = np.random.rand(m_add, 2) * (max_xy - min_xy)[None, :] + min_xy[None, :]
        Y = np.concatenate([Y, rnd], axis=0)

    y_idx = mapping.get(int(q_idx), -1)
    X_norm = normalize_points(X)
    Y_norm = normalize_points(Y)
    return X_norm, Y_norm, mapping, int(q_idx), int(y_idx)

# ------------------------- 可视化函数 -------------------------
def _scatter(ax, P: np.ndarray, color: str, label: str):
    ax.scatter(P[:, 0], P[:, 1], s=40, c=color, label=label, edgecolors='k', linewidths=0.5)

def visualize_pair(X: np.ndarray, Y: np.ndarray, mapping: Dict[int, int], q_idx: int, y_idx: int,
                   save_path: str | None = None):
    """可视化混合编队的两个视图（X/Y），并高亮查询/对应目标"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    # 绘制X视图
    _scatter(ax1, X, color="#1f77b4", label="X")
    ax1.scatter(X[q_idx, 0], X[q_idx, 1], s=120, facecolors='none', edgecolors='r', linewidths=2.0, label="query")
    ax1.set_title("View X (red circle = query)")
    ax1.axis('equal'); ax1.grid(True, ls=':'); ax1.legend(loc='best')

    # 绘制Y视图
    _scatter(ax2, Y, color="#2ca02c", label="Y")
    if 0 <= y_idx < Y.shape[0]:
        ax2.scatter(Y[y_idx, 0], Y[y_idx, 1], s=120, facecolors='none', edgecolors='r', linewidths=2.0, label="target")

        # 绘制对应关系线
        for x_idx, y_idx_mapped in mapping.items():
            if y_idx_mapped < Y.shape[0]:  # 确保映射点存在
                ax2.plot([X[x_idx, 0], Y[y_idx_mapped, 0]],
                         [X[x_idx, 1], Y[y_idx_mapped, 1]],
                         'k--', alpha=0.3, linewidth=0.5)

    ax2.set_title("View Y (red circle = match if exists)")
    ax2.axis('equal'); ax2.grid(True, ls=':'); ax2.legend(loc='best')

    plt.tight_layout()
    if save_path is None:
        save_path = 'Dataset/fleet_pair_preview.png'
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160); plt.close()
    print(f'预览图已保存: {save_path}')

def visualize_single_pair(X: np.ndarray, Y: np.ndarray, mapping: Dict[int, int], q_idx: int, y_idx: int,
                         title: str, save_path: str) -> None:
    """单种编队的双视图可视化，显示对应关系"""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    # 绘制X视图
    _scatter(ax1, X, color="#1f77b4", label="X")
    ax1.scatter(X[q_idx, 0], X[q_idx, 1], s=120, facecolors='none', edgecolors='r', linewidths=2.0, label="query")
    ax1.set_title(f"{title} - View X")
    ax1.axis('equal'); ax1.grid(True, ls=':'); ax1.legend(loc='best')

    # 绘制Y视图
    _scatter(ax2, Y, color="#2ca02c", label="Y")
    if 0 <= y_idx < Y.shape[0]:
        ax2.scatter(Y[y_idx, 0], Y[y_idx, 1], s=120, facecolors='none', edgecolors='r', linewidths=2.0, label="target")

        # 绘制对应关系线
        for x_idx, y_idx_mapped in mapping.items():
            if y_idx_mapped < Y.shape[0]:  # 确保映射点存在
                ax2.plot([X[x_idx, 0], Y[y_idx_mapped, 0]],
                         [X[x_idx, 1], Y[y_idx_mapped, 1]],
                         'k--', alpha=0.3, linewidth=0.5)

    ax2.set_title(f"{title} - View Y")
    ax2.axis('equal'); ax2.grid(True, ls=':'); ax2.legend(loc='best')

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160); plt.close()
    print(f'已保存: {save_path}')

def visualize_single(P: np.ndarray, title: str, save_path: str) -> None:
    """单种编队的单视图可视化"""
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    _scatter(ax, P, color="#1f77b4", label=title)
    ax.set_title(title)
    ax.axis('equal'); ax.grid(True, ls=':') #ax.legend(loc='best')
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=160); plt.close()
    print(f'已保存: {save_path}')

# ------------------------- 批量生成：每种编队各一张 -------------------------
def demo_visualize_each_kind(n_range: Tuple[int, int] = (8, 12), seed: int = 42) -> None:
    """
    为每一种编队各生成一张图，保存到 Dataset/each_kind 目录。
    n_range: 每种编队的舰船数量随机范围
    """
    set_seed(seed)
    kinds = [
        "v",
        "single_column",
        "double_column",
        "single_row",
        "double_row",
        "echelon_left",
        "echelon_right",
        "circular",
    ]
    out_dir = Path("Dataset/each_kind")
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in kinds:
        n = int(np.random.randint(n_range[0], n_range[1] + 1))
        # 也可以在此根据 kind 定制 n、spacing、radius 等参数
        P = generate_formation(kind, n)
        P = normalize_points(P)  # 为一致性，做一次归一化
        title = f"{kind}"
        save_path = out_dir / f"formation_{kind}.png"
        visualize_single(P, title, str(save_path))

def demo_visualize_each_kind_pair(n_range: Tuple[int, int] = (8, 12), seed: int = 42) -> None:
    """
    为每一种编队各生成一对视图（X和Y），并显示对应关系，保存到 Dataset/each_kind_pair 目录。
    n_range: 每种编队的舰船数量随机范围
    """
    set_seed(seed)
    kinds = [
        "v",
        "single_column",
        "double_column",
        "single_row",
        "double_row",
        "echelon_left",
        "echelon_right",
        "circular",
    ]
    out_dir = Path("Dataset/each_kind_pair")
    out_dir.mkdir(parents=True, exist_ok=True)

    for kind in kinds:
        n = int(np.random.randint(n_range[0], n_range[1] + 1))
        # 生成单种编队的样本对
        X, Y, mapping, q_idx, y_idx = sample_single_formation_pair(kind, n)
        title = f"Formation: {kind}"
        save_path = out_dir / f"formation_{kind}_pair.png"
        visualize_single_pair(X, Y, mapping, q_idx, y_idx, title, str(save_path))

# ------------------------- 仍保留的混合编队批量预览 -------------------------
def demo_visualize(num_examples: int = 4, seed: int = 42) -> None:
    """生成多个混合编队样本对并保存为编号图片（保留原有功能）"""
    set_seed(seed)
    output_dir = Path('Dataset/mixed_kind/')
    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(num_examples):
        X, Y, mapping, q_idx, y_idx = sample_pair()
        save_path = output_dir / f'fleet_pair_{i + 1:03d}.png'
        visualize_pair(X, Y, mapping, q_idx, y_idx, save_path=str(save_path))

# ------------------------- 主入口 -------------------------
if __name__ == "__main__":
    # 运行： python visualization.py
    # 生成单种编队的单视图
    demo_visualize_each_kind(n_range=(8, 12), seed=42)

    # 生成单种编队的双视图（带对应关系）
    demo_visualize_each_kind_pair(n_range=(8, 12), seed=42)

    # 生成混合编队样本对
    demo_visualize(num_examples=20, seed=42)