"""
data.py
-------
将 visualization.sample_pair / sample_single_formation_pair 封装为 PyTorch 友好的数据产生器。
"""
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np
import torch

# 如果你的文件结构是包 Data/visualization.py，保留这一行；
# 若与 visualization.py 同目录，请改为: from visualization import sample_pair, sample_single_formation_pair
from Data.visualization import sample_pair, sample_single_formation_pair

def numpy_to_torch(P: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(P.astype("float32"))

def sample_torch_pair(**kwargs) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], int, int]:
    """
    生成一个“混合编队”样本对并转为 torch 张量（保持原有接口不变）。
    """
    X, Y, mapping, q_idx, y_idx = sample_pair(**kwargs)
    return numpy_to_torch(X), numpy_to_torch(Y), mapping, q_idx, y_idx

# === 新增：单一编队样本 ===
_DEFAULT_KINDS: Tuple[str, ...] = (
    "v", "single_column", "double_column", "single_row",
    "double_row", "echelon_left", "echelon_right", "circular"
)

def sample_torch_single_pair(
    kind: Optional[str] = None,
    n: Optional[int] = None,
    n_range: Tuple[int, int] = (8, 12),
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor, Dict[int, int], int, int]:
    """
    生成一个“单一编队”的 (X, Y) 样本并转为 torch 张量。
    - kind: 指定编队类型；若为 None 将从常见类型里随机选择
    - n:    指定舰数；若为 None 将在 n_range 内随机
    - 其余参数直接透传给 visualization.sample_single_formation_pair（如 noise_sigma 等）
    """
    if kind is None:
        kind = np.random.choice(_DEFAULT_KINDS)
    if n is None:
        n = int(np.random.randint(n_range[0], n_range[1] + 1))

    X, Y, mapping, q_idx, y_idx = sample_single_formation_pair(kind=kind, n=n, **kwargs)
    return numpy_to_torch(X), numpy_to_torch(Y), mapping, q_idx, y_idx

def make_row_targets(N: int, M: int, mapping: Dict[int, int], q_idx: int) -> int:
    """
    构造“单行监督”的目标列索引（含 dustbin）。
    若 q 在 Y 中有对应 j，则返回 j；否则返回 M（即 dustbin 列）。
    """
    return mapping.get(int(q_idx), M)
