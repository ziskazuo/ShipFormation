
"""
ot_utils.py
-----------
与最优传输/匹配相关的 PyTorch 工具，包括可微 Sinkhorn 归一化、
以及“dustbin”扩展（用于成员数不等/杂点的情况）。
"""
from __future__ import annotations
from typing import Tuple
import torch
import torch.nn.functional as F

def sinkhorn(log_alpha: torch.Tensor, n_iters: int = 30, eps: float = 1e-9) -> torch.Tensor:
    """
    对 logit 矩阵 log_alpha（形状 [N,M]）做 Sinkhorn 迭代，
    输出近似的双随机矩阵 P（各行列 ~1）。
    注意：输入是“log-空间”，内部使用 softmax 正则化的稳定实现。
    """
    # 初始在 log 空间
    log_P = log_alpha
    for _ in range(n_iters):
        # 规范每一行
        log_P = log_P - torch.logsumexp(log_P, dim=1, keepdim=True)
        # 规范每一列
        log_P = log_P - torch.logsumexp(log_P, dim=0, keepdim=True)
    P = torch.exp(log_P).clamp_min(eps)
    return P

def add_dustbin(logits: torch.Tensor, dustbin_value: float = -4.0) -> torch.Tensor:
    """
    在匹配矩阵的最后一行与最后一列添加“dustbin”（未匹配槽）。
    输入 logits: [N,M]，返回 [N+1, M+1]。
    """
    N, M = logits.shape
    device = logits.device
    # 扩展到 [N+1, M+1] 并填充 dustbin
    extended = torch.full((N + 1, M + 1), dustbin_value, device=device, dtype=logits.dtype)
    extended[:N, :M] = logits
    return extended

def row_ce_loss(P: torch.Tensor, target_index: torch.Tensor) -> torch.Tensor:
    """
    针对“只关心其中一行匹配”的场景：
    P: [N+1, M+1] 的双随机矩阵（含 dustbin）
    target_index: 长度为1的张量，给出查询节点在 X 中的索引 i0；
                  我们用该行与监督的列（含可能的 dustbin）计算交叉熵。
    这里的监督列 index 需要在外部提供。
    """
    raise NotImplementedError("该函数仅做占位，训练时请使用全矩阵监督或单行监督自定义实现。")
