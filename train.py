"""
train.py
--------
一个极简训练脚本：在合成数据上训练 MatchingModel，
目标是让模型仅在“查询节点那一行”给出正确对应（行级监督）。
使用说明：
    python train.py
会在 snapshots 下保存 'matching.pt'。
"""
from __future__ import annotations
import os, time
from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from model import MatchingModel
from data import sample_torch_pair, sample_torch_single_pair, make_row_targets

SNAP_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

def train(
    seed: int = 42,
    steps: int = 20000,
    lr: float = 1e-3,
    device: str = "cpu",
    single_prob: float = 0.5,  # 使用“单一编队”样本的概率
) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ====== 修改点 1：用新模型的参数名 ======
    model = MatchingModel(
        k=5,              # 距离签名长度
        d_model=128,      # 取代原来的 emb_dim
        nhead=4,
        num_layers=4,
        dropout=0.1,
        rbf_kernels=16,
        learnable_dustbin=True,   # 可学习的 dustbin（推荐开启）
        dustbin_bias=-2.0
    ).to(device)
    model.train()

    opt = optim.Adam(model.parameters(), lr=lr)

    avg_loss = 0.0
    for step in range(1, steps + 1):
        # 按概率混合“单一编队”和“混合编队”样本
        if np.random.rand() < single_prob:
            X, Y, mapping, q_idx, y_idx = sample_torch_single_pair()
        else:
            X, Y, mapping, q_idx, y_idx = sample_torch_pair()

        X = X.to(device); Y = Y.to(device)

        # ====== 修改点 2：可把迭代数略增（可选） ======
        P = model(X, Y, sinkhorn_iters=50)  # [N+1, M+1]
        Np1, Mp1 = P.shape
        N, M = Np1 - 1, Mp1 - 1

        row = P[q_idx, :]  # [M+1]
        target_j = make_row_targets(N, M, mapping, q_idx)
        target = torch.tensor([target_j], dtype=torch.long, device=device)

        # NLL on probabilities (row 已是概率分布)
        loss = F.nll_loss(torch.log(row[None, :] + 1e-12), target)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        avg_loss = avg_loss * 0.99 + float(loss.item()) * 0.01

        if step % 200 == 0:
            print(f"[{step:04d}/{steps}] loss={avg_loss:.4f}")

    model.eval()
    ckpt_path = os.path.join(SNAP_DIR, "matching.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"模型已保存到: {ckpt_path}")
    return ckpt_path

if __name__ == "__main__":
    train()
