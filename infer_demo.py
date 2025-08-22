"""
infer_demo.py
-------------
批量加载训练好的模型，在若干随机样本上做推理与可视化：
- snapshots/infer_inputs_###.png         （X 与 Y 的点分布，X 上标出查询点）
- snapshots/infer_pred_vs_gt_###.png     （X、Y[GT]、Y[PRED] 三联图）
- snapshots/infer_row_prob_###.png       （查询行的匹配概率分布，最后一列为 dustbin）
- snapshots/infer_P_heatmap_###.png      （完整匹配矩阵热力图，含 dustbin）
- snapshots/infer_summary.csv            （每个样本的结果汇总）
用法示例：
    python infer_demo.py --num 12 --mode mixed
    python infer_demo.py --num 20 --mode single
"""
from __future__ import annotations
import os, csv, argparse
import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")  # 无交互后端
import matplotlib.pyplot as plt

# —— 中文字体设置（可选） ——
from matplotlib import rcParams, font_manager
_CJK_FALLBACKS = [
    "Noto Sans CJK SC", "Source Han Sans SC",
    "Microsoft YaHei", "SimHei",
    "PingFang SC", "WenQuanYi Zen Hei", "Arial Unicode MS"
]
for fname in _CJK_FALLBACKS:
    try:
        path = font_manager.findfont(fname, fallback_to_default=False)
        if os.path.exists(path):
            rcParams["font.sans-serif"] = [fname]
            break
    except Exception:
        pass
local_font = os.path.join(os.path.dirname(__file__), "fonts", "NotoSansCJKsc-Regular.otf")
if os.path.exists(local_font):
    font_manager.fontManager.addfont(local_font)
    rcParams["font.sans-serif"] = ["Noto Sans CJK SC"]
rcParams["axes.unicode_minus"] = False

from model import MatchingModel
from data import sample_torch_pair, make_row_targets
# 若有“单一编队”采样器则可选用
try:
    from data import sample_torch_single_pair
except Exception:
    sample_torch_single_pair = None

SNAP_DIR = os.path.join(os.path.dirname(__file__), "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)
CKPT = os.path.join(SNAP_DIR, "matching.pt")

# -------------------- 画图工具 --------------------
def _scatter(ax, P: np.ndarray, color: str, label: str, annotate: bool = True):
    if P.size == 0:
        ax.set_title(label + " (EMPTY)")
        return
    ax.scatter(P[:, 0], P[:, 1], s=40, c=color, label=label, edgecolors="k", linewidths=0.5)
    if annotate:
        for i, (x, y) in enumerate(P):
            ax.text(x, y, str(i), fontsize=8, ha="center", va="bottom")

def _equalize(ax):
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

def load_model_or_warn(device: str = "cpu") -> MatchingModel:
    # 新模型的参数名
    model = MatchingModel(
        k=5,
        d_model=128,      # ← 取代 emb_dim
        nhead=4,
        num_layers=4,
        dropout=0.1,
        rbf_kernels=16,
        learnable_dustbin=True,
        dustbin_bias=-2.0,
    ).to(device)

    if os.path.exists(CKPT):
        try:
            sd = torch.load(CKPT, map_location=device)
            model.load_state_dict(sd, strict=True)  # 旧权重若架构不一致会抛异常
            print(f"已加载模型: {CKPT}")
        except Exception as e:
            print(f"加载 {CKPT} 失败：{e}\n将使用随机初始化模型。请用新的 model.py 重新训练（python train.py）。")
    else:
        print("未找到训练权重，将使用随机初始化模型（效果可能较差）。请先运行 python train.py 进行训练。")
    model.eval()
    return model


def sample_one(mode: str):
    """根据 mode 采样一个样本对。"""
    if mode == "single":
        if sample_torch_single_pair is None:
            print("[提示] 未检测到 data.sample_torch_single_pair，已回退为 mixed。")
            return sample_torch_pair()
        return sample_torch_single_pair()
    # mixed
    return sample_torch_pair()

def run_one(idx: int, model: MatchingModel, device: str, mode: str):
    """跑单个样本并保存4张图与结果，返回记录字典。"""
    X, Y, mapping, q_idx, _ = sample_one(mode)
    X = X.to(device); Y = Y.to(device)

    with torch.no_grad():
        P = model(X, Y, sinkhorn_iters=20)  # [N+1, M+1]
    Np1, Mp1 = P.shape
    N, M = Np1 - 1, Mp1 - 1

    row = P[q_idx, :].detach().cpu().numpy()   # [M+1]
    pred_j = int(row[:-1].argmax())
    conf = float(row[pred_j])
    gt_j = make_row_targets(N, M, mapping, q_idx)

    # numpy for plots
    X_np = X.detach().cpu().numpy()
    Y_np = Y.detach().cpu().numpy()
    P_np = P.detach().cpu().numpy()

    tag = f"{idx:03d}"

    # 图A：输入分布
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes
    _scatter(ax1, X_np, color="#1f77b4", label="X", annotate=True)
    ax1.scatter(X_np[q_idx, 0], X_np[q_idx, 1], s=120, facecolors="none",
                edgecolors="r", linewidths=2.0, label="query")
    ax1.set_title(f"输入视图 X（查询 q_idx={q_idx}）"); _equalize(ax1)

    _scatter(ax2, Y_np, color="#2ca02c", label="Y", annotate=True)
    ax2.set_title("输入视图 Y"); _equalize(ax2)
    out_inputs = os.path.join(SNAP_DIR, f"infer_inputs_{tag}.png")
    plt.tight_layout(); plt.savefig(out_inputs, dpi=160); plt.close()

    # 图B：GT vs PRED
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _scatter(axes[0], X_np, color="#1f77b4", label="X", annotate=True)
    axes[0].scatter(X_np[q_idx, 0], X_np[q_idx, 1], s=140, facecolors="none",
                    edgecolors="r", linewidths=2.0, label="query")
    axes[0].set_title(f"X（查询 q_idx={q_idx}）"); _equalize(axes[0])

    _scatter(axes[1], Y_np, color="#2ca02c", label="Y", annotate=True)
    if gt_j == M:
        axes[1].set_title("Y（GT=dustbin：Y 中不存在对应）")
    else:
        axes[1].scatter(Y_np[gt_j, 0], Y_np[gt_j, 1], s=160, facecolors="none",
                        edgecolors="b", linewidths=2.0, label="GT")
        axes[1].set_title(f"Y（GT 索引={gt_j}）")
    _equalize(axes[1])

    _scatter(axes[2], Y_np, color="#2ca02c", label="Y", annotate=True)
    axes[2].scatter(Y_np[pred_j, 0], Y_np[pred_j, 1], s=160, facecolors="none",
                    edgecolors="r", linewidths=2.0, label="PRED")
    axes[2].set_title(f"Y（预测 PRED 索引={pred_j} | 置信度={conf:.3f}）")
    _equalize(axes[2])

    plt.suptitle("查询目标的 GT 与 PRED 对比", fontsize=12)
    out_predgt = os.path.join(SNAP_DIR, f"infer_pred_vs_gt_{tag}.png")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(out_predgt, dpi=160); plt.close()

    # 图C：查询行概率分布
    x = np.arange(M+1)
    plt.figure(figsize=(6, 3))
    stem_out = plt.stem(x, row)
    try:
        markerline, stemlines, baseline = stem_out
        plt.setp(stemlines, linewidth=1.0); plt.setp(markerline, markersize=4)
    except Exception:
        pass
    plt.title("查询行的匹配概率分布（最后一个为 dustbin）")
    plt.xlabel("Y 索引（含 dustbin=M）"); plt.ylabel("P")
    plt.tight_layout()
    out_row = os.path.join(SNAP_DIR, f"infer_row_prob_{tag}.png")
    plt.savefig(out_row, dpi=160); plt.close()

    # 图D：P 的热力图
    plt.figure(figsize=(6, 5))
    im = plt.imshow(P_np, aspect="auto", cmap="viridis")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title("匹配矩阵 P（含 dustbin 行/列）")
    plt.xlabel("Y 列（含 dustbin=M）"); plt.ylabel("X 行（含 dustbin=N）")
    plt.tight_layout()
    out_heat = os.path.join(SNAP_DIR, f"infer_P_heatmap_{tag}.png")
    plt.savefig(out_heat, dpi=160); plt.close()

    hit = (pred_j == gt_j)
    rec = {
        "idx": idx,
        "mode": mode,
        "N": int(N),
        "M": int(M),
        "q_idx": int(q_idx),
        "pred_j": int(pred_j),
        "gt_j": int(gt_j),
        "conf": float(conf),
        "hit": int(hit),
        "gt_is_dustbin": int(gt_j == M),
        "inputs_png": out_inputs,
        "pred_vs_gt_png": out_predgt,
        "row_prob_png": out_row,
        "heatmap_png": out_heat,
    }
    return rec

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=8, help="测试样本个数")
    parser.add_argument("--device", type=str, default="cpu", help="cpu 或 cuda")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--mode", type=str, default="mixed", choices=["mixed", "single"],
                        help="mixed=混合编队；single=单一编队（若实现了 sample_torch_single_pair）")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = load_model_or_warn(device=args.device)

    # 多样本测试
    recs = []
    for i in range(1, args.num + 1):
        rec = run_one(i, model=model, device=args.device, mode=args.mode)
        recs.append(rec)

    # 输出汇总 CSV 和总体命中率
    csv_path = os.path.join(SNAP_DIR, "infer_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
        writer.writeheader()
        writer.writerows(recs)
    acc = np.mean([r["hit"] for r in recs]) if recs else 0.0
    print(f"\n已保存汇总：{csv_path}")
    print(f"总体命中率（top-1）：{acc:.3f}  （样本数={len(recs)}）")

if __name__ == "__main__":
    main()
