# traditional_match_eval_v3.py
from __future__ import annotations
import os, re
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 中文字体兜底（可选）
from matplotlib import rcParams, font_manager
_CJK_FALLBACKS = ["Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei","SimHei","PingFang SC","WenQuanYi Zen Hei","Arial Unicode MS"]
for fname in _CJK_FALLBACKS:
    try:
        path = font_manager.findfont(fname, fallback_to_default=False)
        if os.path.exists(path):
            rcParams["font.sans-serif"] = [fname]; break
    except Exception:
        pass
rcParams["axes.unicode_minus"] = False

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def parse_full_txt(path: str):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if len(lines) < 6:
        raise ValueError(f"{path} 行数不足 6，实际 {len(lines)}")
    N1, T1 = map(float, lines[0].split()); N1, T1 = int(N1), int(T1)
    xs1 = list(map(float, lines[1].split())); ys1 = list(map(float, lines[2].split()))
    if len(xs1) < N1 or len(ys1) < N1: raise ValueError(f"{path} 目指 X/Y 数量不足")
    P1 = np.stack([np.array(xs1[:N1]), np.array(ys1[:N1])], axis=1)

    N2, T2 = map(float, lines[3].split()); N2, T2 = int(N2), int(T2)
    xs2 = list(map(float, lines[4].split())); ys2 = list(map(float, lines[5].split()))
    if len(xs2) < N2 or len(ys2) < N2: raise ValueError(f"{path} 测量 X/Y 数量不足")
    P2 = np.stack([np.array(xs2[:N2]), np.array(ys2[:N2])], axis=1)

    T1 = T1 - 1 if 1 <= T1 <= N1 else -1
    T2 = T2 - 1 if 1 <= T2 <= N2 else -1
    return (N1, T1, P1), (N2, T2, P2)

def _scatter(ax, P: np.ndarray, color: str, label: str, annotate: bool = False):
    if P.size == 0:
        ax.set_title(label + " (EMPTY)"); return
    ax.scatter(P[:,0], P[:,1], s=40, c=color, label=label, edgecolors="k", linewidths=0.5)
    if annotate:
        for i,(x,y) in enumerate(P):
            ax.text(x, y, str(i+1), fontsize=8, ha="center", va="bottom")
    ax.set_aspect("equal", adjustable="box"); ax.grid(True, ls=":"); ax.legend(loc="best")
    ax.margins(x=0.5, y=0.5)

def visualize_triplet(out_png: str,
                      P1: np.ndarray, N1: int, T1: int,
                      P2: np.ndarray, N2: int, T2: int,
                      pred_j: int, probs: np.ndarray,
                      annotate_index: bool = False):
    M = P2.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax1, ax2, ax3 = axes

    _scatter(ax1, P1, color="#1f77b4", label=f"目指（N={N1}）", annotate=annotate_index)
    if 0 <= T1 < N1: ax1.scatter(P1[T1,0], P1[T1,1], s=160, facecolors="none", edgecolors="r", linewidths=2.0, label=f"查询 T={T1+1}")
    ax1.legend(loc="best"); ax1.set_title("左：目指（圈出查询）")

    _scatter(ax2, P2, color="#2ca02c", label=f"测量（N={N2}）", annotate=annotate_index)
    if 0 <= T2 < N2: ax2.scatter(P2[T2,0], P2[T2,1], s=160, facecolors="none", edgecolors="b", linewidths=2.0, label=f"GT={T2+1}")
    if 0 <= pred_j < N2: ax2.scatter(P2[pred_j,0], P2[pred_j,1], s=160, facecolors="none", edgecolors="r", linewidths=2.0, label=f"Pred={pred_j+1}")
    ax2.legend(loc="best"); ax2.set_title("中：测量（蓝=GT 红=Pred）")

    if M == 0 or probs.size == 0:
        ax3.text(0.5, 0.5, "测量集合为空", ha="center", va="center", transform=ax3.transAxes); ax3.set_axis_off()
    else:
        x = np.arange(1, M+1)
        stem_out = ax3.stem(x, probs)  # 不使用 use_line_collection
        try:
            markerline, stemlines, baseline = stem_out
            plt.setp(stemlines, linewidth=1.0); plt.setp(markerline, markersize=4)
        except Exception:
            pass
        ax3.set_xlabel("Y 索引（1-based）"); ax3.set_ylabel("置信度"); ax3.set_title("右：置信度分布"); ax3.grid(True, ls=":")

    plt.tight_layout(); os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160); plt.close()

# ===== 新模型 =====
from model_traditional_v3 import predict_query_match

def main(data_dir: str = "Data/data",
         out_dir: str = "Traditional Method/visualization_traditional_v3",
         max_files: int = 1000,
         annotate_index: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
    if not files:
        print(f"[错误] 数据目录 {data_dir} 内未找到 .txt 文件。"); return
    files.sort(key=natural_key)

    total = correct = skipped = 0

    # —— 关键参数：强化“全局/局部形状不变”的权重与假设 —— #
    nbins_r, nbins_theta = 5, 12
    k_sig = 8
    # angle 基本不变 → rotation_grid 只用 0°；若想小范围微调可设 (-5,0,5)
    rotation_grid_deg = (0.0,)
    # 权重（全局位置 + 定向SC 权重大）
    w_global_pos, w_sc, w_sig, w_centroid, w_quant = 1.30, 1.20, 0.60, 0.80, 0.80
    tau = 0.35

    for i, fname in enumerate(files, start=1):
        if i > max_files: break
        path = os.path.join(data_dir, fname)
        base = os.path.splitext(fname)[0]
        out_png = os.path.join(out_dir, f"{i:04d}_{base}.png")

        try:
            (N1, T1, P1), (N2, T2, P2) = parse_full_txt(path)
            pred_j, probs, _dbg = predict_query_match(
                P1, T1, P2,
                nbins_r=nbins_r, nbins_theta=nbins_theta,
                k_sig=k_sig,
                w_global_pos=w_global_pos,
                w_sc=w_sc, w_sig=w_sig,
                w_centroid=w_centroid, w_quant=w_quant,
                rotation_grid_deg=rotation_grid_deg,
                tau=tau
            )
            visualize_triplet(out_png, P1,N1,T1, P2,N2,T2, pred_j, probs, annotate_index)

            if 0 <= T2 < N2 and N2 > 0 and pred_j >= 0:
                total += 1
                if pred_j == T2: correct += 1
            else:
                skipped += 1
        except Exception as e:
            print(f"[跳过] {fname}: {e}"); skipped += 1

    if total > 0:
        acc = correct / total
        print(f"\n[统计] 参与统计: {total}（跳过 {skipped}）")
        print(f"[统计] 前 {min(max_files, len(files))} 个样本 Top-1 准确率：{acc:.3f}")
    else:
        print(f"\n[统计] 没有有效样本参与统计（跳过 {skipped}）。")

if __name__ == "__main__":
    main()
