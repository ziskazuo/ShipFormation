# traditional_match_eval.py  (FIXED)
from __future__ import annotations
import os
import re
from typing import List, Tuple, Dict
import numpy as np

import matplotlib
matplotlib.use("Agg")  # 纯保存图片，无需交互
import matplotlib.pyplot as plt

# —— 中文字体兜底（可选，不存在也不报错）——
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
rcParams["axes.unicode_minus"] = False


# =============== 数据解析 ===============
def natural_key(s: str):
    """用于对文件名进行数字自然排序"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def parse_full_txt(path: str):
    """
    解析单个样本文件。
    返回:
      (N1, T1, P1[N1,2]), (N2, T2, P2[N2,2])
    其中 T1/T2 返回为 0-based；若越界则为 -1。
    文件格式（允许空行，会忽略）:
      L1: N1 T1
      L2: N1 个 X
      L3: N1 个 Y
      L4: N2 T2
      L5: N2 个 X
      L6: N2 个 Y
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if len(lines) < 6:
        raise ValueError(f"{path} 行数不足 6，实际 {len(lines)}")

    # 目指
    N1_T1 = lines[0].split()
    if len(N1_T1) < 2:
        raise ValueError(f"{path} 第1行应包含 N 和 T")
    N1 = int(float(N1_T1[0])); T1_raw = int(float(N1_T1[1]))
    xs1 = list(map(float, lines[1].split()))
    ys1 = list(map(float, lines[2].split()))
    if len(xs1) < N1 or len(ys1) < N1:
        raise ValueError(f"{path} 目指 X/Y 数量不足: 期待 {N1}, 实际 {len(xs1)}/{len(ys1)}")
    P1 = np.stack([np.array(xs1[:N1], dtype=float), np.array(ys1[:N1], dtype=float)], axis=1)

    # 测量
    N2_T2 = lines[3].split()
    if len(N2_T2) < 2:
        raise ValueError(f"{path} 第4行应包含 N 和 T")
    N2 = int(float(N2_T2[0])); T2_raw = int(float(N2_T2[1]))
    xs2 = list(map(float, lines[4].split()))
    ys2 = list(map(float, lines[5].split()))
    if len(xs2) < N2 or len(ys2) < N2:
        raise ValueError(f"{path} 测量 X/Y 数量不足: 期待 {N2}, 实际 {len(xs2)}/{len(ys2)}")
    P2 = np.stack([np.array(xs2[:N2], dtype=float), np.array(ys2[:N2], dtype=float)], axis=1)

    # 1-based -> 0-based，越界置 -1
    T1 = T1_raw - 1 if 1 <= T1_raw <= N1 else -1
    T2 = T2_raw - 1 if 1 <= T2_raw <= N2 else -1
    return (N1, T1, P1), (N2, T2, P2)


# =============== 传统匹配模型 ===============
def k_neigh_signature(P: np.ndarray, idx: int, k: int) -> np.ndarray:
    """
    距离签名：点 idx 到其它点的欧氏距离（去自身），取最小 k 个（升序）。
    为实现尺度稳健，返回向量将以最大值（最后一个）归一化。
    若可用邻居数 < k，则使用 k_eff 并用“最后一个值”右侧重复填充到长度 k。
    """
    N = P.shape[0]
    if N <= 1:
        return np.zeros((k,), dtype=np.float32)
    # 与所有点的距离
    d = np.sqrt(((P[idx][None, :] - P) ** 2).sum(axis=1) + 1e-12)
    d[idx] = np.inf
    k_eff = int(min(k, N - 1))
    d_small = np.partition(d, k_eff - 1)[:k_eff]
    d_small.sort()
    # 归一化（尺度稳健）
    denom = max(d_small[-1], 1e-6) if k_eff > 0 else 1.0
    sig_small = (d_small / denom).astype(np.float32)
    if k_eff < k:
        if k_eff == 0:
            return np.zeros((k,), dtype=np.float32)
        pad = np.full((k - k_eff,), sig_small[-1], dtype=np.float32)
        sig = np.concatenate([sig_small, pad], axis=0)
    else:
        sig = sig_small
    return sig


def predict_query_match(P1: np.ndarray, q_idx: int, P2: np.ndarray, k: int = 5) -> Tuple[int, np.ndarray]:
    """
    用距离签名近邻匹配：比较“目指中查询点”的签名与“测量中每个点”的签名（均取最小 k），
    使用 L2 距离作为不相似度；选最小者作为预测。
    返回：(pred_j, probs[M])，其中 probs 是按 softmax(-dist) 归一化的“置信度分布”。
    """
    N1 = P1.shape[0]; N2 = P2.shape[0]
    if N1 == 0 or N2 == 0 or q_idx < 0 or q_idx >= N1:
        return -1, np.zeros((N2,), dtype=np.float32)

    k_eff = int(min(k, N1 - 1, N2 - 1))
    if k_eff <= 0:
        # 极端情况：任一侧只有 1 个点，则只能任意匹配
        return (0 if N2 > 0 else -1), (np.ones((N2,), dtype=np.float32) / max(N2, 1) if N2 > 0 else np.zeros((0,), dtype=np.float32))

    sig_q = k_neigh_signature(P1, q_idx, k_eff)  # [k_eff]
    dists = np.zeros((N2,), dtype=np.float32)
    for j in range(N2):
        sig_j = k_neigh_signature(P2, j, k_eff)
        dists[j] = np.linalg.norm(sig_q - sig_j, ord=2)

    # softmax(-dist) 得到“置信度”
    d = dists - dists.min()  # 数值稳定
    scores = -d
    exp = np.exp(scores - scores.max())
    probs = exp / max(exp.sum(), 1e-12)
    pred_j = int(probs.argmax()) if N2 > 0 else -1
    return pred_j, probs


# =============== 可视化 ===============
def _scatter(ax, P: np.ndarray, color: str, label: str, annotate: bool = False):
    if P.size == 0:
        ax.set_title(label + " (EMPTY)")
        return
    ax.scatter(P[:, 0], P[:, 1], s=40, c=color, label=label, edgecolors="k", linewidths=0.5)
    if annotate:
        for i, (x, y) in enumerate(P):
            ax.text(x, y, str(i + 1), fontsize=8, ha="center", va="bottom")  # 1-based 展示
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

    # ✅ 新增：增加边距，防止点太靠近边界或重叠严重时被裁剪
    ax.margins(x=0.5, y=0.5)  # 可调，比如 0.02 ~ 0.1


def visualize_triplet(
    out_png: str,
    P1: np.ndarray, N1: int, T1: int,
    P2: np.ndarray, N2: int, T2: int,
    pred_j: int, probs: np.ndarray,
    annotate_index: bool = False
):
    """
    三联图：
      左：目指出点，圈出查询 T1
      中：测量出点，蓝圈=GT(T2), 红圈=Pred
      右：查询行的“置信度分布”（softmax(-dist)）
    """
    M = P2.shape[0]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax1, ax2, ax3 = axes

    # 左：目指信息
    _scatter(ax1, P1, color="#1f77b4", label=f"目指（N={N1}）", annotate=annotate_index)
    if 0 <= T1 < N1 and N1 > 0:
        ax1.scatter(P1[T1, 0], P1[T1, 1], s=160, facecolors="none", edgecolors="r", linewidths=2.0,
                    label=f"查询 T（索引={T1+1}）")
        ax1.legend(loc="best")
    ax1.set_title("左：目指信息（圈出查询目标）")

    # 中：测量信息
    _scatter(ax2, P2, color="#2ca02c", label=f"测量（N={N2}）", annotate=annotate_index)
    if 0 <= T2 < N2 and N2 > 0:
        ax2.scatter(P2[T2, 0], P2[T2, 1], s=160, facecolors="none", edgecolors="b", linewidths=2.0,
                    label=f"GT（索引={T2+1}）")
    if 0 <= pred_j < N2 and N2 > 0:
        ax2.scatter(P2[pred_j, 0], P2[pred_j, 1], s=160, facecolors="none", edgecolors="r", linewidths=2.0,
                    label=f"Pred（索引={pred_j+1}）")
    ax2.legend(loc="best")
    ax2.set_title("中：测量信息（蓝=GT，红=Pred）")

    # 右：置信度分布（兼容旧版 Matplotlib：不使用 use_line_collection）
    if M == 0 or probs.size == 0:
        ax3.text(0.5, 0.5, "测量集合为空", ha="center", va="center", transform=ax3.transAxes)
        ax3.set_title("右：查询对应的置信度分布")
        ax3.set_axis_off()
    else:
        x = np.arange(1, M + 1)
        stem_out = ax3.stem(x, probs)  # 关键修复：去掉 use_line_collection
        # 不同 Matplotlib 版本返回值形式略有差异，做一下兼容美化
        try:
            markerline, stemlines, baseline = stem_out
            plt.setp(stemlines, linewidth=1.0)
            plt.setp(markerline, markersize=4)
        except Exception:
            pass
        ax3.set_xlabel("Y 索引（1-based）")
        ax3.set_ylabel("置信度")
        ax3.set_title("右：查询对应的置信度分布")
        ax3.grid(True, ls=":")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


# =============== 主流程 ===============
def main(
    data_dir: str = "Data/minidata",
    out_dir: str = "Traditional Method/mini_visualization_traditional_v1",
    k: int = 5,
    annotate_index: bool = False,
    max_files: int = 1000
):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
    if not files:
        print(f"[错误] 数据目录 {data_dir} 内未找到 .txt 文件。")
        return

    files.sort(key=natural_key)
    total = 0
    correct = 0
    skipped = 0

    for i, fname in enumerate(files, start=1):
        if i > max_files:
            break
        in_path = os.path.join(data_dir, fname)
        base = os.path.splitext(fname)[0]
        out_png = os.path.join(out_dir, f"{i:04d}_{base}.png")

        try:
            (N1, T1, P1), (N2, T2, P2) = parse_full_txt(in_path)
            pred_j, probs = predict_query_match(P1, T1, P2, k=k)

            visualize_triplet(
                out_png=out_png,
                P1=P1, N1=N1, T1=T1,
                P2=P2, N2=N2, T2=T2,
                pred_j=pred_j, probs=probs,
                annotate_index=annotate_index
            )

            # 统计准确率（仅当 GT 有效时参与统计）
            if 0 <= T2 < N2 and N2 > 0 and pred_j >= 0:
                total += 1
                if pred_j == T2:
                    correct += 1
            else:
                skipped += 1

        except Exception as e:
            print(f"[跳过] {fname}: {e}")
            skipped += 1

    if total > 0:
        acc = correct / total
        print(f"\n[统计] 参与统计的样本数: {total}（跳过 {skipped}）")
        print(f"[统计] 前 {min(max_files, len(files))} 个样本的 Top-1 准确率: {acc:.3f}")
    else:
        print(f"\n[统计] 没有有效样本参与统计（跳过 {skipped}）。")

if __name__ == "__main__":
    # annotate_index=True 时在点旁标注 1..N 的序号，便于人工核对
    main(annotate_index=False)
