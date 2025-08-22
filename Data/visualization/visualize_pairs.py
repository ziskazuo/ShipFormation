# visualize_pairs.py
from __future__ import annotations
import os
import re
from typing import Tuple, List
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# （可选）中文字体兜底
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

def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def parse_full_txt(path: str):
    """
    完整解析单个样本文件。
    返回：
      (N1, T1, X1[N1,2]), (N2, T2, X2[N2,2])
    其中 T1/T2 都转换为 0-based 索引；若无效则为 -1。
    """
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip() != ""]
    if len(lines) < 6:
        raise ValueError(f"{path} 行数不足 6，实际 {len(lines)}")

    # --- 目指 ---
    N1_T1 = lines[0].split()
    if len(N1_T1) < 2:
        raise ValueError(f"{path} 第1行应包含 N 和 T 两个数")
    N1 = int(float(N1_T1[0])); T1_raw = int(float(N1_T1[1]))
    xs1 = list(map(float, lines[1].split()))
    ys1 = list(map(float, lines[2].split()))
    if len(xs1) < N1 or len(ys1) < N1:
        raise ValueError(f"{path} 目指 X/Y 数量不足: 期待 {N1}，实际 {len(xs1)}/{len(ys1)}")
    P1 = np.stack([np.array(xs1[:N1], dtype=float), np.array(ys1[:N1], dtype=float)], axis=1)

    # --- 测量 ---
    N2_T2 = lines[3].split()
    if len(N2_T2) < 2:
        raise ValueError(f"{path} 第4行应包含 N 和 T 两个数")
    N2 = int(float(N2_T2[0])); T2_raw = int(float(N2_T2[1]))
    xs2 = list(map(float, lines[4].split()))
    ys2 = list(map(float, lines[5].split()))
    if len(xs2) < N2 or len(ys2) < N2:
        raise ValueError(f"{path} 测量 X/Y 数量不足: 期待 {N2}，实际 {len(xs2)}/{len(ys2)}")
    P2 = np.stack([np.array(xs2[:N2], dtype=float), np.array(ys2[:N2], dtype=float)], axis=1)

    # 1-based -> 0-based，越界则置为 -1
    T1 = T1_raw - 1 if 1 <= T1_raw <= N1 else -1
    T2 = T2_raw - 1 if 1 <= T2_raw <= N2 else -1

    return (N1, T1, P1), (N2, T2, P2)

def _scatter(ax, P: np.ndarray, label: str, annotate: bool = False):
    ax.scatter(P[:, 0], P[:, 1], s=40, c="#1f77b4", label=label, edgecolors="k", linewidths=0.5)
    if annotate:
        for i, (x, y) in enumerate(P):
            ax.text(x, y, str(i+1), fontsize=8, ha="center", va="bottom")  # 1-based 显示
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, ls=":")
    ax.legend(loc="best")

def visualize_one(path: str, out_png: str, annotate_index: bool = False):
    (N1, T1, P1), (N2, T2, P2) = parse_full_txt(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax1, ax2 = axes

    # 左：目指
    _scatter(ax1, P1, label=f"目指（N={N1}）", annotate=annotate_index)
    if 0 <= T1 < N1:
        ax1.scatter(P1[T1, 0], P1[T1, 1], s=160, facecolors="none", edgecolors="r",
                    linewidths=2.0, label=f"查询目标 T（索引={T1+1}）")
        ax1.legend(loc="best")
    ax1.set_title("左：目指信息（圈出要查询的目标）")

    # 右：测量
    _scatter(ax2, P2, label=f"测量（N={N2}）", annotate=annotate_index)
    if 0 <= T2 < N2:
        ax2.scatter(P2[T2, 0], P2[T2, 1], s=160, facecolors="none", edgecolors="r",
                    linewidths=2.0, label=f"真实值标签（索引={T2+1}）")
        ax2.legend(loc="best")
    ax2.set_title("右：测量信息（圈出真实值标签）")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[保存] {out_png}")

def main(data_dir: str = "Data/data", out_dir: str = "Data/visualization", annotate_index: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
    if not files:
        print(f"[警告] 数据目录 {data_dir} 内未找到 .txt 文件。")
        return
    files.sort(key=natural_key)

    for fname in files:
        in_path = os.path.join(data_dir, fname)
        base = os.path.splitext(fname)[0]
        out_png = os.path.join(out_dir, f"{base}.png")
        try:
            visualize_one(in_path, out_png, annotate_index=annotate_index)
        except Exception as e:
            print(f"[跳过] {fname}: {e}")

    print(f"[完成] 可视化已输出到文件夹：{out_dir}")

if __name__ == "__main__":
    # annotate_index=True 时会在点旁标注 1..N 的序号，便于人工核对
    main(annotate_index=False)
