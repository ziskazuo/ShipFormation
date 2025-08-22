# plot_ship_counts.py
from __future__ import annotations
import os
import re
from typing import List, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# （可选）中文字体兜底，避免标题/标签中文乱码；没有也不报错
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
    """用于对文件名进行数字自然排序的 key。"""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def parse_sample_txt(path: str) -> Tuple[int, int]:
    """
    解析单个样本文件，只返回两侧的 N（目指N、测量N）。
    文件格式（忽略空行）:
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

    # 目指信息
    N1_T1 = lines[0].split()
    if len(N1_T1) < 2:
        raise ValueError(f"{path} 第1行应包含 N 和 T 两个数")
    N1 = int(float(N1_T1[0]))
    # T1 = int(float(N1_T1[1]))  # 此脚本不需要 T1

    xs1 = lines[1].split()
    ys1 = lines[2].split()
    if len(xs1) < N1 or len(ys1) < N1:
        raise ValueError(f"{path} 目指 X/Y 数量不足: 期待 {N1}，实际 {len(xs1)}/{len(ys1)}")

    # 测量信息
    N2_T2 = lines[3].split()
    if len(N2_T2) < 2:
        raise ValueError(f"{path} 第4行应包含 N 和 T 两个数")
    N2 = int(float(N2_T2[0]))
    # T2 = int(float(N2_T2[1]))  # 此脚本不需要 T2

    xs2 = lines[4].split()
    ys2 = lines[5].split()
    if len(xs2) < N2 or len(ys2) < N2:
        raise ValueError(f"{path} 测量 X/Y 数量不足: 期待 {N2}，实际 {len(xs2)}/{len(ys2)}")

    return N1, N2

def main(data_dir: str = "./Data/data", out_dir: str = "Data/visualization"):
    os.makedirs(out_dir, exist_ok=True)
    files = [f for f in os.listdir(data_dir) if f.lower().endswith(".txt")]
    if not files:
        print(f"[警告] 数据目录 {data_dir} 内未找到 .txt 文件。")
        return
    files.sort(key=natural_key)

    N1_list: List[int] = []
    N2_list: List[int] = []
    used_files: List[str] = []

    for i, fname in enumerate(files, start=1):
        if i > 1000:  # 题目要求横坐标 1-1000
            break
        path = os.path.join(data_dir, fname)
        try:
            n1, n2 = parse_sample_txt(path)
            N1_list.append(n1)
            N2_list.append(n2)
            used_files.append(fname)
        except Exception as e:
            print(f"[跳过] {fname}: {e}")

    if not N1_list:
        print("[错误] 有效文件数量为 0，无法绘图。")
        return

    x = np.arange(1, len(N1_list) + 1)
    plt.figure(figsize=(10, 5))
    plt.plot(x, N1_list, marker="o",markersize=1, linewidth=2, label="目指信息 Eye")
    plt.plot(x, N2_list, marker="s",markersize=0.5, linewidth=0.5, label="测量信息 Mea")
    plt.xlabel("文件序号")
    plt.ylabel("数量 N")
    plt.title("数据集各样本的数量统计")
    plt.grid(True, ls=":")
    plt.legend(loc="best")
    plt.tight_layout()

    out_png = os.path.join(out_dir, "ship_counts_overview.png")
    plt.savefig(out_png, dpi=160)
    plt.close()
    print(f"[完成] 折线图已保存：{out_png}")

if __name__ == "__main__":
    main()
