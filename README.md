
# Fleet Matching Prototype

这是一个可跑的原型，演示“在两个二维编队中指认同一条船”的深度学习实现思路（坐标/图方法）。

## 目录
- `geom_utils.py`：几何与点集 numpy 工具。
- `formations.py`：**参数化编队生成器**（含可视化主函数）。
- `ot_utils.py`：Sinkhorn 与 dustbin 等最优传输工具（PyTorch 实现）。
- `model.py`：轻量匹配模型（节点距离签名 + MLP + Sinkhorn）。
- `data.py`：将采样器封装成 PyTorch 友好接口。
- `train.py`：在合成数据上训练（仅对“查询节点那一行”做监督）。
- `infer_demo.py`：加载权重，做一次随机样本推理并可视化查询行的概率分布。

## 使用
1. 生成器可视化（会把图保存到 `/mnt/data/fleet_pair_preview.png`）：
   ```bash
   python formations.py
   ```

2. 训练：
   ```bash
   python train.py
   ```

3. 推理演示：
   ```bash
   python infer_demo.py
   ```

> 备注：若运行环境不能直接弹出图形窗口，`formations.py` 已使用 non-interactive backend 并将预览图直接保存为 PNG。
