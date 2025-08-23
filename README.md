
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
- `model_traditional_v1.py`：初始可运行demo
- `model_traditional_v2`: k近邻距离签名 + 最近邻”的简版传统模型
- `model_traditional_v3`: Shape Context（形状上下文）+ 旋转对齐（角度循环移位） 做强力的局部结构描述，再和原来的距离签名融合打分。
- `model_traditional_v4`:全局形状基本不变 → 用“全局刚体-尺度对齐（中心+稳健尺度，角度基本不变）”把查询点从目指系映射到测量系，形成全局几何残差作为强约束；局部形状基本不变 → 用“定向（不做角度循环）Shape Context + 最近邻距离序列”刻画局部几何，作为高权的重要项。
- 
## 结果
1. model_traditional_v3_1：
   ``
   [统计] 参与统计: 1000（跳过 0）
   [统计] 前 1000 个样本 Top-1 准确率：0.729
   ``
2. model_traditional_v3_1：
   ```
   [统计] 参与统计: 1000（跳过 0）
   [统计] 前 1000 个样本 Top-1 准确率：0.726
   ```

> 备注：若运行环境不能直接弹出图形窗口，`formations.py` 已使用 non-interactive backend 并将预览图直接保存为 PNG。
