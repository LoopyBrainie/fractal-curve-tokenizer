# 第一章：项目概述与设计理念

## 1.1 核心创新点：分形曲线与 Vision Transformer 的结合

Fractal Curve Tokenizer 项目代表了 Vision Transformer (ViT) 架构的一次重要演进。传统的 ViT 将图像视为固定的 16x16 或 32x32 网格，这种刚性的划分方式忽略了图像内容的内在复杂性差异。

本项目的核心创新在于引入了**分形几何**和**自适应分词**的概念：

1.  **自适应分辨率**：对于图像中平坦、信息量低的区域（如天空、背景），使用较大的 Patch；对于纹理复杂、边缘密集的区域（如物体边缘、细节），递归地将其分割为更小的 Patch。
2.  **可学习的分割策略**：分割过程不是基于固定的启发式规则，而是由一个轻量级的神经网络（MiniCNN + MLP）动态决策，并通过 REINFORCE 算法进行端到端优化。
3.  **分形结构**：这种递归分割自然形成了一种四叉树（QuadTree）结构，具有分形特征。

## 1.2 Hilbert 曲线简介与空间填充性质

在将 2D 的 Patch 转换为 1D 的 Token 序列输入 Transformer 时，传统的“光栅扫描”（Raster Scan，即逐行扫描）会破坏空间局部性。相邻的像素在 1D 序列中可能会相距甚远。

本项目采用了 **Hilbert 曲线**（希尔伯特曲线）作为遍历路径：

*   **空间局部性保持**：Hilbert 曲线是一种空间填充曲线（Space-filling curve），它能够极好地保持 2D 空间中的邻近关系。在 2D 空间相邻的区域，在映射到 1D 希尔伯特曲线上后，大概率仍然保持相邻。
*   **递归自相似性**：Hilbert 曲线的递归构造方式与我们的分形图像分割策略完美契合。我们在每一层级的分割中，都遵循 Hilbert 曲线的遍历顺序（“U”形路径），确保了多尺度下的空间连贯性。

## 1.3 与传统 ViT 的架构对比

| 特性 | 传统 ViT (Standard ViT) | Fractal Curve ViT (本项目) |
| :--- | :--- | :--- |
| **分词方式** | 固定网格 (Fixed Grid) | 自适应递归分割 (Adaptive Recursive Split) |
| **Token 数量** | 固定 (如 196 个) | 动态变化 (取决于图像复杂度) |
| **序列化顺序** | 光栅扫描 (Raster Scan) | Hilbert 曲线遍历 (Hilbert Curve Traversal) |
| **位置编码** | 绝对/相对位置编码 | 高级分形位置编码 (深度 + 路径历史) |
| **计算效率** | 对所有区域一视同仁 | 聚焦复杂区域，节省简单区域计算量 |
| **特征提取** | 线性投影 | 增强型特征提取 (含边缘、统计特征) |

## 1.4 项目目录结构说明

```text
fractal-curve-tokenizer/
├── src/
│   └── vit_pytorch/
│       ├── fractal_vit.py          # [核心] 完整模型定义 (NextGenerationFractalViT)
│       ├── fractal_curve_tokenizer.py # [核心] 分形分词器与 Hilbert 算法
│       ├── transformer.py          # 增强型 Transformer 编码器块
│       ├── attention.py            # Hilbert 感知多尺度注意力机制
│       ├── positional.py           # 高级分形位置编码
│       ├── feedforward.py          # 自适应前馈网络
│       ├── tokenization.py         # 基础数据结构与抽象基类
│       ├── features.py             # Token 特征计算工具
│       └── utils.py                # 通用工具函数
├── examples/
│   └── training/
│       └── train_fractal_vit.py    # 完整的训练脚本
├── tests/                          # 测试套件
│   ├── unit/                       # 单元测试
│   ├── integration/                # 集成测试
│   └── benchmarks/                 # 性能基准测试
├── experiments/                    # 实验输出 (Checkpoints, Logs)
└── documents/                      # 项目文档
```
