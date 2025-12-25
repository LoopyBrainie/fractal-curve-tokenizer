# 第一章：项目概述与设计理念

## 1.1 核心创新点：分形曲线与 Vision Transformer 的结合

Fractal Curve Tokenizer 项目代表了 Vision Transformer (ViT) 架构的一次实验性探索。传统的 ViT 将图像视为固定的 16x16 或 32x32 网格，这种刚性的划分方式忽略了图像内容的内在复杂性差异。

本项目的核心创新在于引入了**分形几何**和**Hilbert 曲线**的概念：

1. **Hilbert 曲线遍历**：使用 Hilbert 曲线将 2D 网格映射为 1D 序列，保持空间局部性
2. **多尺度 tokenization**：通过 Variable Depth Tokens 实现内容自适应分割
3. **端到端可微**：使用自适应四叉树分割 + 区域池化实现可微的多尺度融合 (V3 推荐)

## 1.2 Hilbert 曲线简介与空间填充性质

在将 2D 的 Patch 转换为 1D 的 Token 序列输入 Transformer 时，传统的"光栅扫描"（Raster Scan，即逐行扫描）会破坏空间局部性。

本项目采用了 **Hilbert 曲线**（希尔伯特曲线）作为遍历路径：

* **空间局部性保持**：Hilbert 曲线是一种空间填充曲线，它能够极好地保持 2D 空间中的邻近关系
* **递归自相似性**：Hilbert 曲线的递归构造方式与分形结构完美契合
* **数学定义**：$H: [0, n^2) \leftrightarrow [0, n)^2$，双向映射

## 1.3 与传统 ViT 的架构对比

| 特性           | 传统 ViT       | Fractal Curve ViT (本项目) |
|:------------ |:------------ |:----------------------- |
| **分词方式**     | 固定网格         | 多尺度卷积金字塔                |
| **Token 数量** | 固定 (如 196 个) | 固定 (取决于最细粒度)            |
| **序列化顺序**    | 光栅扫描         | Hilbert 曲线遍历            |
| **位置编码**     | 绝对/相对位置编码    | 深度 + 路径编码               |
| **注意力偏置**    | 无/相对位置       | Low-Rank Hilbert Bias   |
| **FFN 类型**   | GELU         | SwiGLU (可选)             |
| **训练方式**     | 端到端          | 端到端可微 (无 REINFORCE)     |

## 1.4 项目目录结构说明

```text
fractal-curve-tokenizer/
├── src/
│   └── vit_pytorch/
│       ├── __init__.py             # 包入口，模块导出
│       ├── vit.py                  # [核心] FractalCurveViT 完整模型
│       ├── streaming_tokenizer.py  # [核心] 流式分形 Tokenizer (V1/V3)
│       ├── adaptive_split.py       # [核心] 自适应四叉树分割算法
│       ├── patch_embed.py          # [核心] Hilbert-Native Patch Embedding
│       ├── transformer.py          # 增强型 Transformer 编码器
│       ├── attention.py            # Hilbert 感知多尺度注意力 (LCA Bias)
│       ├── positional.py           # 分形位置编码
│       ├── feedforward.py          # SwiGLU / 自适应前馈网络
│       ├── hilbert.py              # Hilbert 曲线算法与缓存
│       ├── tokenization.py         # 基础数据结构与抽象基类
│       ├── features.py             # Token 特征计算
│       ├── constants.py            # [配置] 全局常量与超参数
│       ├── utils.py                # 通用工具函数
│       ├── fractal_config.py       # 统一配置管理
│       └── fractal_path.py         # 四叉树路径编码
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

## 1.5 Tokenizer 类型对照表

| tokenizer_type | 实现类                           | 特点                          | 状态   |
|:-------------- |:----------------------------- |:---------------------------- |:---- |
| `streaming`    | `StreamingFractalTokenizer`   | 固定多尺度卷积                    | ✅ 稳定 |
| `streaming_v3` | `StreamingFractalTokenizerV3` | Variable Depth Tokens + 自适应四叉树分割 | ✅ **推荐** |

> **注意**: V2 (Gumbel-Softmax) 已从代码库移除。
