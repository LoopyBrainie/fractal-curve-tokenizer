# 第一章：项目概览与架构

## 1.1 项目简介
**Fractal Curve Tokenizer** 是一个实验性的视觉 Transformer 架构，旨在解决传统 ViT 固定 Patch 划分的局限性。它引入了基于图像内容复杂度的自适应分词机制。

*   **核心理念**: 图像中的平坦区域（如天空）可以用大 Patch 表示，而细节区域（如物体边缘）需要小 Patch。
*   **技术手段**: 结合 QuadTree（四叉树）递归分割和 Hilbert Curve（希尔伯特曲线）遍历，将 2D 图像转化为 1D 变长序列。

## 1.2 顶层数据流架构

```mermaid
graph TD
    A[原始图像 (B, C, H, W)] --> B(Fractal Tokenizer);
    B -->|递归分割 & 决策| C{Token 序列};
    C -->|变长序列| D[Token Processor];
    D -->|特征增强 & 投影| E[Batch Padding];
    E -->|填充对齐| F[Positional Embedding];
    F -->|注入分形位置信息| G[Transformer Encoder];
    G -->|层级感知注意力| H[Pooling & Head];
    H --> I[分类结果];
    
    subgraph "训练回路"
    I --> J[分类 Loss];
    J --> K[Reward 计算];
    K -->|REINFORCE| B;
    end
```

## 1.3 模块映射

| 模块 | 对应文件 | 核心功能 |
| :--- | :--- | :--- |
| **Tokenizer** | `src/vit_pytorch/fractal_curve_tokenizer.py` | 图像 -> 变长 Token 序列，包含策略网络。 |
| **Model** | `src/vit_pytorch/fractal_vit.py` | 组装整个模型，处理 Batch 对齐和 Loss 计算。 |
| **Transformer** | `src/vit_pytorch/transformer.py` | 定制的 Transformer 块，支持层级感知。 |
| **Attention** | `src/vit_pytorch/attention.py` | 希尔伯特感知注意力机制。 |
| **Embedding** | `src/vit_pytorch/positional.py` | 分形树位置编码。 |
| **Data Structures** | `src/vit_pytorch/tokenization.py` | 定义 `TokenSequence` 等基础数据结构。 |

## 1.4 阅读指南
建议按以下顺序阅读文档，以理解数据流向：
1.  **03_fractal_tokenizer.md**: 理解数据源头，如何从图像变成 Token。
2.  **08_fractal_vit_model.md**: 理解宏观架构，数据如何在各个模块间传递。
3.  **07_transformer_encoder.md**: 理解核心计算单元。
4.  **05_attention_mechanism.md**: 理解模型如何处理空间关系。
5.  **04_positional_embedding.md**: 理解位置信息的编码方式。
