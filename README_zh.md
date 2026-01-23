# Fractal Curve ViT - 技术文档

[English](README.md) | [数学文档](README_math.md) | [中文](README_zh.md)

一个采用 **Hilbert 曲线分词** 和 **自适应多尺度 patch 选择** 的视觉 Transformer。Fractal ViT 根据图像内容复杂度动态分配 token——复杂区域分配更多 token，均匀背景分配更少 token。

## 批判性分析摘要

该实现提供了几种用于自适应视觉理解的创新机制。数学基础包括：用于 2D 到 1D 映射的 Hilbert 曲线空间填充（复杂度 O(log n)）、基于 Gumbel-Softmax 的可微分流选择（使用直通估计 STE）、以及基于最低公共祖先（LCA）的注意力偏置，通过层次结构编码空间关系。

**核心优势：**
- Hilbert 曲线实现正确保持了坐标转换的 O(log n) 复杂度
- Gumbel-Top-K 分割器并行评估所有候选区域，消除了串行依赖
- 基于 LCA 的注意力偏置将位置编码参数从 O(N²) 减少到 O(D×H)，其中 D 是最大深度，H 是头数
- 深度方差归一化在数学上解决了不同四叉树深度的方差不平衡问题

**已知局限性（透明记录）：**
- Hilbert 曲线局部性边界：文档中的边界 ||p1-p2||_2 ≤ C·|d1-d2|^{1/2} 表示上界，实际局部性保持取决于特定的曲线遍历顺序
- LCA 计算：基于路径的 LCA 使用 Hilbert 曲线索引，与四叉树结构 LCA 有良好的对应但不精确
- 梯度覆盖范围：当 K < N 时，子集 softmax 仅向选中的候选提供梯度
- 温度退火：T_end = 0.3 是实际权衡；过低的温度可能导致梯度饱和

## 架构概览

![架构对比](workspace/visualizations/architecture_comparison.png)

*固定网格 ViT 与自适应分形分词的根本区别。*

### 数据流程

```
图像 (B, C, H, W)
       │
       ▼
┌─────────────────────────────────────┐
│  StreamingFractalTokenizerV3        │
│  ├─ SharedConv: 特征提取            │
│  ├─ GumbelTopKSplitter: 自适应      │
│  ├─ ROI-Align: 变尺寸池化           │
│  └─ HilbertSort: 曲线排序           │
└─────────────────────────────────────┘
       │
       ▼ (T ∈ ℝ^{B×N×D}, L ∈ ℤ^{B×N×(1+depth)})
┌─────────────────────────────────────┐
│  FractalPositionEmbedding           │
│  ├─ 深度嵌入                        │
│  └─ 象限路径编码                    │
└─────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│  FractalTransformer (×L 层)         │
│  ├─ HilbertAwareMultiScaleAttention │
│  │  └─ 基于 LCA 的注意力偏置        │
│  └─ AdaptiveFractalFeedForward      │
│     └─ SwiGLU + 层级缩放            │
└─────────────────────────────────────┘
       │
       ▼
   [CLS] 池化 → MLP Head → Logits
```

## Hilbert 曲线排序

![Hilbert 曲线](workspace/visualizations/hilbert_curve.png)

*从架构工具生成的 Hilbert 曲线（阶数=4）。*

Hilbert 排序用保持 2D 局部性的空间填充曲线替换光栅扫描：

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

**关键性质：** 空间上相邻的 patch 在 1D 序列中也保持接近，提升局部注意力的有效性。

### 排序方式对比

![排序方式对比](workspace/visualizations/orderings_comparison.png)

*不同 patch 排序策略的对比：光栅扫描 vs Hilbert 曲线。*

### 局部性保持分析

![局部性保持](workspace/visualizations/locality_preservation.png)

*不同排序方法的量化局部性保持分析。*

## 自适应多尺度分词

### 混合深度区域

![混合深度区域](workspace/visualizations/mixed_depth_regions.png)

*自适应四叉树分词为复杂区域（边缘、纹理）分配更多 token，为均匀区域（天空、背景）分配更少 token。*

### Token 数量变异性

![Token 数量范围](workspace/visualizations/token_count_range.png)

*Token 数量根据图像复杂度自适应，范围从最小（均匀图像）到最大（复杂纹理）。*

**公式：**

$$N_{tokens} \in [K_{min}, K_{max}] \quad \text{其中} \quad K_{min}=8, K_{max}=64$$

### 多尺度表示

![多尺度表示](workspace/visualizations/multi_scale_representation.png)

*多尺度表示使模型能够同时捕获全局结构和局部细节。*

## 位置编码对比

![位置编码对比](workspace/visualizations/position_encoding_comparison.png)

*位置编码方法对比：可学习 N² 偏置 vs 基于 LCA 的层次编码。*

**基于 LCA 编码的优势：**
- O(D×H) 参数 vs 可学习偏置的 O(N²)
- 明确的几何意义：更深的 LCA = 更近的空间邻近性
- 层次结构自然编码尺度信息

### 深度嵌入相似性

![深度嵌入相似性](workspace/visualizations/depth_embedding_similarity.png)

*深度嵌入相似性矩阵，显示层次关系编码。*

## 技术规格

### 核心参数

| 参数 | 默认值 | 范围 | 描述 |
|------|--------|------|------|
| `dim` | 384 | 256-768 | 模型嵌入维度 |
| `depth` | 6 | 6-12 | Transformer 层数 |
| `heads` | 8 | 6-12 | 注意力头数 |
| `min_patch_size` | 4 | 4-16 | 最细 patch 粒度 |
| `K_min` | 8 | 4-16 | 最小 token 数量 |
| `K_max` | 64 | 32-256 | 最大 token 数量 |

### 计算复杂度

| 组件 | 时间 | 空间 | 备注 |
|------|------|------|------|
| 分词器 | O(B·N·D) | O(B·N·D) | N 自适应，平均 ~32 |
| 注意力 | O(B·H·N²·d) | O(B·H·N²) | N² 但 N << 固定 ViT |
| FFN | O(B·N·D·D_ff) | O(B·N·D_ff) | D_ff ≈ 4D |

### 内存占用

对于 224×224 图像，典型 N ≈ 32：
- 注意力矩阵：B × H × N² ≈ 1 × 8 × 1024 = 8K 元素
- 对比标准 ViT（N=196）：1 × 8 × 38416 = 307K 元素
- **内存减少：~40×** 用于注意力存储

## 使用示例

### 基础分类

```python
import torch
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    min_patch_size=4,
    K_min=8,
    K_max=64,
)

img = torch.randn(1, 3, 224, 224)
logits = model(img)  # (1, 1000)
```

### 高级选项

```python
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=512,
    depth=8,
    heads=8,
    mlp_dim=2048,
    min_patch_size=4,
    K_min=12,
    K_max=96,
    dropout=0.2,
    drop_path_rate=0.1,
    ffn_type='swiglu_level',
    lca_temperature=1.5,
    learnable_temperature=True,
    use_checkpoint=True,  # 梯度检查点
)
```

### 动态分辨率 (I78)

```python
# 支持任意输入尺寸
model = FractalCurveViT(
    image_size=None,  # 动态分辨率
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
)

# 同一批次中不同尺寸
img1 = torch.randn(1, 3, 224, 224)
img2 = torch.randn(1, 3, 256, 192)
logits = model(torch.cat([img1, img2], dim=0))
```

### 访问内部状态

```python
# 获取分词统计信息
analysis = model.analyze_tokenization(img)
# {
#     'batch_size': 1,
#     'per_image_stats': [{'num_tokens': 42, 'levels_used': [0, 1, 2, 3]}],
#     'overall_stats': {'avg_tokens_per_image': 42.0}
# }

# 获取 transformer tokens 用于困难样本挖掘
logits, tokens, lengths = model(img, return_tokens=True)
# tokens: [B, N, D] - 不含 CLS 的 transformer 输出
# lengths: [B] - 每个图像的有效 token 数量
```

## 训练配置

### 推荐训练命令

```bash
# CIFAR-10 快速验证
python src/training/train_fractal_vit.py --quick-test --use-amp

# Tiny-ImageNet 完整训练
python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet --epochs 100 \
    --dim 320 --depth 12 --heads 8 \
    --dropout 0.2 --drop-path 0.2 --weight-decay 0.1 \
    --use-amp --gradient-checkpoint --compile --channels-last

# 小数据集（减少过拟合）
python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet --epochs 150 \
    --dim 256 --depth 8 --heads 6 \
    --dropout 0.25 --drop-path 0.25 --freeze-tokenizer --use-amp
```

### Windows RTX 4070 优化

```powershell
.\src\training\train_tiny_imagenet_4070_optimal.ps1
.\src\training\train_cub200_4070_optimal.ps1
```

## 测试

```bash
# 完整测试套件
pytest tests/

# 跳过慢速测试
pytest -m "not slow"

# 核心分割器测试
pytest tests/test_gumbel_topk_splitter.py

# Scheme E 配额测试
pytest tests/test_i24_2_learnable_quota.py
```

## 数学参考

### Hilbert 曲线性质

Hilbert 曲线提供 1D 索引与 2D 坐标之间的双射：

$$d = xy\_to\_d(n, x, y) = \sum_{k=0}^{log_2(n)-1} 4^k \cdot ((3 \cdot rx_k) \oplus ry_k)$$

其中 $rx_k, ry_k$ 是 x 和 y 坐标的第 k 位。

### 基于 LCA 的注意力偏置

Token i 和 j 之间的注意力偏置计算为：

$$B[i,j] = \tau_h \cdot \text{LCAEmbed}(\text{LCA}(i,j))$$

其中 $\tau_h$ 是每头可学习的温度参数。

### Gumbel-Top-K 选择

可微分的流选择机制：

$$g_i \sim \text{Gumbel}(0, 1)$$
$$z_i = \text{logits}_i + g_i$$
$$\text{selected} = \text{TopK}(z_i / \tau, K)$$

### 深度方差归一化

解决跨深度的方差不平衡：

$$z_i^{\text{norm}} = \frac{z_i - \mu_d}{\sigma_d + \epsilon}$$

其中 $\mu_d, \sigma_d$ 是深度 d 的批次统计量。

## 相关工作

| 方法 | 分词方式 | 位置编码 | 注意力机制 |
|------|----------|----------|------------|
| **ViT** (Dosovitskiy 2020) | 固定 16×16 | 可学习 N² | 完整 N² |
| **Swin** (Liu 2021) | 固定网格 | 相对位置 | 移位窗口 |
| **Quadtree Attn** (Zhang 2022) | 自适应四叉树 | 相对位置 | 四叉树感知 |
| **Fractal ViT (本项目)** | 自适应 Hilbert | 基于 LCA | Hilbert 感知 |

## 安装

```bash
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer
pip install -e .  # 或: uv sync
```

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)。