# Fractal Curve ViT

[English](README.md) | [中文](README_zh.md)

采用 **Hilbert 曲线分词** 和 **自适应多尺度 patch 选择** 的视觉 Transformer。Fractal ViT 根据图像复杂度动态分配 token。

## 批判性分析摘要

**核心优势：**
- Hilbert 曲线保持 O(log n) 复杂度的坐标转换
- **HilbertOptimalSplitter**: 推荐默认选择，基于6条数学公理的最优区域选择
- **HilbertOrderedEntmaxSplitter**: 100% 梯度覆盖率（vs Gumbel-STE 的 37%）
- 基于 LCA 的注意力偏置将参数从 O(N²) 减少到 O(D×H)
- 深度方差归一化解决了四叉树深度的方差不平衡问题

**已知局限性：**
- Hilbert 局部性界是上界，实际保持取决于遍历顺序
- 基于路径的 LCA 使用 Hilbert 索引，与四叉树结构有良好但不精确的对应
- Gumbel-STE 梯度覆盖率限于 K 个选中 token（K=32, N=85 时约为 37.6%）
- 温度 T < 0.3 可能导致梯度饱和

## 架构

![架构对比](workspace/visualizations/architecture_comparison.png)

### 数据流程

```
图像 (B, C, H, W)
       │
       ▼
┌─────────────────────────────────────┐
│  StreamingFractalTokenizerV3        │
│  ├─ SharedConv: 特征提取            │
│  ├─ HilbertOptimalSplitter (推荐)   │
│  │   或 HilbertOrderedEntmaxSplitter│
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
│  └─ AdaptiveFractalFeedForward      │
└─────────────────────────────────────┘
       │
       ▼
   [CLS] 池化 → MLP Head → Logits
```

## Hilbert 曲线

![Hilbert 曲线](workspace/visualizations/hilbert_curve.png)

Hilbert 排序用保持 2D 局部性的空间填充曲线替换光栅扫描：

$$H: [0, n^2) \leftrightarrow [0, n) \times [0, n)$$

![排序对比](workspace/visualizations/orderings_comparison.png)

![局部性保持](workspace/visualizations/locality_preservation.png)

## 自适应分词

![混合深度区域](workspace/visualizations/mixed_depth_regions.png)

自适应四叉树分词为复杂区域（边缘、纹理）分配更多 token，为均匀区域（天空、背景）分配更少 token。

![Token 数量范围](workspace/visualizations/token_count_range.png)

$$N_{tokens} \in [K_{min}, K_{max}] \quad \text{其中} \quad K_{min}=8, K_{max}=64$$

![多尺度表示](workspace/visualizations/multi_scale_representation.png)

## 特征流形分析

![特征流形](workspace/visualizations/feature_manifold.png)

t-SNE/UMAP 可视化展示不同深度 token 的特征分布。

## Hilbert 注意力

![Hilbert 注意力图](workspace/visualizations/hilbert_attention_map.png)

注意力在 Hilbert 空间中的分布热图，展示局部集中模式。

## 效率分析

![效率分析](workspace/visualizations/efficiency_analysis.png)

![Pareto 前沿](workspace/visualizations/pareto_frontier.png)

![计算量对比](workspace/visualizations/computation_comparison.png)

## 位置编码

![位置编码对比](workspace/visualizations/position_encoding_comparison.png)

**基于 LCA 编码的优势：**
- O(D×H) 参数 vs O(N²) 的可学习偏置
- 更深的 LCA = 更近的空间邻近性
- 层次结构自然编码尺度信息

![深度嵌入相似性](workspace/visualizations/depth_embedding_similarity.png)

## 技术规格

### 核心参数

| 参数 | 默认值 | 范围 | 描述 |
|------|--------|------|------|
| `dim` | 384 | 256-768 | 嵌入维度 |
| `depth` | 6 | 6-12 | Transformer 层数 |
| `heads` | 8 | 6-12 | 注意力头数 |
| `min_patch_size` | 4 | 4-16 | 最细 patch 粒度 |
| `K_min` | 8 | 4-16 | 最少 token 数 |
| `K_max` | 64 | 32-256 | 最多 token 数 |

### 计算复杂度

| 组件 | 时间 | 空间 |
|------|------|------|
| 分词器 | O(B·N·D) | O(B·N·D) |
| 注意力 | O(B·H·N²·d) | O(B·H·N²) |
| FFN | O(B·N·D·D_ff) | O(B·N·D_ff) |

### 内存占用（N ≈ 32 tokens）

- 注意力矩阵：1 × 8 × 1024 = 8K 元素
- 标准 ViT（N=196）：1 × 8 × 38416 = 307K 元素
- **约 40× 减少**

## 使用方法

```python
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

### 动态分辨率

```python
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
# 分词统计
analysis = model.analyze_tokenization(img)
# {'num_tokens': 42, 'levels_used': [0, 1, 2, 3]}

# Transformer tokens
logits, tokens, lengths = model(img, return_tokens=True)
```

### 支持的分割器

| 分割器 | 适用场景 |
|:-------|:---------|
| **HilbertOptimalSplitter** | 推荐默认选择，基于6条数学公理 |
| HilbertOrderedEntmaxSplitter | 100% 梯度覆盖率（vs Gumbel-STE 的 37%） |

### 双路径模式

```python
# 启用双路径模式（V2 功能）
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    use_pattern_plugin=True,  # 启用双路径模式
)
```

## 训练

```bash
# 快速验证
uv run python src/training/train_fractal_vit.py --quick-test --use-amp

# Tiny-ImageNet
uv run python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet --epochs 100 \
    --dim 320 --depth 12 --heads 8 \
    --dropout 0.2 --drop-path 0.2 --weight-decay 0.1 \
    --use-amp --gradient-checkpoint --compile --channels-last
```

## 测试

```bash
uv run pytest tests/ -v                    # 完整测试套件
uv run pytest tests/unit/ -v               # 按层单元测试
uv run pytest tests/integration/ -v        # 集成测试
uv run pytest -m "not slow"                # 跳过慢速测试
```

## 数学核心

### Hilbert 曲线

$$d = xy\_to\_d(n, x, y) = \sum_{k=0}^{\log_2(n)-1} 4^k \cdot ((3 \cdot rx_k) \oplus ry_k)$$

### 基于 LCA 的注意力偏置

$$B[i,j] = \tau_h \cdot \text{LCAEmbed}(\text{LCA}(i,j))$$

### Gumbel-Top-K 选择

$$g_i \sim \text{Gumbel}(0, 1), \quad \text{selected} = \text{TopK}((\text{logits}_i + g_i) / \tau, K)$$

### 深度方差归一化

$$z_i^{\text{norm}} = \frac{z_i - \mu_d^{\text{EMA}}}{\sigma_d^{\text{EMA}} + \epsilon}$$

## 相关工作

| 方法 | 分词方式 | 位置编码 | 注意力机制 |
|------|----------|----------|------------|
| ViT (2020) | 固定 16×16 | 可学习 N² | 完整 N² |
| Swin (2021) | 固定网格 | 相对位置 | 移位窗口 |
| Quadtree Attn (2022) | 自适应四叉树 | 相对位置 | 四叉树感知 |
| **Fractal ViT** | **自适应 Hilbert** | **基于 LCA** | **Hilbert 感知** |

## 安装

```bash
git clone https://github.com/LoopyBrainie/fractal-curve-tokenizer.git
cd fractal-curve-tokenizer
uv sync
```

## 许可证

MIT 许可证 - 详见 [LICENSE](LICENSE)。
