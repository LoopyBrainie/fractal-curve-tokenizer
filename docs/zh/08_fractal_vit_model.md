# 第八章：完整 ViT 模型

## 8.1 概述

`FractalCurveViT` 是完整的 Vision Transformer 模型，整合了所有组件：分词、位置编码、Transformer 编码器和分类头。

**架构说明**：模型使用可配置的分割器进行 token 选择：

- **H1SS (Hilbert Splitter with Stable Selection)** - `HilbertOptimalSplitter` - **推荐**
- **H-entmax** - `HilbertOrderedEntmaxSplitter` - 替代方案，提供完整梯度流
- **GumbelTopKSplitter** - 遗留选项，使用 STE 近似

分词器期望从分割器获取分割结果，遵循 I98-1 管道架构。

---

## 8.2 数学形式化

### 端到端管道

$$I \xrightarrow{\text{Splitter}} \text{split\_result} \xrightarrow{\text{Tokenizer}} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{CLS}} [c; T'] \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$$

### 返回格式

`forward()` 方法返回分类 logits，可选地包含辅助信息：

$$\text{forward}(I) \rightarrow (\hat{y}, \text{aux\_infos})$$

其中 $\text{aux\_infos}$ 包含：
- `num_tokens`：每样本的有效 token 数量
- `levels_used`：实际使用的最大深度

### 损失函数

$$\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y})$$

流式分词器支持端到端可微性，无需辅助损失。

---

## 8.3 类定义

```python
class FractalCurveViT(nn.Module):
    def __init__(
        self,
        *,
        image_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_classes: int = 1000,
        dim: int = 512,
        num_layers: int = 6,
        heads: int = 8,
        mlp_dim: int = 1024,
        pool: str = 'weighted',
        channels: int = 3,
        dim_head: int = 64,
        min_patch_size: int = 4,
        tokenizer_dropout: float = 0.0,
        transformer_dropout: float = 0.0,
        emb_dropout: float = 0.0,
        drop_path_rate: float = 0.0,
        # 分割器配置
        splitter_type: str = 'gumbel_topk',
        splitter_token_ratio_min: float = 0.02,
        splitter_token_ratio_max: float = 0.15,
        splitter_temp_start: float = 1.0,
        splitter_temp_end: float = 0.5,
        K_min_abs: int = 8,
        quota_learnable: bool = True,
        quota_entropy_weight: float = 0.5,
        # H1SS 特定参数
        entmax_alpha: float = 1.2,
        tree_constraint_weight: float = 0.1,
        density_field_hidden_dim: int = 32,
        # 模型配置
        ffn_type: str = 'swiglu_level',
        use_checkpoint: bool = False,
        depth_scale_range: Tuple[float, float] = (0.5, 2.0),
        **kwargs,
    ):
        ...
```

### 关键参数

| 参数 | 类型 | 默认值 | 描述 |
|:----------|:-----|:--------|:------------|
| `image_size` | int / Tuple | None | 输入图像大小（支持动态分辨率） |
| `num_classes` | int | 1000 | 输出类别数 |
| `dim` | int | 512 | 模型嵌入维度 |
| `num_layers` | int | 6 | Transformer 层数 |
| `heads` | int | 8 | 注意力头数 |
| `mlp_dim` | int | 1024 | FFN 隐藏维度 |
| `pool` | str | 'weighted' | 池化策略（'cls', 'mean', 'weighted'） |
| `min_patch_size` | int | 4 | 最小 patch 大小 |
| `drop_path_rate` | float | 0.0 | DropPath 率 |
| `splitter_type` | str | 'gumbel_topk' | 分割器：`hilbert_optimal` (H1SS)、`hilbert_entmax`、`gumbel_topk` |
| `K_min_abs` | int | 8 | 最小绝对 token 数量 |
| `splitter_token_ratio_min` | float | 0.02 | 最小 token 比例 |
| `splitter_token_ratio_max` | float | 0.15 | 最大 token 比例 |
| `depth_scale_range` | tuple | (0.5, 2.0) | P6-1 sigmoid 参数化的深度尺度范围 |
| `ffn_type` | str | 'swiglu_level' | FFN 类型 |
| `quota_learnable` | bool | True | 启用可学习配额分配 |
| `entmax_alpha` | float | 1.2 | H1SS Entmax alpha 参数 |
| `tree_constraint_weight` | float | 0.1 | 树一致性约束权重 |

---

## 8.4 前向传播

### 逐步数据流

```python
def forward(self, img: Tensor) -> Union[Tensor, TrainingStats]:
    """
    参数:
        img: (B, C, H, W) - 输入图像

    返回:
        TrainingStats，包含：
        - logits: (B, num_classes) - 分类 logits
        - num_tokens: int - 有效 token 数量
        - depth_used: int - 实际使用的最大深度
        - depth_distribution: Tensor - 每深度级别的 token 数量
        - features: Tensor - Transformer 输出特征
        - transformer_tokens: Tensor - Transformer 后的 token 嵌入
    """
```

### 步骤 1：分词

```python
token_output = self.tokenizer.tokenize(img)
# token_output.sequences: List[TokenSequence]
# - 每个序列: tokens (N_i, D), levels (N_i, max_depth+1)
```

### 步骤 2：批次填充

```python
# 提取 token 和深度
tokens_list = [seq.tokens for seq in token_output.sequences]
levels_list = [seq.get_levels() for seq in token_output.sequences]

# 填充到最大长度
padded_tokens = pad_sequence(tokens_list, batch_first=True)  # (B, N_max, D)
padded_levels = pad_sequence(levels_list, batch_first=True)  # (B, N_max, Info)

# 创建填充掩码
lengths = [seq.tokens.shape[0] for seq in token_output.sequences]
key_padding_mask = create_padding_mask(lengths)  # (B, N_max)
```

### 步骤 3：位置编码

```python
pos_emb = self.pos_embedding(padded_levels)  # (B, N_max, D)
x = padded_tokens + pos_emb
```

### 步骤 4：CLS Token

```python
B = x.shape[0]
cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
x = torch.cat([cls_tokens, x], dim=1)  # (B, N_max+1, D)

# 更新 levels_info 以包含 CLS
cls_levels = torch.zeros(B, 1, padded_levels.shape[-1], device=x.device, dtype=torch.long)
levels_info = torch.cat([cls_levels, padded_levels], dim=1)
```

### 步骤 5：Dropout

```python
x = self.dropout(x)
```

### 步骤 6：Transformer

```python
# 准备注意力掩码: True = 关注, False = 忽略
attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N+1)

x = self.transformer(x, levels_info, attn_mask)  # (B, N+1, D)
```

### 步骤 7：池化

```python
if self.pool == 'cls':
    pooled = x[:, 0]  # (B, D)
elif self.pool == 'mean':
    # 排除 CLS 和填充的 token
    mask = ~key_padding_mask  # (B, N)
    token_x = x[:, 1:]  # (B, N, D)
    pooled = (token_x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
```

### 步骤 8：分类头

```python
logits = self.mlp_head(pooled)  # (B, num_classes)

# mlp_head 结构:
# LayerNorm → Linear(D, D) → GELU → Dropout → Linear(D, num_classes)
```

---

## 8.5 辅助方法

### get_tokenizer_loss()

返回辅助分词器损失（流式分词器为零）。

```python
def get_tokenizer_loss(self, reward: Optional[Tensor] = None) -> Tensor:
    return torch.tensor(0.0, device=self.device)
```

### clear_tokenizer_cache()

清除分词器缓存以防止训练期间的内存泄漏。

```python
def clear_tokenizer_cache(self):
    if hasattr(self.tokenizer, 'clear_cache'):
        self.tokenizer.clear_cache()
```

---

## 8.6 架构图

```
输入图像 (B, C, H, W)
        │
        ▼
┌─────────────────────────────────┐
│     HilbertOptimalSplitter       │
│     (H1SS - 推荐)               │
│  ┌────────────────────────────┐ │
│  │ HilbertConv1D 复杂度        │ │
│  │ Entmax 稀疏选择              │ │
│  │ 树一致性约束                 │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  StreamingFractalTokenizerV3     │
│  ├── 复杂度估计                  │
│  ├── 自适应四叉树分割             │
│  ├── HilbertNativePatchEmbed     │
│  └── Hilbert 重排序              │
└─────────────────────────────────┘
        │
        ▼
   (tokens, levels_info)
        │
        ▼
┌─────────────────────────────────┐
│     FractalPositionEmbedding      │
│     深度 + 路径编码               │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     预置 CLS Token               │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     FractalTransformer × L        │
│  ┌────────────────────────────┐ │
│  │ 级别感知 LayerNorm          │ │
│  │ HilbertAwareAttention      │ │
│  │   + LCA Hilbert 偏置       │ │
│  │ DropPath + 残差            │ │
│  │ 级别感知 LayerNorm          │ │
│  │ SwiGLU FFN + 级别自适应    │ │
│  │ DropPath + 残差            │ │
│  └────────────────────────────┘ │
│     级别聚合器                   │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     池化（CLS / Mean）           │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     MLP 分类头                   │
│     LN → Linear → GELU → Linear │
└─────────────────────────────────┘
        │
        ▼
   Logits (B, num_classes)
```

---

## 8.7 使用示例

### 基本用法与 H1SS（推荐）

```python
from vit_pytorch import FractalCurveViT

# H1SS (Hilbert Splitter with Stable Selection)
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=512,
    num_layers=6,
    heads=8,
    mlp_dim=1024,
    splitter_type='hilbert_optimal',  # H1SS - 推荐
    ffn_type='swiglu_level',
    depth_scale_range=(0.5, 2.0),  # P6-1: sigmoid 参数化
)

images = torch.randn(4, 3, 224, 224)
logits = model(images)  # 返回包含 logits 的 TrainingStats
```

### 使用 H-entmax（完整梯度流）

```python
# H-entmax 提供 100% 梯度覆盖
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    splitter_type='hilbert_entmax',  # 完整梯度流
)
```

### 使用 FractalConfig

```python
from vit_pytorch import FractalConfig, FractalCurveViT

config = FractalConfig(
    d_model=512,
    num_heads=8,
    max_level=8,
)

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=config.d_model,
    heads=config.num_heads,
    config=config,
)
```

### 训练循环集成

```python
for images, labels in dataloader:
    with torch.cuda.amp.autocast():
        logits = model(images)
        loss = criterion(logits, labels)

    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # 清除缓存以防止内存累积
    model.clear_tokenizer_cache()
```

> **下一章**: [09_training_system.md](09_training_system.md) - 训练系统
