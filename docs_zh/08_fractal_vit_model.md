# 第八章：完整 ViT 模型

## 8.1 概述

`FractalCurveViT` 是完整的 Vision Transformer 模型，整合了所有组件：分词、位置编码、Transformer 编码器和分类头。

**架构说明**：模型内部创建 `GumbelTopKSplitter`（方案 D/E）用于 token 选择。分词器期望从分隔器获取分割结果，遵循 I98-1 管道架构。

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
        image_size: int,
        num_classes: int,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        mlp_dim: int = None,  # 默认: dim × 4
        pool: str = 'cls',
        channels: int = 3,
        dim_head: int = 64,
        dropout: float = 0.1,
        emb_dropout: float = 0.0,
        tokenizer_type: str = 'streaming_v3',
        hilbert_bias_mode: str = 'lca',
        ffn_type: str = 'swiglu_level',
        low_rank_r: int = 32,
        lca_temperature: float = 1.5,
        learnable_temperature: bool = True,
    ):
        ...
```

### 关键参数

| 参数 | 类型 | 默认值 | 描述 |
|:----------|:-----|:--------|:------------|
| `image_size` | int | - | 输入图像大小 |
| `num_classes` | int | - | 输出类别数 |
| `dim` | int | 512 | 模型嵌入维度 |
| `depth` | int | 6 | Transformer 层数 |
| `heads` | int | 8 | 注意力头数 |
| `mlp_dim` | int | dim × 4 | FFN 隐藏维度 |
| `pool` | str | 'cls' | 池化策略（'cls' 或 'mean'） |
| `max_level` | int | None | 最大递归级别（自动推断） |
| `dropout` | float | 0.1 | dropout 率 |
| `tokenizer_type` | str | 'streaming_v3' | 分词器类型（仅 V3） |
| `hilbert_bias_mode` | str | 'lca' | 注意力偏置模式（仅支持 'lca'） |
| `ffn_type` | str | 'swiglu_level' | FFN 类型 |

---

## 8.4 前向传播

### 逐步数据流

```python
def forward(self, img: Tensor) -> Tensor:
    """
    参数:
        img: (B, C, H, W) - 输入图像

    返回:
        logits: (B, num_classes) - 分类 logits
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
│  StreamingFractalTokenizerV3    │
│  ├── 复杂度估计                  │
│  ├── 自适应四叉树分割            │
│  ├── HilbertNativePatchEmbed    │
│  └── Hilbert 重排序             │
└─────────────────────────────────┘
        │
        ▼
   (tokens, levels_info)
        │
        ▼
┌─────────────────────────────────┐
│     FractalPositionEmbedding    │
│     深度 + 路径编码             │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     预置 CLS Token              │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     FractalTransformer × L      │
│  ┌────────────────────────────┐ │
│  │ 级别感知 LayerNorm         │ │
│  │ HilbertAwareAttention      │ │
│  │   + LCA Hilbert 偏置       │ │
│  │ DropPath + 残差            │ │
│  │ 级别感知 LayerNorm         │ │
│  │ SwiGLU FFN + 级别自适应    │ │
│  │ DropPath + 残差            │ │
│  └────────────────────────────┘ │
│     级别聚合器                  │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     池化（CLS / Mean）           │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     MLP 分类头                  │
│     LN → Linear → GELU → Linear │
└─────────────────────────────────┘
        │
        ▼
   Logits (B, num_classes)
```

---

## 8.7 使用示例

### 基本用法

```python
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v3',
    hilbert_bias_mode='lca',
    ffn_type='swiglu_level',
)

images = torch.randn(4, 3, 224, 224)
logits = model(images)  # (4, 1000)
```

### 使用 FractalConfig

```python
from vit_pytorch import FractalConfig, FractalCurveViT

config = FractalConfig(
    d_model=384,
    num_heads=6,
    hilbert_bias_mode='lca',
    max_depth=4,
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
