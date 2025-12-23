# 第八章：完整 ViT 模型 (fractal_vit.py)

本章详尽描述了数据在 `FractalCurveViT` 模型中的完整流动过程。

## 8.1 数学形式化

$I \xrightarrow{\text{Tokenize}} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{CLS}} [c; T'] \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$

**损失函数**:
$$\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y})$$

Streaming Tokenizer 实现端到端可微，无需额外辅助损失。

---

## 8.2 Tokenizer 类型选项

| tokenizer_type | 实现类                           | 特点             | 状态   |
|:-------------- |:----------------------------- |:-------------- |:---- |
| `streaming`    | `StreamingFractalTokenizer`   | 固定多尺度          | ✅ 稳定 |
| `streaming_v2` | `StreamingFractalTokenizerV2` | Gumbel-Softmax | ⚠️ 废弃 |
| `streaming_v3` | `StreamingFractalTokenizerV3` | Cross-Scale Attention | ✅ **推荐** |

---

## 8.3 核心类：FractalCurveViT

### 初始化参数

| 参数               | 类型            | 默认值            | 说明                         |
|:---------------- |:------------- |:-------------- |:-------------------------- |
| `image_size`     | int           | -              | 输入图像尺寸                     |
| `num_classes`    | int           | -              | 分类类别数                      |
| `dim`            | int           | -              | 模型维度                       |
| `depth`          | int           | 6              | Transformer 层数             |
| `heads`          | int           | 8              | 注意力头数                      |
| `mlp_dim`        | int           | dim × 4        | FFN 隐藏层维度 (P0 修复: 2×→4×)   |
| `pool`           | str           | 'cls'          | 池化策略                       |
| `dropout`        | float         | 0.1            | Dropout 比率 (P0 修复: 0.3→0.1) |
| `tokenizer_type` | str           | 'streaming_v3' | Tokenizer 类型 (✅ V3 推荐)      |
| `variable_tokens`| bool          | False          | 可变 Token 模式 (P0 修复: True→False) |
| `bias_mode`      | str           | 'lca'          | Hilbert Bias 模式 (推荐 'lca') |
| `ffn_type`       | str           | 'swiglu_level' | FFN 类型                     |
| `config`         | FractalConfig | None           | 统一配置对象 (优先级高于单独参数)         |

### forward(img, ...) 数据流详解

**Step 1: 分形分词 (Tokenization)**

```python
token_output = self.tokenizer.tokenize(img)
# token_output: TokenizerOutput
# - sequences: List[TokenSequence]
# - 每个序列: tokens (N, D), levels (N, Info_Len)
```

**Step 2: 批次对齐 (Batch Padding)**

```python
# 提取 tokens 和 levels
tokens_list = [seq.tokens for seq in token_output.sequences]
levels_list = [seq.get_levels() for seq in token_output.sequences]

# Padding
padded_tokens = pad_sequence(tokens_list, batch_first=True)  # (B, S_max, D)
padded_levels = pad_sequence(levels_list, batch_first=True)  # (B, S_max, Info)

# 生成 mask
key_padding_mask = create_padding_mask(lengths)  # (B, S_max)
```

**Step 3: 位置编码 (Positional Embedding)**

```python
pos_emb = self.pos_embedding(padded_levels)  # (B, S_max, D)
x = padded_tokens + pos_emb
```

**Step 4: 添加 CLS Token**

```python
cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
x = torch.cat([cls_tokens, x], dim=1)  # (B, S_max + 1, D)

# 更新 mask
cls_levels = torch.zeros(B, 1, Info_Len)
levels_info = torch.cat([cls_levels, padded_levels], dim=1)
```

**Step 5: Dropout**

```python
x = self.dropout(x)
```

**Step 6: Transformer 编码**

```python
attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, S+1)
x = self.transformer(x, levels_info, attn_mask)  # (B, S+1, D)
```

**Step 7: 池化 (Pooling)**

```python
if self.pool == 'cls':
    pooled = x[:, 0]  # (B, D)
elif self.pool == 'mean':
    # 忽略 CLS 和 Padding
    mask = ~key_padding_mask[:, 1:]  # (B, S)
    pooled = (x[:, 1:] * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
```

**Step 8: 分类头 (Classification Head)**

```python
logits = self.mlp_head(pooled)  # (B, num_classes)
# mlp_head: LayerNorm -> Linear -> GELU -> Dropout -> Linear
```

---

## 8.4 辅助方法

### get_tokenizer_loss()

用于获取 tokenizer 的辅助损失。对于 Streaming Tokenizer，返回零损失。

**输入**: `reward` (可选参数，为了接口兼容性保留)

**输出**: `torch.tensor(0.0)` (对于 streaming tokenizer)

### clear_tokenizer_cache()

清空 tokenizer 的缓存，防止内存泄漏。

---

## 8.5 使用示例

### 推荐：使用 FractalConfig

```python
from vit_pytorch import FractalConfig, FractalCurveViT

# 使用统一配置对象（推荐）
config = FractalConfig(
    d_model=384,
    num_heads=6,
    hilbert_bias_mode='lca',  # 默认值，可省略
)

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=config.d_model,
    depth=6,
    heads=config.num_heads,
    mlp_dim=768,
    tokenizer_type='streaming_v2',
    config=config,
)

images = torch.randn(2, 3, 224, 224)
logits = model(images)  # (2, 1000)
```

### 直接参数传递

```python
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v3',  # ✅ 推荐
    bias_mode='lca',                # 推荐 (参数量最少)
    ffn_type='swiglu_level',        # 推荐
)

images = torch.randn(2, 3, 224, 224)
logits = model(images)  # (2, 1000)
```

---

## 8.6 架构图

```
Input Image (B, C, H, W)
        │
        ▼
┌─────────────────────────────┐
│ StreamingFractalTokenizerV3 │
│ - MultiScalePatchEncoder    │
│ - CrossScaleAttention       │
│ - HilbertIndexer            │
└─────────────────────────────┘
        │
        ▼
  TokenizerOutput
  (tokens, levels)
        │
        ▼
┌─────────────────────────────┐
│    Batch Padding & Mask     │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ FractalPositionEmbedding    │
│ - Depth Embedding           │
│ - Path Embedding            │
│ - Fusion Network            │
└─────────────────────────────┘
        │
        ▼
    Add CLS Token
        │
        ▼
┌─────────────────────────────┐
│ FractalTransformer          │
│ - Level-Aware LayerNorm     │
│ - HilbertAwareAttention     │
│ - SwiGLU FFN                │
│ - DropPath                  │
└─────────────────────────────┘
        │
        ▼
    Pooling (cls/mean)
        │
        ▼
┌─────────────────────────────┐
│       MLP Head              │
│ LN -> Linear -> GELU -> Linear
└─────────────────────────────┘
        │
        ▼
   Logits (B, num_classes)
```
