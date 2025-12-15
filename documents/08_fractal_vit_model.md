# 第八章：完整 ViT 模型 (fractal_vit.py)

本章详尽描述了数据在 `NextGenerationFractalViT` 模型中的完整流动过程。

## 8.1 数学形式化

$$I \xrightarrow{\text{Tokenize}} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{CLS}} [c; T'] \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$$

**损失函数**:
$$\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y}) + \lambda \cdot \mathcal{L}_{aux}$$

- Legacy Tokenizer: $\mathcal{L}_{aux} = \mathcal{L}_{REINFORCE}$ (策略梯度)
- Streaming Tokenizer: $\mathcal{L}_{aux} = 0$ (端到端可微)

---

## 8.2 Tokenizer 类型选项

| tokenizer_type | 实现类 | 特点 | 状态 |
| :--- | :--- | :--- | :--- |
| `legacy` | `FractalHilbertTokenizer` | BFS + REINFORCE | ⚠️ 废弃 |
| `streaming` | `StreamingFractalTokenizer` | 固定多尺度 | ✅ 稳定 |
| `streaming_v2` | `StreamingFractalTokenizerV2` | Gumbel-Softmax | ✅ 推荐 |

---

## 8.3 核心类：NextGenerationFractalViT

### 初始化参数

| 参数 | 类型 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- |
| `image_size` | int | - | 输入图像尺寸 |
| `num_classes` | int | - | 分类类别数 |
| `dim` | int | - | 模型维度 |
| `depth` | int | 6 | Transformer 层数 |
| `heads` | int | 8 | 注意力头数 |
| `mlp_dim` | int | - | FFN 隐藏层维度 |
| `pool` | str | 'cls' | 池化策略 |
| `dropout` | float | 0.1 | Dropout 比率 |
| `tokenizer_type` | str | 'streaming_v2' | Tokenizer 类型 |
| `bias_mode` | str | 'low_rank' | Hilbert Bias 模式 |
| `ffn_type` | str | 'swiglu_level' | FFN 类型 |

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

## 8.4 SimpleFractalViT

简化版模型，用于快速实验和调试。

### 与 NextGenerationFractalViT 的区别

| 特性 | NextGeneration | Simple |
| :--- | :--- | :--- |
| 层级自适应 | ✅ | ❌ |
| Hilbert Bias | ✅ 可选 | ❌ |
| 动态深度 | ✅ 可选 | ❌ |
| 全局注意力 | ✅ | ❌ |
| 参数量 | 较大 | 较小 |

---

## 8.5 辅助方法

### get_tokenizer_loss() (仅 Legacy Tokenizer)

用于训练阶段的策略更新。

**输入**: `reward` (通常是 `-CrossEntropyLoss`)

**计算流程**:
1. **堆叠**: 将所有 Log Probs 堆叠为张量
2. **优势计算**: `advantage = reward - baseline`
3. **策略梯度**: `policy_loss = -advantage * log_probs.mean()`
4. **熵正则化**: `loss -= coef * entropy`

**输出**: `aux_loss` (标量)

### clear_tokenizer_cache()

清空 tokenizer 的缓存，防止内存泄漏。

---

## 8.6 使用示例

```python
from vit_pytorch import NextGenerationFractalViT

model = NextGenerationFractalViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    depth=6,
    heads=6,
    mlp_dim=768,
    tokenizer_type='streaming_v2',
    bias_mode='low_rank',
    ffn_type='swiglu_level',
)

images = torch.randn(2, 3, 224, 224)
logits = model(images)  # (2, 1000)
```

---

## 8.7 架构图

```
Input Image (B, C, H, W)
        │
        ▼
┌─────────────────────────────┐
│ StreamingFractalTokenizerV2 │
│ - MultiScalePatchEncoder    │
│ - Gumbel-Softmax Selection  │
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
│ AdvancedFractalPosition     │
│ Embedding                   │
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
│ EnhancedFractalTransformer  │
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
