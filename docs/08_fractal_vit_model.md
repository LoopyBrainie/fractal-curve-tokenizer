# Chapter 8: Complete ViT Model

## 8.1 Overview

`FractalCurveViT` is the complete Vision Transformer model that integrates all components: tokenization, position encoding, transformer encoder, and classification head.

**Architecture Note**: The model internally creates a `GumbelTopKSplitter` (Scheme D/E) for token selection. The tokenizer expects split results from the splitter, following the I98-1 pipeline architecture.

---

## 8.2 Mathematical Formulation

### End-to-End Pipeline

$$I \xrightarrow{\text{Splitter}} \text{split\_result} \xrightarrow{\text{Tokenizer}} (T, L) \xrightarrow{E_{pos}} T' \xrightarrow{\text{CLS}} [c; T'] \xrightarrow{\text{Transformer}} X' \xrightarrow{\text{Pool}} z \xrightarrow{\text{MLP}} \hat{y}$$

### Return Format

The `forward()` method returns classification logits with optional auxiliary information:

$$\text{forward}(I) \rightarrow (\hat{y}, \text{aux\_infos})$$

where $\text{aux\_infos}$ contains:
- `num_tokens`: Number of valid tokens per sample
- `levels_used`: Maximum depth actually used

### Loss Function

$$\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y})$$

The streaming tokenizer enables end-to-end differentiability without auxiliary losses.

---

## 8.3 Class Definition

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
        mlp_dim: int = None,  # Default: dim × 4
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

### Key Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `image_size` | int | - | Input image size |
| `num_classes` | int | - | Number of output classes |
| `dim` | int | 512 | Model embedding dimension |
| `depth` | int | 6 | Number of transformer layers |
| `heads` | int | 8 | Number of attention heads |
| `mlp_dim` | int | dim × 4 | FFN hidden dimension |
| `pool` | str | 'cls' | Pooling strategy ('cls' or 'mean') |
| `max_level` | int | None | Max recursion level (auto-inferred) |
| `dropout` | float | 0.1 | Dropout rate |
| `tokenizer_type` | str | 'streaming_v3' | Tokenizer type (V3 only) |
| `hilbert_bias_mode` | str | 'lca' | Attention bias mode (only 'lca' supported) |
| `ffn_type` | str | 'swiglu_level' | FFN type |

---

## 8.4 Forward Pass

### Step-by-Step Data Flow

```python
def forward(self, img: Tensor) -> Tensor:
    """
    Args:
        img: (B, C, H, W) - Input images
    
    Returns:
        logits: (B, num_classes) - Classification logits
    """
```

### Step 1: Tokenization

```python
token_output = self.tokenizer.tokenize(img)
# token_output.sequences: List[TokenSequence]
# - each sequence: tokens (N_i, D), levels (N_i, max_depth+1)
```

### Step 2: Batch Padding

```python
# Extract tokens and levels
tokens_list = [seq.tokens for seq in token_output.sequences]
levels_list = [seq.get_levels() for seq in token_output.sequences]

# Pad to max length
padded_tokens = pad_sequence(tokens_list, batch_first=True)  # (B, N_max, D)
padded_levels = pad_sequence(levels_list, batch_first=True)  # (B, N_max, Info)

# Create padding mask
lengths = [seq.tokens.shape[0] for seq in token_output.sequences]
key_padding_mask = create_padding_mask(lengths)  # (B, N_max)
```

### Step 3: Position Encoding

```python
pos_emb = self.pos_embedding(padded_levels)  # (B, N_max, D)
x = padded_tokens + pos_emb
```

### Step 4: CLS Token

```python
B = x.shape[0]
cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
x = torch.cat([cls_tokens, x], dim=1)  # (B, N_max+1, D)

# Update levels_info for CLS
cls_levels = torch.zeros(B, 1, padded_levels.shape[-1], device=x.device, dtype=torch.long)
levels_info = torch.cat([cls_levels, padded_levels], dim=1)
```

### Step 5: Dropout

```python
x = self.dropout(x)
```

### Step 6: Transformer

```python
# Prepare attention mask: True = attend, False = ignore
attn_mask = ~key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N+1)

x = self.transformer(x, levels_info, attn_mask)  # (B, N+1, D)
```

### Step 7: Pooling

```python
if self.pool == 'cls':
    pooled = x[:, 0]  # (B, D)
elif self.pool == 'mean':
    # Exclude CLS and padded tokens
    mask = ~key_padding_mask  # (B, N)
    token_x = x[:, 1:]  # (B, N, D)
    pooled = (token_x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)
```

### Step 8: Classification Head

```python
logits = self.mlp_head(pooled)  # (B, num_classes)

# mlp_head structure:
# LayerNorm → Linear(D, D) → GELU → Dropout → Linear(D, num_classes)
```

---

## 8.5 Auxiliary Methods

### get_tokenizer_loss()

Returns auxiliary tokenizer loss (zero for streaming tokenizer).

```python
def get_tokenizer_loss(self, reward: Optional[Tensor] = None) -> Tensor:
    return torch.tensor(0.0, device=self.device)
```

### clear_tokenizer_cache()

Clears tokenizer cache to prevent memory leaks during training.

```python
def clear_tokenizer_cache(self):
    if hasattr(self.tokenizer, 'clear_cache'):
        self.tokenizer.clear_cache()
```

---

## 8.6 Architecture Diagram

```
Input Image (B, C, H, W)
        │
        ▼
┌─────────────────────────────────┐
│  StreamingFractalTokenizerV3    │
│  ├── Complexity Estimation      │
│  ├── Adaptive Quadtree Split    │
│  ├── HilbertNativePatchEmbed    │
│  └── Hilbert Reordering         │
└─────────────────────────────────┘
        │
        ▼
   (tokens, levels_info)
        │
        ▼
┌─────────────────────────────────┐
│     FractalPositionEmbedding    │
│     Depth + Path Encoding       │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     Prepend CLS Token           │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     FractalTransformer × L      │
│  ┌────────────────────────────┐ │
│  │ Level-Aware LayerNorm      │ │
│  │ HilbertAwareAttention      │ │
│  │   + LCA Hilbert Bias       │ │
│  │ DropPath + Residual        │ │
│  │ Level-Aware LayerNorm      │ │
│  │ SwiGLU FFN + Level Adapt   │ │
│  │ DropPath + Residual        │ │
│  └────────────────────────────┘ │
│     Level Aggregator            │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     Pooling (CLS / Mean)        │
└─────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│     MLP Classification Head     │
│     LN → Linear → GELU → Linear │
└─────────────────────────────────┘
        │
        ▼
   Logits (B, num_classes)
```

---

## 8.7 Usage Examples

### Basic Usage

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

### With FractalConfig

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

### Training Loop Integration

```python
for images, labels in dataloader:
    with torch.cuda.amp.autocast():
        logits = model(images)
        loss = criterion(logits, labels)
    
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Clear cache to prevent memory accumulation
    model.clear_tokenizer_cache()
```

> **Next**: [09_training_system.md](09_training_system.md) - Training System
