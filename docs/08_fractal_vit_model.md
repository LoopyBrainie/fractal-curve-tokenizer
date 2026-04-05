# Chapter 8: Complete ViT Model

## 8.1 Overview

`FractalCurveViT` is the complete Vision Transformer model that integrates all components: tokenization, position encoding, transformer encoder, and classification head.

**Architecture Note**: The model uses a configurable splitter for token selection:

- **H1SS (Hilbert Splitter with Stable Selection)** - `HilbertOptimalSplitter` - **Recommended**
- **H-entmax** - `HilbertOrderedEntmaxSplitter` - Alternative with full gradient flow
- **GumbelTopKSplitter** - Legacy option with STE approximation

The tokenizer expects split results from the splitter, following the I98-1 pipeline architecture.

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
        # Splitter configuration
        splitter_type: str = 'gumbel_topk',
        splitter_token_ratio_min: float = 0.02,
        splitter_token_ratio_max: float = 0.15,
        splitter_temp_start: float = 1.0,
        splitter_temp_end: float = 0.5,
        K_min_abs: int = 8,
        quota_learnable: bool = True,
        quota_entropy_weight: float = 0.5,
        # H1SS specific
        entmax_alpha: float = 1.2,
        tree_constraint_weight: float = 0.1,
        density_field_hidden_dim: int = 32,
        # Model configuration
        ffn_type: str = 'swiglu_level',
        use_checkpoint: bool = False,
        depth_scale_range: Tuple[float, float] = (0.5, 2.0),
        **kwargs,
    ):
        ...
```

### Key Parameters

| Parameter | Type | Default | Description |
|:----------|:-----|:--------|:------------|
| `image_size` | int / Tuple | None | Input image size (supports dynamic resolution) |
| `num_classes` | int | 1000 | Number of output classes |
| `dim` | int | 512 | Model embedding dimension |
| `num_layers` | int | 6 | Number of transformer layers |
| `heads` | int | 8 | Number of attention heads |
| `mlp_dim` | int | 1024 | FFN hidden dimension |
| `pool` | str | 'weighted' | Pooling strategy ('cls', 'mean', 'weighted') |
| `min_patch_size` | int | 4 | Minimum patch size |
| `drop_path_rate` | float | 0.0 | DropPath rate |
| `splitter_type` | str | 'gumbel_topk' | Splitter: `hilbert_optimal` (H1SS), `hilbert_entmax`, `gumbel_topk` |
| `K_min_abs` | int | 8 | Minimum absolute token count |
| `splitter_token_ratio_min` | float | 0.02 | Minimum token ratio |
| `splitter_token_ratio_max` | float | 0.15 | Maximum token ratio |
| `depth_scale_range` | tuple | (0.5, 2.0) | P6-1 depth scale range for sigmoid parameterization |
| `ffn_type` | str | 'swiglu_level' | FFN type |
| `quota_learnable` | bool | True | Enable learnable quota allocation |
| `entmax_alpha` | float | 1.2 | H1SS Entmax alpha parameter |
| `tree_constraint_weight` | float | 0.1 | Tree consistency constraint weight |

---

## 8.4 Forward Pass

### Step-by-Step Data Flow

```python
def forward(self, img: Tensor) -> Union[Tensor, TrainingStats]:
    """
    Args:
        img: (B, C, H, W) - Input images

    Returns:
        TrainingStats with:
        - logits: (B, num_classes) - Classification logits
        - num_tokens: int - Number of valid tokens
        - depth_used: int - Maximum depth actually used
        - depth_distribution: Tensor - Token count per depth level
        - features: Tensor - Transformer output features
        - transformer_tokens: Tensor - Token embeddings after transformer
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
│     HilbertOptimalSplitter      │
│     (H1SS - Recommended)       │
│  ┌────────────────────────────┐ │
│  │ HilbertConv1D Complexity   │ │
│  │ Entmax Sparse Selection      │ │
│  │ Tree Consistency Constraint │ │
│  └────────────────────────────┘ │
└─────────────────────────────────┘
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

### Basic Usage with H1SS (Recommended)

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
    splitter_type='hilbert_optimal',  # H1SS - Recommended
    ffn_type='swiglu_level',
    depth_scale_range=(0.5, 2.0),  # P6-1: sigmoid parameterization
)

images = torch.randn(4, 3, 224, 224)
logits = model(images)  # Returns TrainingStats with logits
```

### With H-entmax (Full Gradient Flow)

```python
# H-entmax provides 100% gradient coverage
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    splitter_type='hilbert_entmax',  # Full gradient flow
)
```

### With FractalConfig

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
