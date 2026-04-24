# Appendix

## A. Class Hierarchy

```mermaid
classDiagram
    nn_Module <|-- BaseTokenizer
    nn_Module <|-- FractalCurveViT

    BaseTokenizer <|-- StreamingFractalTokenizerV3

    FractalCurveViT *-- StreamingFractalTokenizerV3
    FractalCurveViT *-- FractalPositionEmbedding
    FractalCurveViT *-- FractalTransformer

    FractalTransformer *-- FractalTransformerBlock
    FractalTransformerBlock *-- HilbertAwareMultiScaleAttention
    FractalTransformerBlock *-- AdaptiveFractalFeedForward

    HilbertAwareMultiScaleAttention *-- LCAHilbertBias
    HilbertAwareMultiScaleAttention *-- LowRankHilbertBias
    AdaptiveFractalFeedForward *-- SwiGLUFFN
    
    StreamingFractalTokenizerV3 *-- HilbertNativePatchEmbed
    StreamingFractalTokenizerV3 *-- LearnableSplitter
    StreamingFractalTokenizerV3 *-- BalancedGreedySplitter
    
    LearnableSplitter *-- ComplexityMLP
    LearnableSplitter *-- TemperatureScheduler
```

---

## B. Data Flow Diagram

```
1. Input Image (B, C, H, W)
        │
        ▼
2. StreamingFractalTokenizerV3
   ├── GumbelTopKSplitter (Scheme D/E)
   │   └── ROI-Pool → MLP → Gumbel-Softmax → Top-K
   ├── HilbertNativePatchEmbed
   │   └── t = Pool(F[R]) · σ_d + E_d
   └── HilbertIndexer (Reordering)
        │
        ▼
3. TokenizerOutput
   ├── tokens: (B, N, D)
   └── levels: (B, N, max_depth+1)
        │
        ▼
4. Batch Padding
   ├── padded_tokens: (B, N_max, D)
   ├── padded_levels: (B, N_max, Info)
   └── mask: (B, N_max)
        │
        ▼
5. FractalPositionEmbedding
   └── E_pos = Fusion(E_depth(d) + E_path(p))
        │
        ▼
6. Add CLS Token → (B, N_max+1, D)
        │
        ▼
7. FractalTransformer × L
   ├── Level-Aware LayerNorm
   ├── HilbertAwareMultiScaleAttention
   │   ├── QKV Projection
   │   ├── Level Scaling
   │   ├── Hilbert Bias (LCA)
   │   └── Level Bias
   ├── DropPath + Residual
   ├── Level-Aware LayerNorm
   ├── AdaptiveFractalFeedForward (SwiGLU)
   └── DropPath + Residual
        │
        ▼
8. Level Aggregator
        │
        ▼
9. Pooling (CLS / Mean)
        │
        ▼
10. MLP Head
    └── LN → Linear → GELU → Linear
        │
        ▼
11. Logits (B, num_classes)
```

---

## C. Hyperparameter Reference

### Model Parameters

| Parameter | CIFAR-10 | Tiny-ImageNet | ImageNet |
|:----------|:---------|:--------------|:---------|
| `dim` | 192 | 256 | 512 |
| `depth` | 9 | 10 | 12 |
| `heads` | 6 | 8 | 8 |
| `mlp_dim` | 384 | 1024 | 2048 |
| `patch_size` | 4 | 4 | 16 |
| `max_depth` | 3 | 4 | 4 |

### Training Parameters

| Parameter | Default | Range |
|:----------|:--------|:------|
| `lr` | 5e-4 | [1e-4, 1e-3] |
| `weight_decay` | 0.03 | [0.01, 0.1] |
| `dropout` | 0.1 | [0.0, 0.2] |
| `drop_path` | 0.1 | [0.0, 0.2] |
| `warmup_epochs` | 5 | [3, 10] |
| `batch_size` | 128 | [64, 256] |

### Tokenizer Parameters

| Parameter | Default | Range | Description |
|:----------|:--------|:------|:------------|
| `alpha` | 0.5 | [0.3, 0.7] | Variance weight |
| `tau_0` | 0.15 | [0.08, 0.25] | Root threshold |
| `gamma` | 0.85 | [0.75, 0.92] | Decay factor |
| `sigma_0_sq` | 0.01 | [0.005, 0.03] | Variance normalization |
| `g_0_sq` | 0.08 | [0.03, 0.15] | Gradient normalization |

---

## D. API Quick Reference

### Model Initialization

```python
from vit_pytorch import FractalCurveViT, FractalConfig

# Method 1: Direct parameters
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    num_layers=12,
    heads=6,
    mlp_dim=768,
    ffn_type='swiglu_level',
)

# Method 2: Using FractalConfig
config = FractalConfig(image_size=224, min_patch_size=4)
model = FractalCurveViT(
    image_size=224,
    num_classes=1000,
    dim=384,
    heads=6,
)
```

### Forward Pass

```python
# Basic
logits = model(images)  # (B, num_classes)

# With auxiliary info
logits, aux = model(images, return_aux_info=True)
```

### Tokenizer Standalone

```python
from vit_pytorch import StreamingFractalTokenizerV3

tokenizer = StreamingFractalTokenizerV3(
    image_size=224,
    d_model=384,
    base_patch_size=4,
    max_depth=4,
    split_scheme='balanced_greedy',
)

output = tokenizer.tokenize(images)
for seq in output.sequences:
    print(f"Tokens: {seq.tokens.shape}")
    print(f"Levels: {seq.get_levels().shape}")
```

### Individual Components

```python
from vit_pytorch import (
    HilbertAwareMultiScaleAttention,
    SwiGLUFFN,
    FractalPositionEmbedding,
    LCAHilbertBias,
)

# Attention
attn = ManifoldNativeAttention(
    dim=384, heads=6, max_level=8, beta=4.0
)

# FFN
ffn = SwiGLUFFN(dim=384, hidden_dim=512)

# Position embedding
pos_emb = FractalPositionEmbedding(dim=384, max_level=50)
```

---

## E. Common Imports

```python
# Core model
from vit_pytorch import FractalCurveViT

# Configuration
from vit_pytorch import FractalConfig
from vit_pytorch.split_adaptive import AdaptiveSplitConfig

# Tokenizer
from vit_pytorch import StreamingFractalTokenizerV3

# Components
from vit_pytorch import (
    HilbertAwareMultiScaleAttention,
    LCAHilbertBias,
    LowRankHilbertBias,
    SwiGLUFFN,
    AdaptiveFractalFeedForward,
    FractalPositionEmbedding,
    FractalTransformer,
)

# Utilities
from vit_pytorch.utils import extract_depths, normalize_levels_info
from vit_pytorch.curve_hilbert import HilbertCurve
```

---

## F. Mathematical Notation

| Symbol | Definition |
|:-------|:-----------|
| $I$ | Input image $\in \mathbb{R}^{C \times H \times W}$ |
| $T$ | Token sequence $\in \mathbb{R}^{N \times D}$ |
| $L$ | Level information $\in \mathbb{Z}^{N \times (d_{max}+1)}$ |
| $d$ | Quadtree depth $\in [0, d_{max}]$ |
| $p$ | Quadtree path $\in [0,3]^{d}$ |
| $C(R)$ | Region complexity $\in [0, 1]$ |
| $\tau_d$ | Depth-dependent threshold |
| $H$ | Hilbert curve mapping |
| $\text{LCA}(i,j)$ | Lowest common ancestor depth |
| $B_{hilbert}$ | Hilbert attention bias |
| $\alpha_d$ | Level mixing weight |

---

## G. File Reference

| File | Primary Classes |
|:-----|:----------------|
| `model_fractal_vit.py` | `FractalCurveViT` |
| `tokenizer_streaming.py` | `StreamingFractalTokenizerV3` |
| `split_adaptive.py` | `BalancedGreedySplitter`, `FixedBudgetDPSplitter`, `LearnableSplitter` |
| `attn_hilbert_bias.py` | `HilbertAwareMultiScaleAttention`, `LCAHilbertBias`, `LowRankHilbertBias` |
| `ffn_swiglu.py` | `SwiGLUFFN`, `AdaptiveFractalFeedForward` |
| `embed_fractal_position.py` | `FractalPositionEmbedding` |
| `embed_hilbert_patch.py` | `HilbertNativePatchEmbed` |
| `block_transformer.py` | `FractalTransformer`, `FractalTransformerBlock` |
| `curve_hilbert.py` | `HilbertCurve`, `PseudoHilbertCurve` |
| `config_fractal.py` | `FractalConfig` |
| `base_tokenizer.py` | `BaseTokenizer`, `TokenizerOutput` |
| `utils.py` | Utility functions |
| `constants.py` | Default hyperparameters |
