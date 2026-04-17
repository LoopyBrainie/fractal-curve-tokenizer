# Chapter 3: Fractal Tokenization Pipeline

## 3.1 Overview

The **Fractal Tokenization Pipeline** transforms raw input images into adaptive, variable-length sequences of fractal tokens. Unlike standard Vision Transformers (ViTs) that use a rigid grid of fixed-size patches, this pipeline dynamically partitions the image space based on local complexity using a quadtree-structured approach mapped onto a Hilbert curve.

This ensures:
- **High-entropy regions** (edges, textures) receive higher token density
- **Low-entropy regions** (background) are represented by fewer, larger tokens

---

## 3.2 Pipeline Data Flow

```
Image (B, C, H, W)
        │
        ▼
┌───────────────────────────────────────────────┐
│         L2 Components                          │
│  ┌─────────────────────────────────────────┐ │
│  │ SharedConv (Feature Extraction)         │ │
│  │ HilbertSplitter (Optimal/Entmax)        │ │
│  │ Vectorized ROI-Align Pooling           │ │
│  │ HilbertSort (1D Ordering)               │ │
│  └─────────────────────────────────────────┘ │
└───────────────────────────────────────────────┘
        │
        ▼
   Token Sequence (B, N, D)
        │
        ▼
   LevelsInfo (Metadata)
```

---

## 3.3 StreamingFractalTokenizerV3

The `StreamingFractalTokenizerV3` is the unified implementation of the tokenization engine.

### 3.3.1 Key Responsibilities

| Component | Purpose |
|:----------|:--------|
| **SharedConv** | Initial feature extraction to avoid redundant computations |
| **Vectorized Pipeline** | P9-1 optimization for parallel processing of variable-depth tokens |
| **HilbertSort** | Map quadtree nodes to 1D sequence preserving spatial locality |

### 3.3.2 SplitResult Contract

```python
@dataclass
class SplitResult:
    tokens: Tensor              # [B, N, D] Token embeddings
    paths: Tensor              # [B, N, max_level] Quadtree paths
    depths: Tensor             # [B, N] Token depths
    indices: Tensor           # [B, N] Hilbert indices for sorting
    levels_info: LevelsInfo     # Hierarchical metadata
    num_tokens: Tensor         # [B] Token count per sample
    auxiliary_outputs: Dict[str, float]  # Entropy, budget loss, etc.
```

### 3.3.3 Vectorized Processing (P9-1)

The tokenizer uses **vectorized operations** for `torch.compile` compatibility:

```python
def tokenize(self, x: Tensor) -> SplitResult:
    # Step 1: Shared feature extraction
    features = self.shared_conv(x)  # [B, D, H', W']

    # Step 2: Recursive splitting (vectorized)
    candidates = self._generate_candidates(features)

    # Step 3: Parallel ROI pooling
    tokens = self.roi_align(features, regions)

    # Step 4: Hilbert ordering
    indices = self.hilbert_sorter(paths, depths)

    return SplitResult(...)
```

**Source**: `src/vit_pytorch/modules/tokenizer.py`

---

## 3.4 Token Splitters

The splitter is the **"brain"** of the pipeline, determining the granularity of the fractal representation.

### 3.4.1 Splitter Comparison

| Splitter Class | Mechanism | Key Feature |
|:---------------|:----------|:-----------|
| `HilbertOptimalSplitter` | H1SS (6 Axioms) | **Recommended default**; optimal region selection |
| `HilbertDistanceDecayConv` | Spatial Decay | Biases splitting towards central/high-detail anchors |

### 3.4.2 HilbertOptimalSplitter (H1SS)

The recommended default splitter implementing **6 axioms** for tree consistency:

```python
HilbertOptimalSplitter(
    feature_dim=256,
    hidden_dim=64,
    max_level_limit=8,
    K_min=8,              # Minimum token count
    K_max=64,             # Maximum token count
    entmax_alpha=1.2,     # Entmax parameter
    tree_constraint_weight=0.1,  # Tree consistency loss weight
    temperature_init=1.0,
    temperature_min=0.3,
    jump_loss_weight=0.1,  # Encourages depth jumps
    density_field_hidden_dim=32,
    use_distance_decay_conv=True,   # I167-1
    use_sds_regularization=False,   # I167-4
    sds_lambda=0.1,
)
```

**Axioms (H1SS)**:

| Axiom | Description |
|:------|:------------|
| **A1** | Coverage: Root covers entire image |
| **A2** | Containment: Child ⊆ parent |
| **A3** | Disjointness: Siblings don't overlap |
| **A4** | Max Depth: ≤ $D_{max}$ |
| **A5** | Monotonicity: Parent selected ⇒ ≥1 child selected |
| **A6** | Continuity: Selected regions form connected subgraph |

### 3.4.3 HilbertDistanceDecayConv (I167-1)

**Decoupled Hilbert Distance Decay Convolution** for spatial mixing:

```python
z = Pointwise(Depthwise(x, w_decay))
w_decay[k] = 1 / (|k - center| + 1)
```

**Key Properties**:
- **Depthwise**: Fixed distance decay weights (0 parameters)
- **Pointwise**: Learnable channel mixing (D parameters)

### 3.4.4 Auxiliary Losses

Splitters provide auxiliary losses for training stability:

| Loss | Purpose | Code |
|:-----|:--------|:-----|
| **Entropy Loss** | Prevents collapse to single scale | `splitter.entropy` |
| **Budget Loss** | Enforces token count constraints $K_{min}, K_{max}$ | `splitter.budget_loss` |
| **Tree Constraint** | Soft constraint on parent-child quadtree logic | `tree_constraint_weight` |

---

## 3.5 Hilbert Pattern Encoder (I162-1)

The `HilbertPatternEncoder` performs **multi-scale 1D convolution** on Hilbert-ordered tokens.

### 3.5.1 Architecture

```
Hilbert-Ordered Tokens [B, N, D]
        │
        ▼
┌─────────────────────────────────────────────┐
│   Multi-Scale Depthwise Conv1D             │
│   ┌─────────────────────────────────────┐ │
│   │ kernel_size=3  → Local patterns    │ │
│   │ kernel_size=5  → Regional context  │ │
│   │ kernel_size=7  → Broad structure   │ │
│   └─────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
        │
        ▼
   Pattern Features [B, N, D]
```

### 3.5.2 Mathematical Formulation

$$f_{out} = \text{Concat}\left( \text{Conv1D}_3(x), \text{Conv1D}_5(x), \text{Conv1D}_7(x) \right)$$

**Source**: `src/vit_pytorch/core/pattern_encoder.py`

---

## 3.6 SDS Regularization (I167-4)

**SDS (Spatial Discontinuity Score)** regularizes the splitter to favor spatially coherent regions:

$$\text{SDS} = \frac{1}{N^2} \sum_{i,j} \left| \|p_i - p_j\|_2 - \frac{|h_i - h_j|}{H_{\max}} \right|$$

where:
- $p_i, p_j$: 2D physical coordinates
- $h_i, h_j$: Hilbert indices
- $H_{\max}$: Maximum Hilbert index

**Usage**:

```python
HilbertOptimalSplitter(
    use_sds_regularization=True,
    sds_lambda=0.1,  # Regularization weight
)
```

---

## 3.7 Patch Embeddings and Position Encoding

### 3.7.1 FractalPositionEmbedding

Combines depth embeddings with quadrant path encodings:

$$E_{pos}(i) = \text{Fusion}(E_{depth}(d_i) + E_{path}(i))$$

### 3.7.2 Level-Aware Components

| Component | Purpose |
|:----------|:--------|
| `FractalPathEmbedding` | Encodes quadrant path sequence |
| `LevelEmbedding` | Encodes quadtree depth |
| `AreaEnhancedEncoding` | Adjusts magnitude based on patch area |

### 3.7.3 Area-Enhanced Encoding

Adjusts feature magnitude based on physical patch size:

```python
# Prevent signal washout for large patches
scale = sqrt(area_patch / area_total)
enhanced = tokens * scale
```

---

## 3.8 System Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FractalCurveViT                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Image (B, C, H, W)                                                 │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │            StreamingFractalTokenizerV3                       │     │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌────────────┐   │     │
│  │  │   SharedConv    │→ │ HilbertOptimal  │→ │ HilbertSort │   │     │
│  │  │ (Feature Extract)│  │ Splitter (H1SS) │  │            │   │     │
│  │  └─────────────────┘  └─────────────────┘  └────────────┘   │     │
│  └─────────────────────────────────────────────────────────────┘     │
│       │                                                              │
│       ▼                                                              │
│  SplitResult (tokens, paths, depths, levels_info)                   │
│       │                                                              │
│       ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────┐     │
│  │            FractalPositionEmbedding                          │     │
│  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐    │     │
│  │  │ LevelEmbed  │→ │ FractalPath  │→ │ AreaEnhanced  │    │     │
│  │  │ (depth)     │  │ (quadrant)   │  │ (magnitude)  │    │     │
│  │  └──────────────┘  └──────────────┘  └───────────────┘    │     │
│  └─────────────────────────────────────────────────────────────┘     │
│       │                                                              │
│       ▼                                                              │
│  FractalTransformer × L                                              │
│       │                                                              │
│       ▼                                                              │
│  Class Logits                                                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3.9 Key Innovations (I162-1, I167-1, I167-2, I167-4)

| Issue | Solution | Reference |
|:------|:---------|:----------|
| **I162-1** | Hilbert Pattern Encoder for multi-scale features | `pattern_encoder.py` |
| **I167-1** | Distance Decay Conv for spatial mixing | `hilbert_distance_decay_conv.py` |
| **I167-2** | SDSMetric for Hilbert locality validation | `curve_hilbert.py` |
| **I167-4** | SDS regularization in splitter | `HilbertOptimalSplitter` |

---

## 3.10 Document Navigation

| Chapter | Content |
|:--------|:--------|
| [03_fractal_tokenizer](03_fractal_tokenizer.md) | Fractal Tokenization Pipeline (this chapter) |
| [04_positional_embedding](04_positional_embedding.md) | Position encoding |
| [05_attention_mechanism](05_attention_mechanism.md) | Manifold-native attention |

> **Next**: [04_positional_embedding.md](04_positional_embedding.md) - Position Encoding
