# Chapter 6: Feed-Forward Network

## 6.1 Overview

The feed-forward network (FFN) in Fractal ViT uses **SwiGLU activation** with optional **level adaptation**, following the LLaMA/PaLM architecture.

---

## 6.2 Mathematical Formulation

### 6.2.1 SwiGLU FFN

$$\text{SwiGLU}(x) = W_{out} \cdot (\text{Swish}(W_{gate} \cdot x) \odot (W_{value} \cdot x))$$

where:
- $\text{Swish}(x) = x \cdot \sigma(x)$ (also known as SiLU)
- $\sigma$: Sigmoid function
- $\odot$: Element-wise multiplication (gating)

### 6.2.2 Level Adaptation

$$\text{Output} = (1 - \alpha_d) \cdot \text{FFN}(x) + \alpha_d \cdot \text{Adapter}([x; E_{level}(d)])$$

where:
- $\alpha_d = \text{softmax}(\text{MixingWeights})_d$: Depth-dependent mixing weight
- $[;]$: Concatenation operator
- $E_{level}(d)$: Level embedding

---

## 6.3 FFN Type Options

| Type | Class | Description | Use Case |
|:-----|:------|:------------|:---------|
| `gelu` | `AdaptiveFractalFeedForward` | Standard GELU | Baseline |
| `swiglu` | `SwiGLUFFN` | SwiGLU without adaptation | Inference |
| `swiglu_level` | `AdaptiveFractalFeedForward` | SwiGLU + Level Adaptation | **Recommended** |

---

## 6.4 SwiGLU Implementation

### Class: SwiGLUFFN

```python
class SwiGLUFFN(nn.Module):
    """
    SwiGLU Feed-Forward Network (LLaMA/PaLM style).
    
    Mathematical formulation:
        gate = Swish(W_gate · x)
        value = W_value · x
        hidden = gate ⊙ value
        output = W_out · hidden
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        
        self.w_gate = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_value = nn.Linear(dim, hidden_dim, bias=bias)
        self.w_out = nn.Linear(hidden_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.w_gate(x))  # Swish activation
        value = self.w_value(x)
        hidden = gate * value  # Element-wise gating
        return self.dropout(self.w_out(hidden))
```

### Parameter Comparison

For `dim=384`:

| Configuration | GELU (hidden=768) | SwiGLU (hidden=512) | Change |
|:--------------|:------------------|:--------------------|:-------|
| FFN params | 590,592 | 589,824 | -0.1% |
| Total model | 1.27M | 1.15M | **-9.4%** |

**Note**: SwiGLU uses 2/3 of the hidden dimension due to the gating mechanism.

---

## 6.5 Level-Adaptive FFN

### Class: AdaptiveFractalFeedForward

```python
class AdaptiveFractalFeedForward(nn.Module):
    """
    Feed-forward network with level adaptation.
    
    Extends SwiGLU with depth-aware processing.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        max_level: int = 50,
        use_level_adaptation: bool = True,
        ffn_type: str = 'swiglu_level',
    ):
        super().__init__()
        
        # Main FFN
        if 'swiglu' in ffn_type:
            self.main_ffn = SwiGLUFFN(dim, hidden_dim, dropout)
        else:
            self.main_ffn = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
            )
        
        # Level adaptation components
        if use_level_adaptation:
            self.level_embedding = nn.Embedding(max_level + 1, dim)
            self.shared_level_adapter = nn.Sequential(
                nn.Linear(dim * 2, dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            self.level_mixing_weights = nn.Parameter(
                torch.zeros(max_level + 1)
            )
```

### Forward Pass with Level Adaptation

```python
def forward(self, x: Tensor, levels_info: Optional[Tensor] = None) -> Tensor:
    # Main FFN output
    main_output = self.main_ffn(x)
    
    if not self.use_level_adaptation or levels_info is None:
        return main_output
    
    # Extract depths
    depths = levels_info[:, :, 0]  # (B, N)
    
    # Level embedding
    level_emb = self.level_embedding(depths)  # (B, N, D)
    
    # Adapter input
    adapter_input = torch.cat([x, level_emb], dim=-1)  # (B, N, 2D)
    level_adapted = self.shared_level_adapter(adapter_input)  # (B, N, D)
    
    # Mixing weights
    mixing = torch.sigmoid(self.level_mixing_weights[depths])  # (B, N)
    mixing = mixing.unsqueeze(-1)  # (B, N, 1)
    
    # Fuse outputs
    return (1 - mixing) * main_output + mixing * level_adapted
```

---

## 6.6 Design Rationale

### Why SwiGLU?

1. **Built-in gating**: Replaces external gating mechanisms
2. **Smoother gradients**: Better training dynamics
3. **Proven effectiveness**: Used in LLaMA, PaLM, Gemma

### Why Level Adaptation?

1. **Scale-aware processing**: Different depths represent different resolutions
2. **Lightweight**: Adds minimal parameters (~2% overhead)
3. **Residual design**: Preserves main FFN when adaptation is unnecessary

---

## 6.7 Deprecated Features

| Feature | Reason | Status |
|:--------|:-------|:-------|
| `use_feature_gating` | SwiGLU has built-in gating | ⚠️ Deprecated |
| `Dynamic Activation` | Entropy > 90%, ineffective | ⚠️ Deprecated |

---

## 6.8 Usage Example

```python
from vit_pytorch import SwiGLUFFN, AdaptiveFractalFeedForward

# Simple SwiGLU
ffn_simple = SwiGLUFFN(
    dim=384,
    hidden_dim=512,  # 2/3 of typical 768
    dropout=0.1,
)

# Level-adaptive SwiGLU
ffn_adaptive = AdaptiveFractalFeedForward(
    dim=384,
    hidden_dim=768,
    dropout=0.1,
    use_level_adaptation=True,
    ffn_type='swiglu_level',
)

x = torch.randn(2, 100, 384)
levels_info = torch.zeros(2, 100, 5, dtype=torch.long)
levels_info[:, :, 0] = torch.randint(0, 5, (2, 100))  # Random depths

out_simple = ffn_simple(x)  # (2, 100, 384)
out_adaptive = ffn_adaptive(x, levels_info)  # (2, 100, 384)
```

> **Next**: [07_transformer_encoder.md](07_transformer_encoder.md) - Transformer Encoder
