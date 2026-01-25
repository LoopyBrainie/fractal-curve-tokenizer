# Fractal Curve ViT - Claude Code Instructions

## Quick Reference

```bash
# Package import
from vit_pytorch import FractalCurveViT

# Testing
uv run pytest tests/                              # Full test suite
uv run pytest tests/unit/ -v                     # Unit tests by layer
uv run pytest tests/integration/ -v              # Integration tests
uv run pytest -m "not slow"                      # Skip slow tests

# Training
uv run python src/training/train_fractal_vit.py --dataset tiny-imagenet --use-amp
.\src\training\train_tiny_imagenet_4070_optimal.ps1  # RTX 4070 optimized
.\src\training\train_cub200_4070_optimal.ps1         # CUB-200 fine-grained
```

## Package Import

**ALWAYS use this import pattern**:
```python
from vit_pytorch import FractalCurveViT
# NOT from fractal_curve_tokenizer import FractalCurveViT
```
The `src/` directory is already in `sys.path` via `tests/conftest.py` and project structure.

## Development Rules

**All Python operations must run via `uv run`**:
- `uv run pytest ...` instead of `pytest ...`
- `uv run python script.py` instead of `python script.py`
- `uv run --with extra-deps python script.py` for extra dependencies

**Environment**:
- IDE: PowerShell (no CUDA locally)
- Training: Podman container (CUDA), sync via git/network

## Development Workflow

1. **Mathematical Formalization First** - Derive mathematical forms, document problem/constraints/objective
2. **Critical Analysis & Discussion** - Present approach with justification, compare alternatives, identify trade-offs (O(N), gradient flow, numerical stability). **Wait for user confirmation**
3. **Computational Verification** - PoC, ablation studies, profiling
4. **Implementation** - No backward compatibility constraints, clean up legacy code
5. **Code Review** - Verify mathematical properties, numerical stability, complexity claims

## Issue Processing

Process [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) by priority: P0-Critical → P0 → P1 → P2 → P3

## Vectorization Testing

**Two-phase verification**:

1. **VectorizationAuditor** - Detects non-vectorized patterns (scalar extraction `.item()`, Python loops, hardcoded batch_size)
2. **vmap stress test** - Validates true vectorization

```bash
uv run pytest tests/integration/test_system.py::test_batch_consistency -v
uv run pytest tests/unit/utilities/test_vectorization.py -m vectorization -v
```

## Module Hierarchy

| Layer | Purpose | Key Files |
|-------|---------|-----------|
| **L4 Application** | Main model | [model_fractal_vit.py](src/vit_pytorch/model_fractal_vit.py) |
| **L3 Pipeline** | Tokenization & Transformer | [tokenizer_streaming.py](src/vit_pytorch/tokenizer_streaming.py), [block_transformer.py](src/vit_pytorch/block_transformer.py) |
| **L2 Components** | Splitter, Attention, FFN | [gumbel_topk_splitter.py](src/vit_pytorch/gumbel_topk_splitter.py), [attn_hilbert_bias.py](src/vit_pytorch/attn_hilbert_bias.py), [ffn_swiglu.py](src/vit_pytorch/ffn_swiglu.py) |
| **L1 Foundation** | Hilbert curves, config | [curve_hilbert.py](src/vit_pytorch/curve_hilbert.py), [constants.py](src/vit_pytorch/constants.py) |

Test structure mirrors source: `tests/unit/L{1-4}_*/`

## Critical Analysis

**Strengths**:
- Hilbert curve: O(log n) coordinate transformations
- Gumbel-Top-K: Parallel evaluation of all candidate regions
- LCA attention bias: O(D×H) parameters vs O(N²)

**Limitations**:
- Hilbert locality bound is an upper bound
- Gradient coverage limited to K selected tokens
- Very low temperatures (T < 0.3) may cause gradient saturation

## Training Commands

```bash
# Basic quick validation
uv run python src/training/train_fractal_vit.py --quick-test --use-amp

# Tiny-ImageNet full training
uv run python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet --epochs 100 \
    --dim 320 --depth 12 --heads 8 \
    --dropout 0.2 --drop-path 0.2 --weight-decay 0.1 \
    --use-amp --gradient-checkpoint --compile --channels-last

# Dynamic resolution (CUB-200)
uv run python src/training/train_fractal_vit.py \
    --dataset cub200 --image-size None \
    --dim 384 --depth 8 --heads 6 \
    --use-amp --compile
```

## Computational Efficiency

For N ≈ 32 tokens (224×224 image):
- Fractal ViT: ~8K attention elements
- Standard ViT: ~307K elements
- **~40× reduction**

## Conventions

- Constants: Use `from .constants import ...` in [constants.py](src/vit_pytorch/constants.py)
- Default splitter: `GumbelTopKSplitter` (Scheme D/E)
- Recommended: `max_depth >= 2` for optimal adaptive performance
- Test files: `test_<issue_id>_<feature>.py` (e.g., `test_i23_1_depth_balance.py`)

## Critical Patterns

### Cache Decorator (torch.compile)
```python
cached = lru_cache(maxsize=maxsize)(func)
return torch._dynamo.disable(cached)
```

### STE Gradient (I78)
```python
selected_mask = F.gumbel_softmax(logits, hard=True)
loss.backward()
```

### Dynamic Resolution
```python
model = FractalCurveViT(image_size=None, ...)  # Auto-detect from input
```

## Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

`forward()` returns `(logits, aux_infos)` with `num_tokens`, `levels_used`.

**Anti-pattern**: Duplicate ModelConfig in TrainerConfig.

**Callback pattern**:
```python
tokenizer = getattr(model, 'tokenizer', None)
splitter = getattr(tokenizer, 'splitter', None) if tokenizer else None
```

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Issue tracker with mathematical analysis
- [documents/](documents/): Architecture deep-dives (Chinese)
- [examples/training/README.md](examples/training/README.md): Training system
