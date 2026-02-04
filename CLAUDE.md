# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Package import
from vit_pytorch import FractalCurveViT

# Testing
uv run pytest tests/                              # Full test suite
uv run pytest tests/unit/ -v                     # Unit tests by layer
uv run pytest tests/integration/ -v              # Integration tests
uv run pytest -m "not slow"                      # Skip slow tests
uv run pytest tests/unit/L2_components/splitters/test_splitter_gradient_balance.py -v  # Specific test

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

Test files follow convention: `test_<issue_id>_<feature>.py` (e.g., `test_i109_6_gradient_balance.py`)

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

Test structure mirrors source: `tests/unit/L{1-4}_*/` and `tests/integration/`

## Critical Analysis

**Strengths**:
- Hilbert curve: O(log n) coordinate transformations
- Gumbel-Top-K: Parallel evaluation of all candidate regions
- LCA attention bias: O(D×H) parameters vs O(N²)

**Limitations**:
- Hilbert locality bound is an upper bound
- Gradient coverage limited to K selected tokens
- Very low temperatures (T < 0.3) may cause gradient saturation

## Training

```bash
# Quick test
uv run python src/training/train_fractal_vit.py --quick-test --use-amp

# Tiny-ImageNet (RTX 4070 optimized)
.\src\training\train_tiny_imagenet_4070_optimal.ps1

# CUB-200 (dynamic resolution)
uv run python src/training/train_fractal_vit.py --dataset cub200 --image-size None --use-amp --compile
```

## Computational Efficiency

For N ≈ 32 tokens (224×224 image):
- Fractal ViT: ~8K attention elements
- Standard ViT: ~307K elements
- **~40× reduction**

## Conventions

- Constants: Use `from .constants import ...` in [constants.py](src/vit_pytorch/constants.py)
- Default splitter: `GumbelTopKSplitter` (Scheme D/E)
- Recommended: `max_level >= 2` for optimal adaptive performance
- Issue tracking: Prefix code comments with `# I<issue_id>` (e.g., `# I109-6:`)

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

### Lazy Diagnostics (I145)
```python
class LazyDiagnostics:
    def __init__(self, model):
        self._model = model
        self._filled = False
    def __getitem__(self, key):
        if not self._filled:
            self._model._fill_lazy_diagnostics(self)
        return self._filled[key]
```

## Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

### Forward Return Type
`forward()` returns `TrainingStats` object with fields:
- `logits`: [B, num_classes] classification outputs
- `num_tokens`: Union[int, List[int], torch.Tensor] token count
- `depth_used`: int maximum depth utilized
- `depth_distribution`: Dict[int, float] depth percentage distribution
- `features`: [B, dim] pooled features
- `transformer_tokens`: [B, N, dim] transformer outputs

**Anti-pattern**: Duplicate ModelConfig in TrainerConfig.

**Callback pattern**:
```python
tokenizer = getattr(model, 'tokenizer', None)
splitter = getattr(tokenizer, 'splitter', None) if tokenizer else None
```

## Configuration Files

| File | Purpose |
|------|---------|
| [config.py](src/vit_pytorch/config.py) | FractalConfig, SplitterConfig, AttentionConfig |
| [constants.py](src/vit_pytorch/constants.py) | Numerical constants (GUMBEL_EPSILON, TEMPERATURE_MIN, etc.) |
| [pyproject.toml](pyproject.toml) | pytest markers (slow, integration, e2e, unit, stability, 数学, benchmark) |

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Issue tracker with mathematical analysis
- [docs/](docs): Architecture deep-dives (chapters 00-11)
- [docs/10_testing_qa.md](docs/10_testing_qa.md): Testing guidelines
