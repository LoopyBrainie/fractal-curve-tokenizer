# Fractal Curve ViT - Claude Code Instructions

## Quick Reference

```bash
# Package import path
from vit_pytorch import FractalCurveViT  # NOT from fractal_curve_tokenizer

# Testing (use uv run)
uv run pytest tests/                              # Full test suite (~331 tests)
uv run pytest tests/unit/ -v                     # All unit tests
uv run pytest tests/unit/L1_foundation/ -v       # L1 Foundation tests
uv run pytest tests/unit/L2_components/ -v       # L2 Components tests
uv run pytest tests/unit/L3_pipeline/ -v         # L3 Pipeline tests
uv run pytest tests/unit/L4_application/ -v      # L4 Application tests
uv run pytest tests/integration/ -v              # Integration tests
uv run pytest -m "not slow"                      # Skip slow tests

# Training
uv run python src/training/train_fractal_vit.py --dataset tiny-imagenet --use-amp
.\src\training\train_tiny_imagenet_4070_optimal.ps1  # RTX 4070 optimized
.\src\training\train_cub200_4070_optimal.ps1         # CUB-200 fine-grained classification
```

## Installation

```bash
uv sync
uv run pytest tests/
```

**Package import**: Use `from vit_pytorch import FractalCurveViT`, NOT `from fractal_curve_tokenizer`.
**Note**: `tests/conftest.py` adds `src/` to `sys.path` automatically.

---

## Development Rules

### Python Execution

**All Python operations must run via `uv run`**:
- Use `uv run pytest ...` instead of `pytest ...`
- Use `uv run python script.py` instead of `python script.py`
- Use `uv run --with extra-deps python script.py` for dependencies

### Development Environment

- **IDE Development**: PowerShell (no CUDA, no WSL locally)
- **Training**: Separate machine with Podman container (CUDA training)
- Code sync via git/network, not cross-compiling

---

## Development Workflow

### Standard Process for Each Issue

1. **Mathematical Formalization First**
   - Before any code change, derive mathematical forms for the problem
   - Document: problem definition, constraints, objective function
   - Identify: optimal solution criteria, failure modes

2. **Critical Analysis & Discussion**
   - Present proposed approach with mathematical justification
   - Compare against alternative approaches quantitatively
   - Identify trade-offs: O(N) complexity, gradient flow, numerical stability
   - **Wait for user confirmation before proceeding**

3. **Computational Verification**
   - Implement minimal proof-of-concept to validate mathematical claims
   - Run ablation studies comparing alternative approaches
   - Profile: memory, latency, gradient flow

4. **Implementation**
   - Refactor toward optimal Hilbert curve ViT architecture
   - **No backward compatibility constraints** - optimize for best implementation
   - Clean up legacy code paths

5. **Code Review**
   - Verify mathematical properties in implementation
   - Check numerical stability with edge cases
   - Validate computational complexity claims

---

### Issue Prioritization

Process [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) issues: P0-Critical → P0 → P1 → P2 → P3

## Module Hierarchy

| Layer | Purpose | Key Files |
|-------|---------|-----------|
| **L4 Application** | Main model | [model_fractal_vit.py](src/vit_pytorch/model_fractal_vit.py) |
| **L3 Pipeline** | Tokenization & Transformer | [tokenizer_streaming.py](src/vit_pytorch/tokenizer_streaming.py), [block_transformer.py](src/vit_pytorch/block_transformer.py) |
| **L2 Components** | Splitter, Attention, FFN | [base_splitter.py](src/vit_pytorch/base_splitter.py), [gumbel_topk_splitter.py](src/vit_pytorch/gumbel_topk_splitter.py), [attn_hilbert_bias.py](src/vit_pytorch/attn_hilbert_bias.py), [ffn_swiglu.py](src/vit_pytorch/ffn_swiglu.py), [depth_utils.py](src/vit_pytorch/depth_utils.py), [complexity_estimator.py](src/vit_pytorch/complexity_estimator.py) |
| **L1 Foundation** | Hilbert curves, config, levels | [curve_hilbert.py](src/vit_pytorch/curve_hilbert.py), [constants.py](src/vit_pytorch/constants.py), [levels_info.py](src/vit_pytorch/levels_info.py) |
| **Training System** | Modular trainer | [examples/training/](examples/training/) |

## Test Architecture

Tests mirror source code architecture for maintainability.

```
tests/unit/
├── L1_foundation/          # curve_hilbert, constants, levels_info, config
├── L2_components/
│   ├── attention/          # attn_hilbert_bias
│   └── splitters/          # gumbel_topk_splitter, base_splitter
├── L3_pipeline/            # tokenizer_streaming, block_transformer
├── L4_application/         # FractalCurveViT model
├── embeddings/             # patch embeddings, position encodings
└── utilities/              # general utilities
```

Run by layer:

```bash
uv run pytest tests/unit/L1_foundation/ -v
uv run pytest tests/unit/L2_components/ -v
uv run pytest tests/unit/L3_pipeline/ -v
uv run pytest tests/integration/ -v
```

## Critical Analysis

**Key Strengths:**

- Hilbert curve: O(log n) complexity for coordinate transformations
- Gumbel-Top-K: Parallel evaluation of all candidate regions
- LCA attention bias: O(D×H) parameters vs O(N²) learnable bias
- Depth variance normalization: Addresses imbalance across quadtree depths

**Limitations:**

- Hilbert locality bound is an upper bound
- Gradient coverage limited to K selected tokens
- Very low temperatures (T < 0.3) may cause gradient saturation

---

## Developer Workflows

### Training Commands

```bash
# Basic training (CIFAR-10 quick validation)
uv run python examples/training/train_fractal_vit.py --quick-test --use-amp

# Tiny-ImageNet full training (recommended)
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet --epochs 100 \
    --dim 320 --depth 12 --heads 8 \
    --dropout 0.2 --drop-path 0.2 --weight-decay 0.1 \
    --use-amp --gradient-checkpoint --compile --channels-last

# Small dataset (reduce overfitting)
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet --epochs 150 \
    --dim 256 --depth 8 --heads 6 \
    --dropout 0.25 --drop-path 0.25 --freeze-tokenizer --use-amp

# Dynamic resolution (CUB-200, no fixed image_size)
uv run python examples/training/train_fractal_vit.py \
    --dataset cub200 --image-size None \
    --dim 384 --depth 8 --heads 6 \
    --use-amp --compile

# Windows RTX 4070 optimized scripts
.\examples\training\train_tiny_imagenet_4070_optimal.ps1
.\examples\training\train_cub200_4070_optimal.ps1
```

## Computational Efficiency

Attention complexity: B × H × N² vs Standard ViT B × H × 196²

For N ≈ 32 tokens (224×224 image):

- Fractal ViT: ~8K attention elements
- Standard ViT: ~307K elements
- Reduction: **~40×**

---

## Project Conventions

All constants in [constants.py](src/vit_pytorch/constants.py). Use `from .constants import ...`.

### Splitting Schemes

**Default (Scheme D/E)**: `GumbelTopKSplitter` - parallel evaluation, learnable quotas

### Boundary Conditions

- **max_depth=1**: Produces 5 candidate regions (1 depth-0 + 4 depth-1)
- Recommend `max_depth >= 2` for optimal adaptive performance

### Issue Tracking Convention

Test file naming follows Issue ID:
- `test_<module>.py` - Unit tests
- `test_<issue_id>_<feature>.py` - Issue-related tests (e.g., `test_i23_1_depth_balance.py`)

## Critical Patterns

### Cache Decorator Pattern (torch.compile compatibility)

```python
def _dynamo_safe_lru_cache(maxsize: int = 128):
    def decorator(func):
        cached = lru_cache(maxsize=maxsize)(func)
        return torch._dynamo.disable(cached)
    return decorator
```

### STE Gradient Pattern (I78)

```python
selected_mask = F.gumbel_softmax(logits, hard=True)  # [B, N]
loss = loss_fn(outputs, targets)
loss.backward()
```

### EMA Running Statistics (I30-6)

```python
self.register_buffer('running_mean', torch.zeros(depth_dim))
self.register_buffer('running_var', torch.ones(depth_dim))
self.ema_alpha = 0.1
```

### Dynamic Resolution Support (I78)

```python
model = FractalCurveViT(image_size=None, ...)  # Auto-detect from input

---

## Model-Trainer Interface Design

**Core Principle**: Model defines "what I can do", Trainer decides "how to use me".

### Extra Info Convention

`FractalCurveViT.forward()` returns `(logits, aux_infos)` where `aux_infos` contains:
- `num_tokens`: Token count per sample
- `levels_used`: Depth distribution

**Anti-pattern**: Do NOT duplicate ModelConfig parameters in TrainerConfig.

### Callback Pattern

Use `getattr` with fallback for optional model properties:

```python
tokenizer = getattr(model, 'tokenizer', None)
splitter = getattr(tokenizer, 'splitter', None) if tokenizer else None
```

### Two-Trainer Architecture

`CUB200Trainer` does NOT inherit from `ModularTrainer`. Both implement the same Protocol interfaces.

## Recent Changes (v97-v99)

- **I97-10**: Hierarchical independent attention, 4× FLOPs reduction
- **I97-11**: Inference-only dynamic depth, 25% compute savings
- **I96-1~6**: EMA consistency, tree constraint, gradient flow fixes

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Issue tracker with mathematical analysis
- [documents/](documents/): Architecture deep-dives (Chinese)
- [examples/training/README.md](examples/training/README.md): Training system
