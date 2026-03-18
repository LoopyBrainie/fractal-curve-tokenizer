# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Quick Commands

```bash
# Testing
uv run pytest tests/ -v
uv run pytest -m "not slow"          # Skip slow tests

# Training
uv run python src/training/train_fractal_vit.py --quick-test --use-amp
.\src\training\train_tiny_imagenet_4070_optimal.ps1  # RTX 4070
```

## Package Import

**CRITICAL**: Before any Python execution, always add src to path:

```python
import sys
sys.path.insert(0, 'src')
```

**Always use**: `from vit_pytorch import FractalCurveViT` (NOT `from fractal_curve_tokenizer import ...`)

## Development Rules

- Use `uv run pytest ...` / `uv run python ...` (not bare commands)
- IDE: PowerShell (no CUDA locally); Training: Podman container (CUDA)

## Training

```bash
# CUB-200 (dynamic resolution)
uv run python src/training/train_fractal_vit.py --dataset cub200 --image-size None --use-amp --compile
```

## Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

`forward()` returns `TrainingStats`: `logits`, `num_tokens`, `depth_used`, `depth_distribution`, `features`, `transformer_tokens`

**Anti-pattern**: Duplicate ModelConfig in TrainerConfig.

## Three-Layer Parameters

| Type | Definition | Storage |
|------|------------|---------|
| 参数 (Parameters) | Fixed (dim, num_layers, heads) | arch_config |
| 变参数 (Variable) | Runtime (K, max_level) | Computed |
| 超参数 (Hyper) | Fixed architecture (focal_gamma) | arch_config |

**Rule**: Use `arch_config` params, NEVER CLI overrides.

**Save/Load**:

```python
# Save: gene = ModelGene.from_config(arch_config=arch_config, model_state=model.state_dict())
# Load: model = FractalCurveViT(**gene.arch_config.to_dict()); model.load_state_dict(gene.model_state)
```

## Conventions

- Constants: `from vit_pytorch.core.constants import EPS, TEMPERATURE_MIN`
- Default splitter: `GumbelTopKSplitter`
- Issue comments: `# I<issue_id>` (e.g., `# I109-6:`)

## Critical Patterns

- **STE Gradient**: `selected_mask = F.gumbel_softmax(logits, hard=True); loss.backward()`
- **Dynamic Resolution**: `model = FractalCurveViT(image_size=None, ...)`
- **torch.compile cache**: `cached = lru_cache(maxsize=maxsize)(func); return torch._dynamo.disable(cached)`

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Issue tracker
- [docs/](docs): Architecture deep-dives
- [pyproject.toml](pyproject.toml): pytest markers

## Key Strengths

- **Hilbert curve**: O(log n) coordinate transformation complexity
- **Gumbel-Top-K**: Parallel evaluation of all N=85 candidate regions
- **LCA attention**: O(D×H) parameters vs O(N²) for learnable bias
- **Depth variance normalization**: Solves variance imbalance across quadtree depths

## Known Limitations

- **Hilbert locality**: Upper bound, actual preservation depends on traversal order
- **Gradient coverage**: Limited to K selected tokens (K/N ≈ 37.6% with K=32, N=85)
- **Temperature**: T < 0.3 may cause gradient saturation
- **LCA correspondence**: Hilbert indices provide good but not exact quadtree correspondence

## Common Pitfalls

### Numerical Stability

- Always use constants from `constants.py` instead of magic numbers
- Prevent log(0)/div(0) with `*_EPSILON` constants
- Use EMA for stable depth variance normalization

### Gradient Flow

- STE operations require `detach().item()` for loss extraction
- Hierarchical Top-K breaks computation graph - use `detach()` + separate loss calls
- Temperature annealing needs proper warmup configuration

### Performance

- Hilbert curve locality is an upper bound
- LCA computation uses Hilbert indices (good but not exact)
