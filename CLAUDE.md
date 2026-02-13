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

**Always use**: `from vit_pytorch import FractalCurveViT` (NOT `from fractal_curve_tokenizer import ...`)

## Development Rules

- Use `uv run pytest ...` / `uv run python ...` (not bare commands)
- IDE: PowerShell (no CUDA locally); Training: Podman container (CUDA)

## Module Hierarchy

| Layer | Purpose | Key Files |
|-------|---------|-----------|
| L4 Application | Main model | `models/fractal_vit.py` |
| L3 Pipeline | Tokenization & Transformer | `modules/tokenizer.py`, `modules/transformer_block.py` |
| L2 Components | Splitter, Attention, FFN | `layers/splitters/gumbel_topk.py`, `layers/attention/hilbert_bias.py`, `layers/ffn/swiglu.py` |
| L1 Foundation | Hilbert curves, config | `core/curve_hilbert.py`, `core/config.py` |

## Import Rules

**Hierarchical** (L1→L2→L3→L4):

- L1 imports: None (base)
- L2 imports: L1 only
- L3 imports: L1, L2
- L4 imports: All

**Example fixes**:
```python
# WRONG: from vit_pytorch.modules.base_splitter import CoreSplitter
# CORRECT: from vit_pytorch.core.splitter_protocol import CoreSplitter
```

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

## Efficiency

N ≈ 32 tokens (224×224): Fractal ViT ~8K vs Standard ViT ~307K attention elements (**~40× reduction**)
