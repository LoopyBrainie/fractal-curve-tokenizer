# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Sanity Check

```bash
uv run pytest -m "not slow"
```

## Package Import

`src/` path is auto-configured in training scripts. For standalone scripts:
```python
import sys
sys.path.insert(0, 'src')
```

**Always use**: `from vit_pytorch import FractalCurveViT` (NOT `from fractal_curve_tokenizer import ...`)

## Development Rules

- **Mathematical Formalization First**: Derive mathematical forms before any code changes

## Module Hierarchy

| Layer | Purpose |
| ----- | ----- |
| L4 Application | Main model (`FractalCurveViT`) |
| L3 Pipeline | Tokenization & Transformer |
| L2 Components | Splitter, Attention, FFN |
| L1 Foundation | Hilbert curves, config primitives |

## Import Rules

**Hierarchical** (L1→L2→L3→L4):

- L1 imports: None (base)
- L2 imports: L1 only
- L3 imports: L1, L2
- L4 imports: All

**Wrong**: `from vit_pytorch.modules.base_splitter import CoreSplitter`
**Correct**: `from vit_pytorch.core.splitter_protocol import CoreSplitter`

## Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

`forward()` returns `TrainingStats`: `logits`, `num_tokens`, `depth_used`, `depth_distribution`, `features`, `transformer_tokens`

## Logging System Design

Key format: `train/{layer}/{metric}`

| Source | Format |
| ----- | ----- |
| Splitter | `train/splitter/{metric}` |
| Attention | `train/attn_{i}/{metric}` |
| FFN | `train/ffn_{i}/{metric}` |

**Adding a new layer**: Implement `xxx_output` property → add key in `fractal_vit.py` auxiliary_outputs → no trainer changes.

## Critical Patterns

**torch.compile cache** (`lru_cache` breaks Dynamo, must be wrapped):
```python
def _dynamo_safe_lru_cache(maxsize: int = 128):
    cached = lru_cache(maxsize=maxsize)(func)
    return torch._dynamo.disable(cached)
```

## Conventions

- Constants: `from vit_pytorch.core.constants import EPS, TEMPERATURE_MIN`
- Default splitter: `HilbertOptimalSplitter` (H1SS)
- Code comments: `# I24-2: Import Scheme E learnable quota constants`

## Common Pitfalls

- **Hierarchical Top-K breaks computation graph**: use `detach()` + separate loss calls

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Historical Issue Archive
- [docs/00_introduction.md](docs/00_introduction.md): 架构入口 (详见 docs/00-10 各章节)
- [tests/CLAUDE.md](tests/CLAUDE.md): 测试特定约定 (STE 桥损失、T10 keystone 等)
