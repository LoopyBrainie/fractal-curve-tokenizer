# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Commands

```bash
uv run pytest tests/ -v
uv run pytest -m "not slow"
uv run pytest tests/test_fractal_rope.py

uv run python src/training/train_fractal_vit.py --quick-test --use-amp
uv run python src/training/train_fractal_vit.py --dataset cub200 --image-size None --use-amp --compile
.\src\training\train_tiny_imagenet_4070_optimal.ps1
```

## Package Import

**CRITICAL**: `src/` path is auto-configured in training scripts. For standalone scripts:
```python
import sys
sys.path.insert(0, 'src')
```

**Always use**: `from vit_pytorch import FractalCurveViT` (NOT `from fractal_curve_tokenizer import ...`)

## Development Rules

- Use `uv run pytest ...` / `uv run python ...` (not bare commands)
- **Mathematical Formalization First**: Derive mathematical forms before any code changes

## Module Hierarchy

| Layer | Purpose |
| ----- | ----- |
| L4 Application | Main model (`FractalCurveViT`) |
| L3 Pipeline | Tokenization (`StreamingFractalTokenizer`) & Transformer (`FractalTransformer`) |
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

## Core Concepts

### Hilbert Curve Tokenization
Space-filling curve preserving 2D locality. Tokenizer: `StreamingFractalTokenizerV3` + `HilbertOptimalSplitter` (H1SS) for adaptive quadtree decomposition.

### Dual-Path RoPE
- **Physical field** (first D/2): `Cartesian2DRoPE`
- **Topology field** (second D/2): `DirectionAwareSubspacedRoPE` (C+ RoPE)

C+ RoPE provides direction-aware subspace isolation for Hilbert curve topology.

### Adaptive Tokenization
Token count `N ∈ [K_min, K_max]` where `K_min=8, K_max=64`. More tokens → complex regions; fewer → uniform areas.

## Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

`forward()` returns `TrainingStats`: `logits`, `num_tokens`, `depth_used`, `depth_distribution`, `features`, `transformer_tokens`

## Logging System Design

**Layer-Packaged → Trainer-Unpacked**: `LAYER → auxiliary_outputs → FractalCurveViT.forward() → flatten → UnifiedMonitor`

Key format: `train/{layer}/{metric}`

| Source | Format |
| ----- | ----- |
| Splitter | `train/splitter/{metric}` |
| Attention | `train/attn_{i}/{metric}` |
| FFN | `train/ffn_{i}/{metric}` |

**Adding a new layer**: Implement `xxx_output` property → add key in `fractal_vit.py` auxiliary_outputs → no trainer changes.

## Critical Patterns

- **STE Gradient**: `F.gumbel_softmax(logits, hard=True)` + `loss.backward()`
- **Dynamic Resolution**: `model = FractalCurveViT(image_size=None, ...)`
- **torch.compile cache**:
  ```python
  def _dynamo_safe_lru_cache(maxsize: int = 128):
      cached = lru_cache(maxsize=maxsize)(func)
      return torch._dynamo.disable(cached)
  ```

## Conventions

- Constants: `from vit_pytorch.core.constants import EPS, TEMPERATURE_MIN`
- Default splitter: `HilbertOptimalSplitter` (H1SS)
- Issue tracking: `test_<issue_id>_<feature>.py`
- Code comments: `# I24-2: Import Scheme E learnable quota constants`

## Common Pitfalls

### Numerical Stability
- Use constants from `constants.py` instead of magic numbers
- Prevent log(0)/div(0) with `*_EPSILON` constants

### Gradient Flow
- STE operations require `detach().item()` for loss extraction
- Hierarchical Top-K breaks computation graph - use `detach()` + separate loss calls

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Issue tracker
- [docs/01_overview.md](docs/01_overview.md): System architecture
- [docs/03_fractal_tokenizer.md](docs/03_fractal_tokenizer.md): Tokenization
- [docs/05_attention_mechanism.md](docs/05_attention_mechanism.md): Attention
