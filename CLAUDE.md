# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## Module Hierarchy

| Layer | Purpose | Key Files |
| ----- | ----- | ----- |
| L4 Application | Main model | `models/fractal_vit.py` |
| L3 Pipeline | Tokenization & Transformer | `modules/tokenizer.py`, `modules/transformer_block.py` |
| L2 Components | Splitter, Attention, FFN | `layers/splitters/hilbert_optimal_splitter.py`, `layers/attention/manifold_attention.py`, `layers/ffn/swiglu.py` |
| L1 Foundation | Hilbert curves, config | `core/curve_hilbert.py`, `core/config.py` |

## Import Rules

**Hierarchical** (L1→L2→L3→L4):

- L1 imports: None (base)
- L2 imports: L1 only
- L3 imports: L1, L2
- L4 imports: All

**Wrong**: `from vit_pytorch.modules.base_splitter import CoreSplitter`
**Correct**: `from vit_pytorch.core.splitter_protocol import CoreSplitter`

## Training

```bash
# CUB-200 (dynamic resolution)
uv run python src/training/train_fractal_vit.py --dataset cub200 --image-size None --use-amp --compile
```

## Model-Trainer Interface

**Principle**: Model defines capabilities, Trainer decides usage.

`forward()` returns `TrainingStats`: `logits`, `num_tokens`, `depth_used`, `depth_distribution`, `features`, `transformer_tokens`

## Conventions

- Constants: `from vit_pytorch.core.constants import EPS, TEMPERATURE_MIN`
- Default splitter: `HilbertOptimalSplitter` (H1SS)

## Logging System Design

### Layer-Packaged → Trainer-Unpacked Architecture

```
LAYER → auxiliary_outputs → FractalCurveViT.forward() → flatten → UnifiedMonitor
```

**Core files**:

- `src/vit_pytorch/core/layer_output.py` - `LayerOutputProtocol` + `flatten_layer_outputs()`
- `src/vit_pytorch/core/splitter_protocol.py` - `SplitResult.splitter_output` property
- `src/vit_pytorch/models/fractal_vit.py` - collects auxiliary_outputs
- `src/training/monitor/unified.py` - `_record_auxiliary_outputs()` + `_record_training_stats()`

**Flattened key format**: `train/{layer}/{metric}`

| Source | Format | Example |
| ----- | ----- | ----- |
| Splitter | `train/splitter/{metric}` | `train/splitter/entropy` |
| Attention | `train/attn_{i}/{metric}` | `train/attn_0/geometric_bias_mean` |
| FFN | `train/ffn_{i}/{metric}` | `train/ffn_0/level_mixing_mean` |

**Adding a new layer**:

1. Implement `xxx_output` property returning `Dict[str, float]`
2. Add key in `fractal_vit.py` auxiliary_outputs collection
3. No trainer changes needed

## Critical Patterns

- **STE Gradient**: `selected_mask = F.gumbel_softmax(logits, hard=True); loss.backward()`
- **Dynamic Resolution**: `model = FractalCurveViT(image_size=None, ...)`
- **torch.compile cache**: `cached = lru_cache(maxsize=maxsize)(func); return torch._dynamo.disable(cached)`

## Known Limitations

- **Hilbert locality**: Upper bound, actual preservation depends on traversal order
- **Gradient coverage**: Limited to K selected tokens in分裂 regions
- **LCA correspondence**: Hilbert indices provide good but not exact quadtree correspondence

## Common Pitfalls

### Numerical Stability

- Use constants from `constants.py` instead of magic numbers
- Prevent log(0)/div(0) with `*_EPSILON` constants

### Gradient Flow

- STE operations require `detach().item()` for loss extraction
- Hierarchical Top-K breaks computation graph - use `detach()` + separate loss calls

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Issue tracker
- [docs/](docs): Architecture deep-dives
