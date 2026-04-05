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
| L2 Components | Splitter, Attention, FFN | `layers/splitters/hilbert_optimal_splitter.py`, `layers/attention/manifold_attention.py`, `layers/ffn/swiglu.py` |
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
- Default splitter: `HilbertOptimalSplitter` (H1SS)

## Logging System Design

### Layer-Packaged → Trainer-Unpacked Architecture

**设计原则**：每层自打包诊断数据，训练器只负责解开（flatten + record），新增层无需修改训练器。

```
┌─────────────────────────────────────────────────────┐
│  LAYER 模块（自己打包）                                │
│  Splitter → splitter_output = {entropy, alpha, ...}  │
│  Attention → attn_output = {geometric_bias_*, ...}  │
│  FFN → ffn_output = {level_mixing_*, adapter_norm}  │
└─────────────────────────────────────────────────────┘
                          ↓ auxiliary_outputs
┌─────────────────────────────────────────────────────┐
│  主模型 FractalCurveViT.forward()                    │
│  TrainingStats.auxiliary_outputs = {                 │
│      "splitter": splitter_output,                     │
│      "attn_0": attn_output,                         │
│      "ffn_0": ffn_output,                           │
│  }                                                   │
└─────────────────────────────────────────────────────┘
                          ↓ flatten_layer_outputs
┌─────────────────────────────────────────────────────┐
│  训练器 UnifiedMonitor                                 │
│  post_forward():                                     │
│    1. _record_auxiliary_outputs() → train/layer/key  │
│    2. _record_training_stats() → core metrics        │
└─────────────────────────────────────────────────────┘
```

### 核心文件

| 文件 | 职责 |
|------|------|
| `src/vit_pytorch/core/layer_output.py` | `LayerOutputProtocol` + `flatten_layer_outputs()` |
| `src/vit_pytorch/core/splitter_protocol.py` | `SplitResult.splitter_output` 属性 |
| `src/vit_pytorch/layers/attention/manifold_attention.py` | `ManifoldNativeAttention.attn_output` 属性 |
| `src/vit_pytorch/layers/ffn/swiglu.py` | `AdaptiveFractalFeedForward.ffn_output` 属性 |
| `src/vit_pytorch/models/fractal_vit.py` | 收集 auxiliary_outputs 到 TrainingStats |
| `src/training/monitor/unified.py` | `_record_auxiliary_outputs()` + `_record_training_stats()` |

### 展平后的键名格式

所有 auxiliary_outputs 展平后统一为 `train/{layer}/{metric}` 格式：

| 来源 | 键名格式 | 示例 |
|------|---------|------|
| Splitter | `train/splitter/{metric}` | `train/splitter/entropy`, `train/splitter/active_ratio` |
| Attention | `train/attn_{i}/{metric}` | `train/attn_0/geometric_bias_mean` |
| FFN | `train/ffn_{i}/{metric}` | `train/ffn_0/level_mixing_mean` |

### auxiliary_outputs 完整指标清单

**Splitter 输出**（`SplitResult.splitter_output`）：
`active_count`, `active_ratio`, `logits_mean/std/abs_mean`, `probs_max/entropy`, `entropy`, `budget_loss`, `tree_consistency`, `locality_score`, `jump_loss`, `iou_mean/std`, `alpha`

**Attention 输出**（`ManifoldNativeAttention.attn_output`）：
`geometric_bias_mean/std/max`, `bandwidth_mean/min/max`, `poincare_dist_mean/std`, `residual_scale_mean`

**FFN 输出**（`AdaptiveFractalFeedForward.ffn_output`）：
`ffn_gamma_mean/std`, `ffn_beta_mean/std`, `level_mixing_mean/std/min/max`, `adapter_dominance`, `main_ffn_norm`, `adapter_norm`, `contribution_ratio`, `ffn_type`

### TrainingStats 直接字段（不在 auxiliary_outputs 中）

这些字段由 `_record_training_stats()` 直接记录，不经过展平：

`num_tokens`, `depth_used`, `logits_mean/std`, `manifold_bias_max/mean/std`, `backbone_grad_norm`, `splitter_grad_norm`, `loss_*` (auxiliary_losses)

### 新增层的日志集成

新增层只需：
1. 实现 `xxx_output` 属性返回 `Dict[str, float]`
2. 在 `fractal_vit.py` 的 auxiliary_outputs 收集逻辑中添加新 key
3. 无需修改训练器，`flatten_layer_outputs()` 自动处理

### Critical Patterns

- **STE Gradient**: `selected_mask = F.gumbel_softmax(logits, hard=True); loss.backward()`
- **Dynamic Resolution**: `model = FractalCurveViT(image_size=None, ...)`
- **torch.compile cache**: `cached = lru_cache(maxsize=maxsize)(func); return torch._dynamo.disable(cached)`

## Documentation

- [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md): Issue tracker
- [docs/](docs): Architecture deep-dives
- [pyproject.toml](pyproject.toml): pytest markers
