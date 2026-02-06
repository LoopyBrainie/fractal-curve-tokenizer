#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop 8GB)
# Mathematical Formalization (2026-02-04 Update)
# Target: 200 epochs, batch=192

set -e

# ============================================================================
# Mathematical Verification Results
# ============================================================================
# GPU: RTX 4070 Laptop (8GB VRAM, ~7GB usable with safety margin)
#
# [tiny-imagenet Characteristics]
#   Image size: 64×64 (native)
#   max_level = ceil(log2(64/4)) = 4
#   Candidate regions = 341 (vs 21,845 for ImageNet)
#   Conclusion: Fewer tokens than ImageNet, smaller model suffices
#
# [MEMORY CONSTRAINT MODEL]
#   M_total = M_params + M_gradients + M_optimizer + M_activations
#   dim=384, L=8:
#   - Parameters (FP16):  48 MB
#   - Gradients (FP32):  96 MB
#   - Optimizer (FP32): 192 MB
#   - Activations (checkpoint): ~200 MB
#   - Total: ~540 MB << 7GB budget
#
# [ARCHITECTURE PARAMETERS - Optimized]
#   --dim 384         : Performance-memory balance (24M params)
#   --num-layers 8    : Match quadtree max_level=4
#   --heads 6          : dim_head = 384/6 = 64
#   --mlp-dim 1536    : mlp_ratio = 4.0 (SwiGLU)
#
# [TOKENIZER PARAMETERS]
#   --patch-size 8
#   --min-patch-size 4
#   --token-coverage-min 0.01 : α = 1%
#   --token-coverage-max 0.20 : β = 20%
#   K (token range): [8, 64] for Tiny-ImageNet 64×64
#
# [TRAINING PARAMETERS - 200 epochs]
#   --batch-size 192  : Target batch size (checkpoint enabled)
#   --lr 1e-4         : Standard learning rate
#   --weight-decay 0.05
#   --dropout 0.25    : Strong regularization for 200 epochs
#   --drop-path 0.25  : Stochastic depth
#   --warmup-epochs 10
#
# [OPTIMIZATION FLAGS]
#   --use-amp, --gradient-checkpoint, --compile, --channels-last
#
# Model: 24M params | Memory: ~4-5 GB | Expected accuracy: 55-65%
# ============================================================================

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 200 \
  --dim 384 \
  --num-layers 8 \
  --heads 6 \
  --mlp-dim 1536 \
  --min-patch-size 4 \
  --token-coverage-min 0.01 \
  --token-coverage-max 0.20 \
  --batch-size 192 \
  --lr 1e-04 \
  --weight-decay 0.05 \
  --dropout 0.25 \
  --drop-path 0.25 \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --include-soft-entropy \
  --include-elastic-budget \
  --warmup-epochs 10 \
  --patience 20
