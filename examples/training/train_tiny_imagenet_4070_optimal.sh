#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop 8GB)
# Mathematical Formalization (2026-02-09 Update)
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
#   Token range: [8, 36] for 64×64 images
#
# [MEMORY CONSTRAINT MODEL]
#   M_total = M_params + M_gradients + M_optimizer + M_activations
#   dim=384, L=8: ~22M params | VRAM: ~4-5 GB
#
# [KEY OPTIMIZATIONS]
#   - tokenizer_dropout=0.0  : Deterministic tokenization (critical)
#   - transformer_dropout=0.25: Strong regularization (200 epochs)
#   - focal_gamma=2.5        : Hard/easy sample ratio 243x
#   - soft_entropy + elastic_budget: Depth diversity + token budget
#
# Expected: Top-1 Accuracy 55-65% | Time: ~20-30 hours
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
  --K-min-abs 8 \
  --batch-size 192 \
  --lr 1e-04 \
  --weight-decay 0.05 \
  --tokenizer-dropout 0.0 \
  --transformer-dropout 0.25 \
  --emb-dropout 0.0 \
  --drop-path 0.25 \
  --label-smoothing 0.1 \
  --gradient-clip 1.0 \
  --warmup-epochs 10 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 0 \
  --focal-gamma 2.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-coverage-min 0.03 \
  --elastic-coverage-max 0.25 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --quota-learnable enable \
  --quota-entropy-weight 0.01 \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --patience 25 \
  --num-workers 8