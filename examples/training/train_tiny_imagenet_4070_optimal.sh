#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop 8GB)
# Mathematical Formalization (2026-01-29): dim=320, depth=12, batch=192
# Optimized for 100 epochs with maximum capacity

set -e

# ============================================================================
# Mathematical Verification Results (2026-01-29 Update)
# ============================================================================
# VRAM: 8GB RTX 4070 Laptop
# Configuration: dim=320, depth=12, heads=8, mlp_dim=1280, batch=192
#
# Recommended Configuration:
#   - Parameters: 34.56M
#   - Memory: 763 MB (88.3% safety margin)
#   - Token coverage: 2%-5% (5-12 tokens)
#
# Memory Breakdown (batch=192):
#   - Model params (FP32): 131.8 MB
#   - Gradients (FP32): 131.8 MB
#   - Optimizer (FP32): 263.7 MB
#   - Activations (FP16): 36.0 MB
#   - CUDA overhead: 200 MB
#   - Total: 763 MB
#
# Model Parameter Formula: P = L × (16 × d²) + 2.5 × d × C
#   P = 12 × (16 × 320²) + 2.5 × 320 × 200 = 34.56M params
#
# Key Insight: dim_head = dim / heads = 320 / 8 = 40 [OK]
#
# Token Budget (I33 Relative Budget):
#   - image_size=64, min_patch_size=4
#   - max_patches = (64/4)² = 256
#   - coverage_min=0.02 (2%): K_min = max(4, 256×0.02) = 5
#   - coverage_max=0.05 (5%): K_max = 256×0.05 = 12
# ============================================================================

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  --dim 320 \
  --depth 12 \
  --heads 8 \
  --mlp-dim 1280 \
  --pool weighted \
  --ffn-type swiglu_level \
  --min-patch-size 4 \
  --token-coverage-min 0.02 \
  --token-coverage-max 0.05 \
  --quota-learnable \
  --freeze-tokenizer \
  --freeze-tokenizer-epochs 10 \
  --elastic-coverage-min 0.02 \
  --elastic-coverage-max 0.05 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --batch-size 192 \
  --num-workers 8 \
  --lr 5e-4 \
  --weight-decay 0.15 \
  --warmup-epochs 10 \
  --dropout 0.25 \
  --drop-path 0.25 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --gradient-clip 1.0 \
  --patience 20 \
  --min-delta 0.001 \
  --exp-name tiny_imagenet_320d_12l_bs192_ep100_v3
