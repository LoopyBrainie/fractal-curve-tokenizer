#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop)
# Mathematical Formalization (2026-01-26): dim=320, depth=10, batch=192
# Optimized for 200 epochs with maximum throughput

set -e

# ============================================================================
# 数学形式化验证结果
# ============================================================================
# VRAM: 8GB RTX 4070 Laptop
# Configuration: dim=320, depth=10, heads=8, batch=192
#
# Memory Breakdown (checkpoint + amp):
#   - Model params: ~100 MB (25M params)
#   - Activations: ~36 MB (checkpoint saves 65%)
#   - Gradients: ~100 MB
#   - Optimizer states: ~200 MB (AdamW: 2x params)
#   - Total: ~700 MB << 8GB VRAM
#
# Effective Batch = 192 × 3 accum = 576
#
# Token Budget (I33 Relative Budget):
#   - image_size=64, min_patch_size=4
#   - max_patches = (64/4)^2 = 256
#   - coverage: 1% → K_min=3 (→ 8), 5% → K_max=13
# ============================================================================

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 200 \
  --dim 320 \
  --depth 10 \
  --heads 8 \
  --pool cls \
  --ffn-type swiglu_level \
  --tokenizer-type streaming_v3 \
  --min-patch-size 4 \
  --token-coverage-min 0.01 \
  --token-coverage-max 0.05 \
  --batch-size 192 \
  --accum-steps 3 \
  --num-workers 4 \
  --lr 2.7e-4 \
  --weight-decay 0.1 \
  --warmup-epochs 10 \
  --dropout 0.2 \
  --emb-dropout 0.15 \
  --drop-path 0.2 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.5 \
  --splitter-temp-warmup 10 \
  --temp-schedule cosine \
  --lca-temperature 1.5 \
  --use-amp \
  --gradient-checkpoint \
  --channels-last \
  --tf32 \
  --compile \
  --compile-mode max-autotune \
  --gradient-clip 1.0 \
  --patience 30 \
  --min-delta 0.001 \
  --exp-name tiny_imagenet_320d_10l_bs192x3_ep200
