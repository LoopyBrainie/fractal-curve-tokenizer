#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop 8GB)
# Mathematical Formalization (2026-01-26): dim=256, depth=8, batch=192
# Optimized for 200 epochs with maximum throughput

set -e

# ============================================================================
# 数学形式化验证结果 - I107-1 更新
# ============================================================================
# VRAM: 8GB RTX 4070 Laptop
# Configuration: dim=256, depth=8, heads=8, mlp_dim=512, batch=192
#
# 关键发现: batch_size=192 完全可行
# - 实际显存使用率仅 13.1% (约 1073 MB)
# - Fractal ViT 动态 tokenization 相比标准 ViT 节省 ~40× attention 显存
# - gradient checkpoint 有效压缩激活值显存
#
# Memory Breakdown (checkpoint + amp + channels-last + batch=192):
#   - Model params (FP16): ~25 MB (12.9M params)
#   - Optimizer states (FP32): ~50 MB
#   - Activations (CP): ~400-500 MB
#   - Total: ~1073 MB << 8GB VRAM (margin: 7.1 GB)
#
# Token Budget (I33 Relative Budget):
#   - image_size=64, min_patch_size=4
#   - max_patches = (64/4)^2 = 256
#   - coverage: 1% → K_min=8, 5% → K_max=64
#   - Avg tokens: ~32-48 per image
# ============================================================================

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 200 \
  --dim 256 \
  --depth 8 \
  --heads 8 \
  --mlp-dim 512 \
  --max-depth 6 \
  --pool cls \
  --ffn-type swiglu_level \
  --min-patch-size 4 \
  --token-coverage-min 0.01 \
  --token-coverage-max 0.25 \
  --elastic-coverage-min 0.03 \
  --elastic-coverage-max 0.25 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --batch-size 192 \
  --num-workers 8 \
  --lr 5e-4 \
  --weight-decay 0.15 \
  --warmup-epochs 15 \
  --dropout 0.25 \
  --drop-path 0.25 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.05 \
  --include-elastic-budget \
  --elastic-coverage-min 0.03 \
  --elastic-coverage-max 0.25 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --use-amp \
  --gradient-checkpoint \
  --channels-last \
  --gradient-clip 1.0 \
  --patience 20 \
  --min-delta 0.001 \ 
  --compile