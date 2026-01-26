#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop)
# Mathematical Formalization (2026-01-26): dim=256, depth=8, batch=128
# Optimized for 100 epochs with maximum throughput

set -e

# ============================================================================
# 数学形式化验证结果 - I107-1 分析
# ============================================================================
# VRAM: 8GB RTX 4070 Laptop
# Configuration: dim=256, depth=8, heads=8, batch=128
#
# 关键发现: batch_size=192 不可行
# - Fractal ViT 自适应 tokenization 不减少激活值计算
# - 前向传播必须处理所有候选区域
# - 激活值显存由输入分辨率和模型尺寸决定，与 K 无关
#
# Memory Breakdown (checkpoint + amp + channels-last):
#   - Model params (FP16): ~17 MB (8.5M params)
#   - Optimizer states (FP32): ~34 MB
#   - Activations: ~3.7 GB (B=128, S=64, D=256, L=8, alpha=0.45)
#   - Total: ~3.8 GB << 8GB VRAM (margin: 4.2 GB)
#
# Effective Batch = 128 × 1 accum = 128
#
# Token Budget (I33 Relative Budget):
#   - image_size=64, min_patch_size=8
#   - max_patches = (64/8)^2 = 64
#   - coverage: 1% → K_min=8, 5% → K_max=32
# ============================================================================

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  --dim 256 \
  --depth 8 \
  --heads 8 \
  --pool cls \
  --ffn-type swiglu_level \
  --min-patch-size 8 \
  --token-coverage-min 0.01 \
  --token-coverage-max 0.05 \
  --batch-size 128 \
  --num-workers 4 \
  --lr 5e-4 \
  --weight-decay 0.15 \
  --warmup-epochs 10 \
  --dropout 0.25 \
  --emb-dropout 0.15 \
  --drop-path 0.25 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --splitter-temp-start 2.0 \
  --splitter-temp-end 0.5 \
  --splitter-temp-warmup 5 \
  --temp-schedule cosine \
  --lca-temperature 1.5 \
  --use-amp \
  --gradient-checkpoint \
  --channels-last \
  --compile \
  --gradient-clip 1.0 \
  --patience 15 \
  --min-delta 0.001 \
  --exp-name tiny_imagenet_256d_8l_bs128_ep100
