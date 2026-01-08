#!/bin/bash
# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-08)
# ===========================
#
# 1. 模型容量 vs 数据集规模
#    Tiny-ImageNet: N = 100,000 samples, C = 200 classes
#    选择: dim=320, depth=12, heads=5 → θ = 21.14M
#
# 2. 几何极限约束 (I16-2)
#    image_size=64, min_patch_size=4 → grid_size=16
#    max_depth = 4, num_scales = 5, max_tokens = 341
#
# 3. 学习率缩放 (Linear Scaling Rule)
#    lr_base = 5e-4 @ batch_size=64
#    lr = 5e-4 × (196/64) × 0.8 = 1.2e-3
#
# 4. VRAM 预算: ~0.9 GB << 8 GB ✓ (with AMP + Checkpoint)
#
# 5. I10-19 连续松弛: 启用，解决 Splitter 崩塌问题
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  --dim 320 \
  --depth 12 \
  --heads 5 \
  --dim-head 64 \
  --max-level 4 \
  --pool cls \
  --ffn-type swiglu_level \
  --tokenizer-type streaming_v3 \
  --num-scales 5 \
  --target-tokens 80 \
  --split-tau0 0.0 \
  --split-gamma 0.6 \
  --enforce-balance \
  --batch-size 196 \
  --num-workers 4 \
  --lr 1.2e-3 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  --dropout 0.1 \
  --emb-dropout 0.1 \
  --drop-path 0.15 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.8 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-N-min 48 \
  --elastic-N-max 192 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --elastic-lambda-collapse 1.0 \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.3 \
  --splitter-temp-warmup 5 \
  --use-continuous-relaxation \
  --continuous-max-depth 3 \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --accum-steps 1 \
  --gradient-clip 1.0 \
  --patience 15 \
  --min-delta 0.001 \
  --exp-name tiny_imagenet_4070_optimal_320d_12l_bs196
