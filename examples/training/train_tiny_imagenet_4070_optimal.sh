#!/bin/bash
# ============================================================================
# Tiny-ImageNet I78 + I31 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-18) - I78 动态分辨率 + I31 面积编码版本
# =============================================================================
#
# 1. 模型架构参数计算
#    --------------------
#    Tiny-ImageNet: N_train = 100,000 samples, C = 200 classes
#
#    参数公式:
#      P_total = depth × (6 × dim² + 2 × dim × mlp_dim) + tokenizer + head
#             = 12 × (6×384² + 2×384×1536) + 0.46M + 0.5M
#             ≈ 27M ✓
#
#    Attention QKV: 4 × 384² = 589,824
#    FFN SwiGLU: 2 × 384 × 1536 = 1,179,648
#    Per layer: ~1.77M, 12 layers: ~21.2M + 0.96M ≈ 27M
#
# 2. I78 动态分辨率支持
#    --------------------
#    image_size=128 (原始尺寸，无需 resize)
#    min_patch_size=4
#    max_depth = log2(128/4) = 5 (自动计算)
#    候选区域: 1+4+16+64+256+1024 = 1365 (6 尺度)
#    K_max=64, K_min=16 (4:1 压缩比)
#
# 3. I31 面积编码配置 (2026-01-18)
#    -----------------------------
#    形状-尺度编码补充离散 Level 的几何信息:
#      use_area_encoding=True (位置编码增强)
#      fourier_levels=4 (傅里叶特征级别数)
#
# 4. 学习率缩放 (Linear Scaling Rule)
#    ---------------------------------
#    lr = 5e-4 × (192/256) = 3.2e-4
#
# 5. VRAM 预算: ~1300 MB << 8GB ✓ (FP16 + AMP + Checkpoint)
#
# 6. 退火策略
#    ---------
#    温度: T(t) = T_end + (T_start - T_end) × (1 + cos(πt/T)) / 2
#    T_start=1.0, T_end=0.5, Warmup: 10 epochs
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  --dim 384 \
  --depth 12 \
  --heads 8 \
  --dim-head 48 \
  --mlp-dim 1536 \
  --pool cls \
  --ffn-type swiglu_level \
  --tokenizer-type streaming_v3 \
  --min-patch-size 4 \
  --K-min 16 \
  --K-max 64 \
  --use-area-encoding \
  --fourier-levels 4 \
  --batch-size 192 \
  --num-workers 4 \
  --lr 3.2e-4 \
  --weight-decay 0.1 \
  --warmup-epochs 10 \
  --dropout 0.15 \
  --emb-dropout 0.1 \
  --drop-path 0.2 \
  --label-smoothing 0.1 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-N-min 16 \
  --elastic-N-max 80 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --elastic-lambda-collapse 1.0 \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.5 \
  --splitter-temp-warmup 10 \
  --lca-temperature 1.5 \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --accum-steps 1 \
  --gradient-clip 1.0 \
  --patience 15 \
  --min-delta 0.001 
