#!/bin/bash
# ============================================================================
# CUB-200-2011 最优训练脚本 - Fractal ViT 细粒度分类
# RTX 4070 Laptop (8GB VRAM) 优化配置 - 修正版
# ============================================================================
#
# 数学形式化分析 (2026-01-15 修正)
# =================================
#
# [1] 数据集分析
#     ------------------------------------------------------------------
#     N_train = 5,994
#     N_test  = 5,794
#     C       = 200 类
#     图像尺寸 = 224×224×3
#     样本/类 = 30.0 (细粒度分类，每类样本少)
#
# [2] VRAM 预算修正 (224×224 大图像)
#     ------------------------------------------------------------------
#     ⚠️ 关键问题: 224×224 图像的 Attention 矩阵消耗大量 VRAM
#
#     Attention 矩阵 per sample:
#       (H × K × K) × 4 bytes = (8 × 256 × 256) × 4 = 2.1 MB/sample
#
#     实际 PyTorch VRAM 消耗远超理论值 (2-4×)
#
#     修正方案:
#       - batch_size = 24 (实际可用)
#       - accum_steps = 8 (梯度累积)
#       - effective_batch = 24 × 8 = 192
#
# [3] 模型架构 (Double Descent 优化)
#     ------------------------------------------------------------------
#     dim=256, depth=8: P ≈ 8.05M, P/N ≈ 1343×
#
# [4] Token 约束 (信息论推导)
#     ------------------------------------------------------------------
#     K_min = 12: 2 × ceil(log2(200)/1.5)
#     K_max = 196: 限制以节省 VRAM (降低 Attention 矩阵大小)
#     num_scales = 4: 4×4, 8×8, 16×16, 32×32
#
# [5] 学习率 (Linear Scaling + 小数据集修正)
#     ------------------------------------------------------------------
#     lr = 2.6e-4 (不变，因为 effective_batch 仍为 192)
#
# [6] 训练动态
#     ------------------------------------------------------------------
#     实际 steps_per_epoch = ceil(5994/24) = 250
#     但由于 accum_steps=8，优化器步数 = 250/8 ≈ 31/epoch
#     预计训练时间: ~45-60 分钟 (accum 增加 overhead)
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py \
  --dataset cub200 \
  --epochs 100 \
  --dim 256 \
  --depth 8 \
  --heads 8 \
  --dim-head 32 \
  --max-level 3 \
  --pool cls \
  --ffn-type swiglu_level \
  --tokenizer-type streaming_v3 \
  --num-scales 4 \
  --K-min 12 \
  --K-max 196 \
  --batch-size 24 \
  --accum-steps 8 \
  --lr 2.6e-4 \
  --warmup-epochs 5 \
  --lr-scheduler cosine \
  --min-lr 1e-6 \
  --weight-decay 0.20 \
  --dropout 0.20 \
  --emb-dropout 0.15 \
  --drop-path 0.18 \
  --label-smoothing 0.15 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 0.6 \
  --mix-prob 0.5 \
  --compile \
  --channels-last \
  --gradient-checkpointing \
  --amp \
  --log-interval 10 \
  --save-interval 20 
