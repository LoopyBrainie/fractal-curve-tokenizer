#!/bin/bash
# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-21) - 基于第一性原理推导
# ============================================================================
#
# 1. 模型架构参数计算 (I36 激进优化)
#    --------------------
#    Tiny-ImageNet: N_train = 100,000 samples, C = 200 classes
#
#    参数公式 (SwiGLU FFN):
#      P_total = depth * 16 * dim^2 + 2.5 * dim * num_classes
#      mlp_dim = 4 * dim (自动计算)
#
#    激进配置 (dim=448, depth=8, heads=7):
#      mlp_dim = 4 * 448 = 1792
#      P_total = 8 * 16 * 448^2 + 2.5 * 448 * 200 = 45.44M (实际: 31.58M)
#
#    P/N 比率: 31.58M / 100K = 315.8 (ViT 可接受范围: [150, 500])
#
# 2. Tokenizer 配置优化 (I36)
#    ----------------
#    image_size=64 (Tiny-ImageNet 原始尺寸，固定)
#    min_patch_size=4
#    max_depth = log2(64/4) = 4
#    候选区域: 1+4+16+64+256 = 341 (5 尺度)
#
#    I36 新常量 (constants.py):
#      K_COVERAGE_BASE = 0.12 (12% 覆盖率)
#      K_COVERAGE_MAX_HARD = 0.25 (25% 硬上限)
#      K_MAX_SAMPLE_RATIO = 0.25 (25% 采样比例)
#      K_MIN_SAMPLE_RATIO = 0.03 (3% 最小比例)
#
#    动态 Token 数:
#      K_min = max(8, 0.03 * 341) = 11
#      K_max = min(4096, 0.25 * 341) = 85
#      K_avg ≈ 40 (覆盖率 ~12%)
#
# 3. 学习率缩放 (Linear Scaling Rule)
#    ---------------------------------
#    lr_base = 3e-4 @ batch_size=256
#    lr = 3e-4 * (192/256) = 2.25e-4
#
# 4. VRAM 预算 (8GB - RTX 4070 Laptop)
#    ----------------------------------
#    混合精度模型: 316 MB (31.58M * 10 bytes)
#    总计(含开销): ~1.5 GB << 8GB ✓
#
# 5. torch.compile 兼容性说明
#    -----------------------
#    ⚠️ torch.compile 已禁用 - 与动态 token 数量不兼容
#    原因: CUDA 内存分配器与 torch.compile 存在已知问题
#    替代方案: gradient-checkpoint + channels-last + AMP
#    性能损失: 约 10-15% 但更稳定
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  --dim 448 \
  --depth 8 \
  --heads 7 \
  --pool cls \
  --ffn-type swiglu_level \
  --tokenizer-type streaming_v3 \
  --min-patch-size 4 \
  --batch-size 192 \
  --num-workers 4 \
  --lr 2.25e-4 \
  --weight-decay 0.08 \
  --warmup-epochs 10 \
  --dropout 0.15 \
  --emb-dropout 0.1 \
  --drop-path 0.12 \
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
  --lca-temperature 1.5 \
  --use-amp \
  --gradient-checkpoint \
  --channels-last \
  --tf32 \
  --accum-steps 1 \
  --gradient-clip 1.0 \
  --patience 15 \
  --min-delta 0.001 \
  --exp-name tiny_imagenet_448d_8l_bs192
