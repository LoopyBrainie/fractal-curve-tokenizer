#!/bin/bash
# ============================================================================
# Tiny-ImageNet I21 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-11) - I21 深度平衡版本
# ================================================
#
# 1. 模型容量计算
#    Tiny-ImageNet: N_train = 100,000 samples, C = 200 classes
#    
#    经验法则 (考虑正则化):
#      P_target ∈ [N/10, N/5] × regularization_factor = [15M, 30M]
#    
#    选择: dim=320, depth=12, heads=8 → θ ≈ 20M
#    验证:
#      P_transformer = 12 × (4×320² + 3×320×1280) ≈ 19.7M
#      P_total ≈ 20M ✓
#
# 2. I21 深度平衡机制 (自动启用)
#    β: Log-Compensation Bias: b_d = log(N_total / N_d)
#    δ: Subset Softmax: 梯度增强 N/K ≈ 2.7×
#    ε: Depth KL Loss: L = 0.1 × D_KL(π || Uniform)
#
# 3. 几何极限约束
#    image_size=64, num_scales=4 → max_depth=3
#    候选数: N = 1 + 4 + 16 + 64 = 85
#    目标 token 数: target_tokens=48 (压缩比 ~5:1)
#    弹性预算 Dead Zone: [24, 96]
#
# 4. 学习率缩放 (Linear Scaling Rule)
#    lr = lr_base × (B / 256) = 5e-4 × (192/256) = 3.75e-4
#
# 5. 退火策略 (P10-11 验证)
#    温度: T(t) = 1.0 × (0.3)^(t/S_post), T_end ≥ 0.3
#    Warmup: 5 epochs
#
# 6. VRAM 预算: ~3 GB << 8 GB ✓ (with AMP + Checkpoint)
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  \
  `# === 模型架构 (~20M 参数 - I21 优化容量) ===` \
  `# 验证: dim=320, depth=12 → P ≈ 20M` \
  --dim 320 \
  --depth 12 \
  --heads 8 \
  --dim-head 40 \
  --max-level 3 \
  --pool cls \
  --ffn-type swiglu_level \
  \
  `# === Scheme D: GumbelTopKSplitter 配置 ===` \
  `# num_scales=4 → max_depth=3 → 候选数=85` \
  `# I21 深度平衡: 自动启用 (constants.py)` \
  --tokenizer-type streaming_v3 \
  --num-scales 4 \
  --target-tokens 48 \
  --split-gamma 0.85 \
  --enforce-balance \
  \
  `# === 学习率: lr = 5e-4 × (192/256) = 3.75e-4 ===` \
  --batch-size 192 \
  --num-workers 4 \
  --lr 3.75e-4 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  \
  `# === 正则化 (较强 - 防止过拟合) ===` \
  --dropout 0.15 \
  --emb-dropout 0.1 \
  --drop-path 0.15 \
  --label-smoothing 0.1 \
  \
  `# === 数据增强 (Mixup + CutMix + Progressive) ===` \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --progressive-aug \
  \
  `# === 辅助损失配置 ===` \
  `# Soft Entropy: 最大化尺度多样性` \
  `# Elastic Budget: Dead Zone [24, 96]` \
  `# I21 Depth KL: 自动启用 (DEPTH_KL_WEIGHT=0.1)` \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-N-min 24 \
  --elastic-N-max 96 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --elastic-lambda-collapse 1.0 \
  \
  `# === Splitter 温度退火 (P10-11 安全下界) ===` \
  `# 公式: T(t) = T_start × (T_end / T_start)^(t / S_post)` \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.3 \
  --splitter-temp-warmup 5 \
  \
  `# === 性能优化 (4070 Laptop) ===` \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --accum-steps 1 \
  --gradient-clip 1.0 \
  \
  `# === 早停策略 ===` \
  --patience 15 \
  --min-delta 0.001 
