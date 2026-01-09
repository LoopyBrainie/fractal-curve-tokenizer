#!/bin/bash
# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-10) - Scheme D 对齐版本
# ================================================
#
# 1. 模型容量 vs 数据集规模
#    Tiny-ImageNet: N = 100,000 samples, C = 200 classes
#    选择: dim=256, depth=8, heads=8 → θ ≈ 4.2M (防止过拟合)
#
# 2. Scheme D (GumbelTopKSplitter) 架构
#    核心公式: selected = TopK(logits + Gumbel(0,1), K)
#    优势: 100% 梯度覆盖, 硬 K 约束消除死锁
#
# 3. 几何极限约束
#    image_size=64, num_scales=4 → max_depth=3
#    候选数 N = 1 + 4 + 16 + 64 = 85
#    弹性预算: [24, 80] 在候选范围内
#
# 4. 退火策略 (I18-2 安全下界)
#    温度: T(t) = 1.0 · (0.3)^(t/total), T_end ≥ 0.3
#    Warmup: 10 epochs (10% of training)
#
# 5. VRAM 预算: ~1.0 GB << 8 GB ✓ (with AMP + Checkpoint)
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  \
  `# === 模型架构 (4.2M 参数 - 保守容量防止过拟合) ===` \
  --dim 256 \
  --depth 8 \
  --heads 8 \
  --dim-head 32 \
  --max-level 3 \
  --pool cls \
  --ffn-type swiglu_level \
  \
  `# === Scheme D: GumbelTopKSplitter 配置 ===` \
  --tokenizer-type streaming_v3 \
  --num-scales 4 \
  --splitter-token-budget 48 \
  --split-tau0 0.0 \
  --split-gamma 0.7 \
  --enforce-balance \
  \
  `# === 训练配置 (batch_size=192) ===` \
  --batch-size 192 \
  --num-workers 4 \
  --lr 1.2e-3 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  \
  `# === 正则化 (较强 - 防止过拟合) ===` \
  --dropout 0.15 \
  --emb-dropout 0.1 \
  --drop-path 0.15 \
  --label-smoothing 0.1 \
  \
  `# === 数据增强 (Mixup + CutMix) ===` \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  \
  `# === P10 辅助损失 ===` \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-N-min 24 \
  --elastic-N-max 80 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --elastic-lambda-collapse 1.0 \
  \
  `# === Splitter 温度退火 (I18-2 安全下界) ===` \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.3 \
  --splitter-temp-warmup 10 \
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
