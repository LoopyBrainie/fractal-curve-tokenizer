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
#    双倍下降理论最优区间 (Belkin et al., 2019):
#      P_target ∈ [N/2, 2N] = [50K, 200K] (可扩展到 ~30M)
#    
#    选择: dim=384, depth=12, heads=8 → θ ≈ 30M
#    验证:
#      P_transformer = 12 × (4×384² + 3×384×1536) ≈ 28M
#      P_total ≈ 30M ✓
#
# 2. I21 深度平衡机制 (自动启用)
#    β: Log-Compensation Bias: b_d = log(N_total / N_d)
#    δ: Subset Softmax: 梯度增强 N/K ≈ 2.7×
#    ε: Depth KL Loss: L = 0.1 × D_KL(π || Uniform)
#
# 3. 几何极限约束
#    image_size=64, num_scales=4 → max_depth=3
#    候选数: N = 1 + 4 + 16 + 64 = 85
#    K_min=16 (信息论: ⌈log₂(200)/1.5⌉×2), K_max=64 (4:1 压缩比)
#    弹性预算 Dead Zone: [16, 80]
#
# 4. 学习率缩放 (Linear Scaling Rule)
#    lr_base = 5e-4 @ batch_size=256 (AdamW + ViT 标准)
#    lr = 5e-4 × (192/256) × 0.85 ≈ 3.2e-4 (Mixup 补偿)
#
# 5. 退火策略 (Jang et al., 2017)
#    温度: T(t) = T_end + (T_start - T_end) × (1 + cos(πt/T)) / 2
#    T_start=1.0, T_end=0.5 (保持梯度流)
#    Warmup: 5 epochs
#
# 6. VRAM 预算: ~3 GB << 8 GB ✓ (with AMP + Checkpoint)
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  \
  `# === 模型架构 (~30M 参数 - 数学推导最优) ===` \
  `# 验证: dim=384, depth=12 → P ≈ 30M` \
  --dim 384 \
  --depth 12 \
  --heads 8 \
  --dim-head 48 \
  --max-level 3 \
  --pool cls \
  --ffn-type swiglu_level \
  \
  `# === Scheme D: GumbelTopKSplitter 配置 ===` \
  `# num_scales=4 → max_depth=3 → 候选数=85` \
  `# K_min=16 (信息论: ⌈log₂(200)/1.5⌉×2), K_max=64 (4:1 压缩比)` \
  `# I21 深度平衡: 自动启用 (constants.py)` \
  --tokenizer-type streaming_v3 \
  --num-scales 4 \
  --K-min 16 \
  --K-max 64 \
  \
  `# === 学习率: lr = 5e-4 × (192/256) × 0.85 ≈ 3.2e-4 (Mixup 补偿) ===` \
  --batch-size 192 \
  --num-workers 4 \
  --lr 3.2e-4 \
  --weight-decay 0.1 \
  --warmup-epochs 10 \
  \
  `# === 正则化 (适应 30M 模型过拟合风险) ===` \
  --dropout 0.15 \
  --emb-dropout 0.1 \
  --drop-path 0.2 \
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
  `# Elastic Budget: Dead Zone [12, 80]` \
  `# I21 Depth KL: 自动启用` \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-N-min 16 \
  --elastic-N-max 80 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --elastic-lambda-collapse 1.0 \
  \
  `# === Splitter 温度退火 (cosine, T_end=0.5 保持梯度) ===` \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.5 \
  --splitter-temp-warmup 5 \
  \
  `# === LCA Hilbert Bias (P6-2, τ=1.5) ===` \
  --lca-temperature 1.5 \
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
