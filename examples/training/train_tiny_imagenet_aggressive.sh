#!/bin/bash
# Tiny-ImageNet 激进优化训练脚本
# 优化目标: 最大化 GPU 利用率 (>90%), 充分利用 8GB 显存
# 预期: batch_size=384, ~7.5GB VRAM, 15-20 it/s, 55-60% Top-1 Accuracy

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  \
  `# 模型架构 (31.6M 参数, 与 optimal 相同)` \
  --dim 384 \
  --depth 12 \
  --heads 6 \
  --dim-head 64 \
  --max-level 4 \
  --pool cls \
  --ffn-type swiglu_level \
  \
  `# LearnableSplitter 配置` \
  --tokenizer-type streaming_v3 \
  --target-tokens 96 \
  --split-tau0 0.0 \
  --split-gamma 0.5 \
  \
  `# 激进训练配置 (batch_size=384, 3x 原配置)` \
  --batch-size 384 \
  --num-workers 12 \
  --lr 1.2e-3 \
  --weight-decay 0.05 \
  --warmup-epochs 10 \
  \
  `# 正则化 (适中强度)` \
  --dropout 0.1 \
  --emb-dropout 0.1 \
  --drop-path 0.1 \
  --label-smoothing 0.1 \
  \
  `# 数据增强 (Mixup + CutMix)` \
  --mixup-alpha 0.8 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  \
  `# P10 优化 (软熵 + 弹性预算)` \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-N-min 64 \
  --elastic-N-max 128 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --elastic-lambda-collapse 1.0 \
  \
  `# 性能优化 (AMP + Compile + Channels-Last + Checkpointing)` \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --accum-steps 1 \
  --gradient-clip 1.0 \
  \
  `# 早停策略` \
  --patience 15 \
  --min-delta 0.001 \
  \
  `# 实验命名` \
  --exp-name tiny_imagenet_aggressive_384d_12l_bs384

# ============================================================================
# 性能对比
# ============================================================================
# 配置           | batch_size | VRAM   | 速度     | GPU 占用 | 训练时间
# --------------------------------------------------------------------------
# optimal        | 128        | ~5.3GB | 5 it/s   | 40%      | ~40h
# aggressive     | 384        | ~7.5GB | 15-20 it/s| >90%    | ~13h
# max (如果OOM)  | 512        | ~8.0GB | 20-25 it/s| >95%    | ~10h
# ============================================================================

# 数学推导:
# 1. Learning Rate Scaling: lr = 7e-4 * sqrt(384/128) = 7e-4 * sqrt(3) = 1.2e-3
# 2. VRAM 估算: 2GB(model) + 3GB(activation*3x) + 2GB(optimizer) + 0.5GB(batch*3x) = 7.5GB
# 3. 训练时间: 40h / 3 = 13.3h (理想情况，实际约 15h 考虑编译开销)
# 4. GPU 利用率: batch_size 增加 3x → 计算密度提升 → 预期 GPU >90%
