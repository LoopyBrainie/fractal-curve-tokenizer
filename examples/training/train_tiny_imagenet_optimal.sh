#!/bin/bash
# Tiny-ImageNet 优化训练脚本
# 数学推导参数: 31.6M, ~5.3GB VRAM, batch_size=128
# 预期: 55-60% Top-1 Accuracy (100 epochs)

uv run python examples/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  \
  `# 模型架构 (31.6M 参数, 匹配数据集容量)` \
  --dim 384 \
  --depth 12 \
  --heads 6 \
  --mlp-dim 1536 \
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
  `# 训练配置 (batch_size=128, lr 线性缩放)` \
  --batch-size 128 \
  --num-workers 4 \
  --learning-rate 7e-4 \
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
  --exp-name tiny_imagenet_optimal_384d_12l_100ep
