#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop)
# Mathematical derivation: dim=384, depth=8, batch=64, lr=3e-4, epochs=150
# I103-5: batch=64*3 测试显存稳定性

set -e

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 150 \
  --dim 384 \
  --depth 8 \
  --heads 6 \
  --pool cls \
  --ffn-type swiglu_level \
  --tokenizer-type streaming_v3 \
  --min-patch-size 4 \
  --batch-size 64 \
  --accum-steps 3 \
  --num-workers 4 \
  --lr 3e-4 \
  --weight-decay 0.08 \
  --warmup-epochs 15 \
  --dropout 0.15 \
  --emb-dropout 0.1 \
  --drop-path 0.15 \
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
  --splitter-temp-warmup 20 \
  --lca-temperature 1.5 \
  --use-amp \
  --gradient-checkpoint \
  --channels-last \
  --tf32 \
  --compile \
  --gradient-clip 1.0 \
  --patience 25 \
  --min-delta 0.001 
