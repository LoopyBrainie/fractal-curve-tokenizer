#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop)
# Mathematical derivation: dim=448, depth=8, batch=128, lr=1.5e-4

set -e

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 100 \
  --dim 448 \
  --depth 8 \
  --heads 7 \
  --pool cls \
  --ffn-type swiglu_level \
  --tokenizer-type streaming_v3 \
  --min-patch-size 4 \
  --batch-size 128 \
  --num-workers 4 \
  --lr 1.5e-4 \
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
  --no-prefetch 
