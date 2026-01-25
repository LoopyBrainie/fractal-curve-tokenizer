#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop)
# Mathematical derivation: dim=384, depth=8, batch=36x3=108, lr=1.25e-4, epochs=150
# Optimizations: gradient-checkpoint + channels-last + AMP + TF32 (compile disabled)
# Note: Using gradient accumulation to avoid OOM

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
  --batch-size 60 \
  --num-workers 4 \
  --lr 2.1e-4 \
  --accum-steps 3 \
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
  --gradient-clip 1.0 \
  --patience 25 \
  --min-delta 0.001 
