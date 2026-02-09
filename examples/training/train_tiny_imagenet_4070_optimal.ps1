#!/usr/bin/env python3
"""
Fractal ViT Training Script - Tiny-ImageNet 200 Epochs Optimal Configuration
Mathematical Formalization (2026-02-09 Update)

GPU: RTX 4070 Laptop (8GB VRAM)
Target: 200 epochs, batch=192

Key Optimizations:
- dim=384, L=8: 22M params (optimal for Tiny-ImageNet)
- tokenizer_dropout=0.0: Deterministic tokenization
- transformer_dropout=0.25 + drop_path=0.25: Strong regularization
- focal_gamma=2.5: Hard/easy sample ratio 243x
- soft_entropy + elastic_budget: Depth diversity + token budget

Expected Performance:
- Top-1 Accuracy: 55-65%
- Training Time: ~20-30 hours
- VRAM Peak: ~5.5 GB
"""

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 200 `
  --dim 384 `
  --num-layers 8 `
  --heads 6 `
  --mlp-dim 1536 `
  --min-patch-size 4 `
  --token-coverage-min 0.01 `
  --token-coverage-max 0.20 `
  --K-min-abs 8 `
  --batch-size 192 `
  --lr 1e-04 `
  --weight-decay 0.05 `
  --tokenizer-dropout 0.0 `
  --transformer-dropout 0.25 `
  --emb-dropout 0.0 `
  --drop-path 0.25 `
  --label-smoothing 0.1 `
  --gradient-clip 1.0 `
  --warmup-epochs 10 `
  --mixup-alpha 0.4 `
  --cutmix-alpha 0 `
  --focal-gamma 2.5 `
  --include-soft-entropy `
  --soft-entropy-mode maximize `
  --soft-entropy-weight 0.1 `
  --include-elastic-budget `
  --elastic-coverage-min 0.03 `
  --elastic-coverage-max 0.25 `
  --elastic-lambda-over 0.1 `
  --elastic-lambda-under 0.01 `
  --quota-learnable enable `
  --quota-entropy-weight 0.01 `
  --use-amp `
  --gradient-checkpoint `
  --compile `
  --channels-last `
  --patience 25 `
  --exp-name tiny_imagenet_384d_8l_bs192_ep200_v1
"@

Invoke-Expression $script
