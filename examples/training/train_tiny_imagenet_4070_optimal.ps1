# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop 8GB)
# ============================================================================
# 数学形式化分析 (2026-01-29 更新)
#
# [EXPLICITLY CALCULATED PARAMETERS]
#   --dim 320        : From memory constraint M_total ≤ 763 MB
#   --depth 12       : Optimal for Tiny-ImageNet capacity
#   --heads 8        : Required: dim_head = dim/heads = 320/8 = 40
#   --mlp-dim 1280   : SwiGLU formula: 4 × dim
#   --batch-size 192 : Memory budget: 763 MB available
#   --token-coverage-min 0.02 : 2% coverage = 5 tokens (K_min)
#   --token-coverage-max 0.05 : 5% coverage = 12 tokens (K_max)
#
# [EXPERIMENTAL FEATURES]
#   --quota-learnable        : Enable learnable token quota
#   --freeze-tokenizer       : Freeze tokenizer for 10 epochs
#   --elastic-coverage-*     : Elastic budget regularization
#
# [OPTIMIZATION FLAGS]
#   --use-amp, --gradient-checkpoint, --compile, --channels-last
#
# Model: 34.56M params | Memory: 763 MB | Tokens: 5-12 (2%-5% coverage)
# ============================================================================

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 100 `
  --dim 320 `
  --depth 12 `
  --heads 8 `
  --mlp-dim 1280 `
  --min-patch-size 4 `
  --token-coverage-min 0.02 `
  --token-coverage-max 0.05 `
  --freeze-tokenizer `
  --freeze-tokenizer-epochs 10 `
  --elastic-coverage-min 0.02 `
  --elastic-coverage-max 0.05 `
  --batch-size 192 `
  --num-workers 4 `
  --include-soft-entropy `
  --include-elastic-budget `
  --use-amp `
  --gradient-checkpoint `
  --compile `
  --channels-last `
  --exp-name tiny_imagenet_320d_12l_bs192_ep100_v3
"@

# 执行训练脚本
Invoke-Expression $script
