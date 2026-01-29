# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop 8GB)
# ============================================================================
#
# 数学形式化分析 (2026-01-29 更新) - 基于完整数学推导
# Target: RTX 4070 Laptop (8GB VRAM), 100 epochs, batch_size=192
#
# ============================================================================
# 推荐配置: dim=320, depth=12, heads=8
# - 参数量: 34.56M
# - 显存: 763 MB (88.3% 安全边际)
# - Token coverage: 2%-5% (5-12 tokens)
#
# 显存分解 (batch=192):
#   - 参数 (FP32): 131.8 MB
#   - 梯度 (FP32): 131.8 MB
#   - 优化器 (FP32): 263.7 MB
#   - 激活 (FP16): 36.0 MB
#   - CUDA overhead: 200 MB
#   - Total: 763 MB
#
# 模型参数公式: P = L × (16 × d²) + 2.5 × d × C
#   P = 12 × (16 × 320²) + 2.5 × 320 × 200 = 34.56M params
#
# ============================================================================
# Token Budget (I33 相对预算)
# ---------------------------------
# image_size=64, min_patch_size=4
# max_patches = (64/4)² = 256
#
# coverage_min=0.02 (2%): K_min = max(4, 256×0.02) = 5
# coverage_max=0.05 (5%): K_max = 256×0.05 = 12
#
# ============================================================================

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 100 `
  --dim 320 `
  --depth 12 `
  --heads 8 `
  --mlp-dim 1280 `
  --pool weighted `
  --ffn-type swiglu_level `
  --min-patch-size 4 `
  --token-coverage-min 0.02 `
  --token-coverage-max 0.05 `
  --quota-learnable `
  --freeze-tokenizer `
  --freeze-tokenizer-epochs 10 `
  --elastic-coverage-min 0.02 `
  --elastic-coverage-max 0.05 `
  --elastic-lambda-over 0.1 `
  --elastic-lambda-under 0.01 `
  --batch-size 192 `
  --num-workers 4 `
  --lr 5e-4 `
  --weight-decay 0.15 `
  --warmup-epochs 10 `
  --dropout 0.25 `
  --drop-path 0.25 `
  --label-smoothing 0.1 `
  --mixup-alpha 0.4 `
  --cutmix-alpha 1.0 `
  --mixup-prob 0.5 `
  --include-soft-entropy `
  --soft-entropy-mode maximize `
  --soft-entropy-weight 0.1 `
  --include-elastic-budget `
  --use-amp `
  --gradient-checkpoint `
  --compile `
  --channels-last `
  --gradient-clip 1.0 `
  --patience 20 `
  --min-delta 0.001 `
  --exp-name tiny_imagenet_320d_12l_bs192_ep100_v3
"@

# 执行训练脚本
Invoke-Expression $script
