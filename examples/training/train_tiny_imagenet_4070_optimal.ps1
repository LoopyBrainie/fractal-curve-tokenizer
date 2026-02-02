# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop 8GB)
# ============================================================================
# 数学形式化分析 (2026-02-02 更新)
#
# [内存约束模型]
#   M_total = M_params + M_gradients + M_optimizer + M_activations
#   M_params (FP16):     0.11 GB
#   M_gradients (FP32):  0.44 GB
#   M_optimizer (FP32):  0.88 GB
#   M_activations:       0.07 GB (checkpoint)
#   总计:               1.65 GB (安全余量: 4.3 GB)
#
# [架构参数]
#   --dim 512         : 最大化表达力与内存平衡
#   --num-layers 16   : 100 epochs 最佳深度
#   --heads 8         : dim_head = 512/8 = 64
#   --mlp-dim 2048    : mlp_ratio = 4.0 (SwiGLU)
#
# [Tokenizer 参数]
#   --patch-size 8
#   --min-patch-size 4
#   --token-coverage-min 0.02 : α = 2%
#   --token-coverage-max 0.50 : β = 50%
#   K (token 范围): [16, 92] (64×64 图像)
#
# [训练参数]
#   --batch-size 192  : 目标批次 (启用 checkpoint)
#   --lr 2.4e-4       : √(192/32) × 1e-4
#   --weight-decay 0.038
#   --dropout 0.15    : 100 epochs 正则化
#   --drop-path 0.20  : 随机深度
#
# 模型: 117.54M 参数 | 显存: 1.65 GB | FLOPs: 6.34e9/样本
# ============================================================================

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 100 `
  --dim 512 `
  --num-layers 16 `
  --heads 8 `
  --mlp-dim 2048 `
  --min-patch-size 4 `
  --token-coverage-min 0.02 `
  --token-coverage-max 0.50 `
  --batch-size 192 `
  --lr 2.4e-04 `
  --weight-decay 0.0375 `
  --dropout 0.15 `
  --drop-path 0.2 `
  --use-amp `
  --gradient-checkpoint `
  --compile `
  --channels-last `
  --include-soft-entropy `
  --include-elastic-budget `
  --warmup-epochs 5 `
  --exp-name tiny_imagenet_512d_16l_bs192_ep100_v4
"@

# 执行训练脚本
Invoke-Expression $script
