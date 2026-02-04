# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop 8GB)
# 数学形式化分析 (2026-02-04 更新)
# 目标: 200 epochs, batch=192
# ============================================================================
#
# [tiny-imagenet 特性]
#   图像尺寸: 64×64 (原生)
#   max_level = ceil(log2(64/4)) = 4
#   候选区域总数 = 341 (vs ImageNet 的 21,845)
#   结论: token 数量远少于 ImageNet，更小的模型即可
#
# [内存约束模型]
#   M_total = M_params + M_gradients + M_optimizer + M_activations
#   dim=384, L=8:
#   - 参数 (FP16):  48 MB
#   - 梯度 (FP32):  96 MB
#   - 优化器 (FP32): 192 MB
#   - 激活 (checkpoint): ~200 MB
#   - 总计: ~540 MB << 7GB 预算
#
# [架构参数 - 优化版]
#   --dim 384         : 性能-显存平衡点 (24M 参数)
#   --num-layers 8    : 匹配四叉树 max_level=4
#   --heads 6          : dim_head = 384/6 = 64
#   --mlp-dim 1536    : mlp_ratio = 4.0 (SwiGLU)
#
# [Tokenizer 参数]
#   --patch-size 8
#   --min-patch-size 4
#   --token-coverage-min 0.01 : α = 1%
#   --token-coverage-max 0.20 : β = 20%
#   K (token 范围): [8, 64] (64×64 图像)
#
# [训练参数 - 200 epochs]
#   --batch-size 192  : 目标批次 (启用 checkpoint)
#   --lr 1e-4         : 标准学习率
#   --weight-decay 0.05
#   --dropout 0.25    : 200 epochs 强正则化
#   --drop-path 0.25  : 随机深度
#   --warmup-epochs 10
#
# [优化标志]
#   --use-amp, --gradient-checkpoint, --compile, --channels-last
#
# 模型: 24M 参数 | 显存: ~4-5 GB | 预期精度: 55-65%
# ============================================================================

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
  --batch-size 192 `
  --lr 1e-04 `
  --weight-decay 0.05 `
  --dropout 0.25 `
  --drop-path 0.25 `
  --use-amp `
  --gradient-checkpoint `
  --compile `
  --channels-last `
  --include-soft-entropy `
  --include-elastic-budget `
  --warmup-epochs 10 `
  --exp-name tiny_imagenet_384d_8l_bs192_ep200_v1
"@

# 执行训练脚本
Invoke-Expression $script
