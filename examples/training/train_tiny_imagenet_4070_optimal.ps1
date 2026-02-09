# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop 8GB)
# 数学形式化分析 (2026-02-07 更新)
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
#   - 参数 (FP16):  ~10 MB
#   - 梯度 (FP32):  ~19 MB
#   - 优化器 (FP32): ~38 MB
#   - 激活 (checkpoint): ~57 MB
#   - 总计: ~424 MB << 8GB 预算
#
# [架构参数 - 优化版]
#   --dim 384         : 性能-显存平衡点 (4.8M 参数)
#   --num-layers 8    : 匹配四叉树 max_level=4 (2倍深度)
#   --heads 6          : dim_head = 384/6 = 64
#   --mlp-dim 1536    : mlp_ratio = 4.0 (SwiGLU)
#
# [Tokenizer 参数]
#   --min-patch-size 4
#   --token-coverage-min 0.01 : α = 1%
#   --token-coverage-max 0.20 : β = 20%
#   K (token 范围): [8, 68] (64×64 图像)
#
# [正则化参数 - 关键修复]
#   --tokenizer-dropout 0.0  : 确定性分词 (必须为0)
#   --transformer-dropout 0.25 : Transformer dropout (200 epochs 强正则化)
#   --emb-dropout 0.0         : 确定性嵌入 (必须为0)
#   --drop-path 0.25  : 随机深度
#
# [早停与配额]
#   --patience 25     : 200 epochs 的 12.5%，平衡收敛与效率
#   --quota-learnable enable : 启用 Scheme E 可学习配额
#
# [优化标志]
#   --use-amp, --gradient-checkpoint, --compile, --channels-last
#
# 模型: 4.8M 参数 | 显存: ~500 MB | 预期精度: 55-65%
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
  --tokenizer-dropout 0.0 `
  --transformer-dropout 0.25 `
  --emb-dropout 0.0 `
  --drop-path 0.25 `
  --patience 25 `
  --quota-learnable enable `
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
