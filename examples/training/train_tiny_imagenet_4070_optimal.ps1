# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-26) - 150 epochs 优化版本
# ============================================================================
#
# 1. 模型架构参数计算
#    --------------------
#    Tiny-ImageNet: N_train = 100,000 samples, C = 200 classes
#
#    参数公式 (SwiGLU FFN):
#      P_total = depth * 16 * dim^2 + 2.5 * dim * num_classes
#      mlp_dim = 4 * dim (自动计算)
#
#    配置 (dim=384, depth=8, heads=6):
#      mlp_dim = 4 * 384 = 1536
#      P_total = 8 * 16 * 384^2 + 2.5 * 384 * 200 = 23.2M
#
#    P/N 比率: 23.2M / 100K = 232 (ViT 可接受范围: [150, 500])
#
# 2. VRAM 预算 (8GB - RTX 4070 Laptop)
#    ----------------------------------
#    优化技术栈:
#      - FP16 (AMP): 50% 内存节省
#      - Gradient Checkpointing: ~50% 激活内存节省
#      - torch.compile: ~10-15% 额外优化
#      - channels-last: ~10-20% 内存效率提升
#
#    内存分解 (batch=192):
#      权重 (FP16):     44.2 MB
#      梯度 (FP16):     44.2 MB
#      优化器 (FP32):   185.6 MB
#      激活值 (CP):     60 MB
#      输入数据:        294 MB
#      开销/缓冲:       ~500 MB
#      总计:            ~1.9 GB
#
# 3. 学习率缩放 (Linear Scaling Rule)
#    ---------------------------------
#    lr_base = 3e-4 @ batch_size=256
#    lr = 3e-4 * (192/256) = 2.25e-4
#
# 4. Tokenizer 配置 (Tiny-ImageNet 64x64)
#    -------------------------------------
#    image_size=64, min_patch_size=4
#    max_depth = log2(64/4) = 4
#    N_candidates = 341 (5 尺度: 1+4+16+64+256)
#
#    动态 Token 数:
#      K_min = max(8, 0.03 * 341) = 11
#      K_max = min(4096, 0.25 * 341) = 85
#      K_avg = 41 (覆盖率 ~12%)
#
# ============================================================================

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 150 `
  --dim 384 `
  --depth 8 `
  --heads 6 `
  --pool cls `
  --ffn-type swiglu_level `
  --tokenizer-type streaming_v3 `
  --min-patch-size 4 `
  --batch-size 192 `
  --num-workers 4 `
  --lr 2.25e-4 `
  --accum-steps 1 `
  --weight-decay 0.08 `
  --warmup-epochs 15 `
  --dropout 0.15 `
  --emb-dropout 0.1 `
  --drop-path 0.15 `
  --label-smoothing 0.1 `
  --mixup-alpha 0.4 `
  --cutmix-alpha 1.0 `
  --mixup-prob 0.5 `
  --include-soft-entropy `
  --soft-entropy-mode maximize `
  --soft-entropy-weight 0.1 `
  --include-elastic-budget `
  --splitter-temp-start 1.0 `
  --splitter-temp-end 0.5 `
  --splitter-temp-warmup 20 `
  --lca-temperature 1.5 `
  --use-amp `
  --gradient-checkpoint `
  --channels-last `
  --tf32 `
  --compile `
  --gradient-clip 1.0 `
  --patience 25 `
  --min-delta 0.001 `
  --exp-name tiny_imagenet_384d_8l_bs192_ep150
"@

# 执行训练脚本
Invoke-Expression $script
