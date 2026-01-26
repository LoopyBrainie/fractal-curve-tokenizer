# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-26) - I107-1 分析
# Target: RTX 4070 Laptop (8GB VRAM)
#
# ============================================================================
# 关键发现: batch_size=192 不可行
# - Fractal ViT 自适应 tokenization 不减少激活值计算
# - 前向传播必须处理所有候选区域
# - 激活值显存由输入分辨率和模型尺寸决定，与 K 无关
#
# Memory Breakdown (checkpoint + amp + channels-last):
#   - Model params (FP16): ~17 MB (8.5M params)
#   - Optimizer states (FP32): ~34 MB
#   - Activations: ~3.7 GB (B=128, S=64, D=256, L=8, alpha=0.45)
#   - Total: ~3.8 GB << 8GB VRAM (margin: 4.2 GB)
#
# ============================================================================
# 1. 模型架构参数
#    --------------------
#    Configuration: dim=256, depth=8, heads=8
#    Parameters: ~8.5M (P/N ratio: ~42, balanced)
#
# 2. VRAM 预算 (8GB - RTX 4070 Laptop)
#    ----------------------------------
#    Memory Breakdown:
#      Model params (FP16):     ~17 MB
#      Optimizer states (FP32): ~34 MB
#      Activations (CP):        ~3.7 GB
#      CUDA overhead:           ~0.5 GB
#      Total:                   ~4.2 GB
#    Margin: ~3.8 GB (safe for fluctuations)
#
# 3. Token Budget (I33 相对预算)
#    ---------------------------------
#    image_size=64, min_patch_size=8
#    max_patches = (64/8)^2 = 64
#
#    coverage_min=0.01 (1%): K_min = max(8, 64*0.01) = 8
#    coverage_max=0.05 (5%): K_max = 64*0.05 = 32
#
#    Avg tokens: ~20-32 per image
#
# ============================================================================

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 100 `
  --dim 256 `
  --depth 8 `
  --heads 8 `
  --pool cls `
  --ffn-type swiglu_level `
  --min-patch-size 8 `
  --token-coverage-min 0.01 `
  --token-coverage-max 0.05 `
  --batch-size 128 `
  --num-workers 4 `
  --lr 5e-4 `
  --weight-decay 0.15 `
  --warmup-epochs 10 `
  --dropout 0.25 `
  --emb-dropout 0.15 `
  --drop-path 0.25 `
  --label-smoothing 0.1 `
  --mixup-alpha 0.4 `
  --cutmix-alpha 1.0 `
  --mixup-prob 0.5 `
  --include-soft-entropy `
  --soft-entropy-mode maximize `
  --soft-entropy-weight 0.1 `
  --include-elastic-budget `
  --splitter-temp-start 2.0 `
  --splitter-temp-end 0.5 `
  --splitter-temp-warmup 5 `
  --temp-schedule cosine `
  --lca-temperature 1.5 `
  --use-amp `
  --gradient-checkpoint `
  --channels-last `
  --compile `
  --gradient-clip 1.0 `
  --patience 15 `
  --min-delta 0.001 `
  --exp-name tiny_imagenet_256d_8l_bs128_ep100
"@

# 执行训练脚本
Invoke-Expression $script
