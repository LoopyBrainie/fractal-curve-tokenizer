# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-26) - 200 epochs 优化版本
# Target: RTX 4070 Laptop (8GB VRAM)
#
# ============================================================================
# 1. 模型架构参数
#    --------------------
#    Configuration: dim=320, depth=10, heads=8, mlp_dim=1280
#    Parameters: ~25M (P/N ratio: 250, optimal range [150, 500])
#
# 2. VRAM 预算 (8GB - RTX 4070 Laptop)
#    ----------------------------------
#    Memory Breakdown (batch=192 + checkpoint + amp):
#      Model params (FP32):  ~100 MB
#      Gradients (FP32):     ~100 MB
#      Optimizer states:     ~200 MB (AdamW m+v)
#      Activations (CP):     ~36 MB
#      Input data:           ~150 MB
#      CUDA overhead:        ~256 MB
#      Total:                ~700 MB
#    Margin: ~7200 MB (extremely safe)
#
# 3. Token Budget (I33 相对预算)
#    ---------------------------------
#    image_size=64, min_patch_size=4
#    max_patches = (64/4)^2 = 256
#
#    coverage_min=0.01 (1%): K_min = max(8, 256*0.01) = 8
#    coverage_max=0.05 (5%): K_max = 256*0.05 = 13
#
#    Avg tokens: ~10-15 (much less than previous 20!)
#
# ============================================================================

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 200 `
  --dim 320 `
  --depth 10 `
  --heads 8 `
  --pool cls `
  --ffn-type swiglu_level `
  --tokenizer-type streaming_v3 `
  --min-patch-size 4 `
  --token-coverage-min 0.01 `
  --token-coverage-max 0.05 `
  --batch-size 192 `
  --accum-steps 3 `
  --num-workers 4 `
  --lr 2.7e-4 `
  --weight-decay 0.1 `
  --warmup-epochs 10 `
  --dropout 0.2 `
  --emb-dropout 0.15 `
  --drop-path 0.2 `
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
  --splitter-temp-warmup 10 `
  --temp-schedule cosine `
  --lca-temperature 1.5 `
  --use-amp `
  --gradient-checkpoint `
  --channels-last `
  --tf32 `
  --compile `
  --compile-mode max-autotune `
  --gradient-clip 1.0 `
  --patience 30 `
  --min-delta 0.001 `
  --exp-name tiny_imagenet_320d_10l_bs192x3_ep200
"@

# 执行训练脚本
Invoke-Expression $script
