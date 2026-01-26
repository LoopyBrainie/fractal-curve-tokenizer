# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop 8GB)
# ============================================================================
#
# 数学形式化分析 (2026-01-26) - I107-1 更新
# Target: RTX 4070 Laptop (8GB VRAM), 200 epochs
#
# ============================================================================
# 关键发现: batch_size=192 完全可行
# - 实际显存使用率仅 13.1% (约 1073 MB)
# - Fractal ViT 动态 tokenization 相比标准 ViT 节省 ~40× attention 显存
# - 激活值由输入分辨率和模型尺寸决定，但 gradient checkpoint 有效压缩
#
# Memory Breakdown (checkpoint + amp + channels-last + batch=192):
#   - Model params (FP16): ~25 MB (12.9M params)
#   - Optimizer states (FP32): ~50 MB
#   - Activations (CP): ~400-500 MB
#   - Total: ~1073 MB << 8GB VRAM (margin: 7.1 GB)
#
# ============================================================================
# 1. 模型架构参数
#    --------------------
#    Configuration: dim=256, depth=8, heads=8, mlp_dim=512
#    Parameters: ~12.9M (P/N ratio: ~42, balanced)
#    max_depth=6 (四叉树递归深度)
#
# 2. VRAM 预算 (8GB - RTX 4070 Laptop)
#    ----------------------------------
#    Memory Breakdown:
#      Model params (FP16):     ~25 MB
#      Optimizer states (FP32): ~50 MB
#      Activations (CP):        ~500 MB
#      CUDA overhead:           ~500 MB
#      Total:                   ~1073 MB
#    Margin: ~7119 MB (99.7% 可用)
#
# 3. Token Budget (I33 相对预算)
#    ---------------------------------
#    image_size=64, min_patch_size=4
#    max_patches = (64/4)^2 = 256
#
#    coverage_min=0.01 (1%): K_min = max(8, 256*0.01) = 8
#    coverage_max=0.05 (5%): K_max = min(4096, 256*0.05) = 64
#
#    Avg tokens: ~32-48 per image
#
# ============================================================================

$script = @"
uv run python src/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 200 `
  --dim 256 `
  --depth 8 `
  --heads 8 `
  --mlp-dim 512 `
  --max-depth 6 `
  --pool cls `
  --ffn-type swiglu_level `
  --min-patch-size 4 `
  --token-coverage-min 0.01 `
  --token-coverage-max 0.25 `
  --elastic-coverage-min 0.03 `
  --elastic-coverage-max 0.25 `
  --elastic-lambda-over 0.1 `
  --elastic-lambda-under 0.01 `
  --batch-size 192 `
  --num-workers 8 `
  --lr 5e-4 `
  --weight-decay 0.15 `
  --warmup-epochs 15 `
  --dropout 0.25 `
  --drop-path 0.25 `
  --label-smoothing 0.1 `
  --mixup-alpha 0.4 `
  --cutmix-alpha 1.0 `
  --mixup-prob 0.5 `
  --include-soft-entropy `
  --soft-entropy-mode maximize `
  --soft-entropy-weight 0.05 `
  --include-elastic-budget `
  --elastic-coverage-min 0.03 `
  --elastic-coverage-max 0.25 `
  --elastic-lambda-over 0.1 `
  --elastic-lambda-under 0.01 `
  --use-amp `
  --gradient-checkpoint `
  --channels-last `
  --gradient-clip 1.0 `
  --patience 20 `
  --min-delta 0.001 `
  --exp-name tiny_imagenet_256d_8l_bs192_ep200
"@

# 执行训练脚本
Invoke-Expression $script
