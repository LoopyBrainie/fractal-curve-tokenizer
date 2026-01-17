# ============================================================================
# CUB-200-2011 最优训练脚本 - Fractal ViT 细粒度分类
# RTX 4070 Laptop (8GB VRAM) 数学形式化推导版 (2026-01-15)
# ============================================================================
#
# 完整数学形式化分析 (详见 .sh 版本)
# ===================================
#
# §1. 数据集: N_train=5994, C=200, 224×224
# §2. max_level=4: 最小 patch 14×14 (细粒度特征捕获)
# §3. Token: N_candidates=341, K∈[12,256]
# §4. 模型: P≈8.5M, P/N≈1418× (Double Descent 最优区)
# §5. 学习率: lr=2.7e-4 (Linear Scaling + 小数据修正)
# §6. 正则化: wd=0.15, dp=0.22, drop_path=0.16, ls=0.13
# §7. 数据增强: mixup=0.3, cutmix=0.8, prob=0.4 (细粒度优化)
# §8. Warmup: 7 epochs (小数据集延长)
# §9. VRAM: 预计 ~2.5GB << 8GB
# §10. 时间: ~42 分钟
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py `
  --dataset cub200 `
  --epochs 100 `
  --num-workers 4 `
  `
  <# ===== §4: 模型架构 (P≈8.5M, P/N≈1418×) ===== #> `
  --dim 256 `
  --depth 8 `
  --heads 8 `
  --dim-head 32 `
  --max-level 4 `
  --pool cls `
  --ffn-type swiglu_level `
  `
  <# ===== §2-3: Scheme E Token 配置 ===== #> `
  --tokenizer-type streaming_v3 `
  --num-scales 5 `
  --K-min 12 `
  --K-max 256 `
  `
  <# ===== §5: 训练配置 ===== #> `
  --batch-size 64 `
  --accum-steps 3 `
  --lr 2.7e-4 `
  --warmup-epochs 7 `
  `
  <# ===== §6: 正则化 ===== #> `
  --weight-decay 0.15 `
  --dropout 0.22 `
  --emb-dropout 0.15 `
  --drop-path 0.16 `
  --label-smoothing 0.13 `
  `
  <# ===== §7: 数据增强 ===== #> `
  --mixup-alpha 0.3 `
  --cutmix-alpha 0.8 `
  --mixup-prob 0.4 `
  `
  <# ===== 细粒度分类 (CUB-200) ===== #> `
  --use-center-loss `
  --center-loss-weight 0.01 `
  `
  <# ===== 硬件优化 ===== #> `
  --compile `
  --channels-last `
  --gradient-checkpoint `
  --use-amp
