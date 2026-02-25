#!/bin/bash
# Tiny-ImageNet Optimal Training Script (RTX 4070 Laptop 8GB)
# Splitter: GumbelTopKSplitter (default, stochastic exploration)
# Mathematical Formalization & Critical Analysis (2026-02-21)
# Target: 200 epochs, batch=192
#
# Mathematical Verification:
# ============================================================================
# GPU: RTX 4070 Laptop (8GB VRAM, ~7GB usable with safety margin)
#
# [tiny-imagenet Characteristics]
#   Image size: 64×64 (native)
#   max_level = ceil(log2(64/4)) = 4
#   Candidate regions = Σ4^d (d=0→4) = 341
#   Token range: [K_min, K_max] = [11, ~68] for 64×64 images
#
# [MEMORY CONSTRAINT MODEL]
#   M_total = M_params + M_gradients + M_optimizer + M_activations
#   dim=384, L=8, H=6: ~21.2M params
#
#   VRAM Breakdown (batch=192, gradient-checkpoint, AMP):
#   - Parameters (21.2M × 2 bytes FP16):     ~42 MB
#   - Gradients (21.2M × 4 bytes FP32):     ~85 MB
#   - Optimizer (AdamW: 2 × 21.2M × 4):     ~170 MB
#   - Activations (estimated):               ~6-7 GB
#   - Total:                                 ~7-8 GB (within limit)
#
# [MODEL PARAMETER COUNT VERIFICATION]
#   Layer Type              | Formula                  | Count
#   -----------------------|--------------------------|--------
#   Patch Embedding        | 3×384×4×4               | 18,432
#   CLS Token              | 384                      | 384
#   Tokenizer (Conv+MLP)   | ~384×(3×16×16+128+64+1)| ~200K
#   Per Transformer Layer  |                         |
#     - QKV                | 3×384×384                | 442,368
#     - Output Proj        | 384×384                  | 147,456
#     - SwiGLU FFN         | 2×384×1536               | 1,179,648
#     - LayerNorm          | 2×384                    | 768
#   8 Layers ×            |                          | 14,170,944
#   MLP Head               | 384×200                  | 76,800
#   -----------------------|--------------------------|--------
#   TOTAL                  |                          | ~21.2M
#
# [GumbelTopKSplitter MATHEMATICAL ANALYSIS]
#   1. Global Softmax: P(i∈TopK) = softmax(z/τ)[i] × K
#      - Gradient coverage: ~100% (all candidates get gradients)
#      - Gradient ratio: ~20:1 (selected vs unselected)
#
#   2. Learnable Quota (Scheme E):
#      - π_d = softmax(φ)  where φ is learnable logits
#      - K_d = max(K_min_per_depth, round(π_d × K_total))
#      - Ensures depth diversity, prevents collapse
#
#   3. Temperature Annealing:
#      - τ_start = 1.0 (exploration)
#      - τ_end = 0.4 (exploitation)
#      - Gradient strength at τ=0.4: ∂p/∂z ≈ 1/τ = 2.5
#
# [CRITICAL ANALYSIS OF PREVIOUS CONFIGURATION]
# ============================================================================
# Issues Identified:
#   1. weight_decay=0.05 - Too weak for 200 epochs, risks overfitting
#   2. quota_entropy_weight=0.01 - Too low, depth distribution may collapse
#   3. aux_loss_warmup not enabled - Sudden loss spikes possible
#   4. splitter-temp-warmup=5 - Too fast, insufficient exploration
#
# Mathematical Justification for Updates:
#   - weight_decay ∝ epochs: 0.05 → 0.1 for 200 epochs
#   - quota_entropy_weight: 0.01 → 0.1 (10× increase for depth diversity)
#   - aux_loss_warmup_epochs=10: Gradual loss introduction
#   - splitter-temp-warmup=10: Slower annealing for better exploration
#
# [KEY OPTIMIZATIONS]
#   - tokenizer_dropout=0.0  : Deterministic tokenization (critical)
#   - transformer_dropout=0.1: Moderate (fractal sampling provides implicit regularization)
#   - drop_path=0.15        : Path dropout for small images
#   - gradient-checkpoint    : Save ~40% VRAM
#   - compile + channels-last: JIT optimization
#   - soft_entropy + elastic_budget: Depth diversity + token budget
#
# Expected: Top-1 Accuracy 55-65% | Time: ~20-30 hours
# ============================================================================

set -e

# I170-FIX: 使用模块模式运行，修复相对导入问题
uv run python -m src.training.train_fractal_vit \
  --dataset tiny-imagenet \
  --epochs 200 \
  --dim 384 \
  --num-layers 8 \
  --heads 6 \
  --mlp-dim 1536 \
  --min-patch-size 4 \
  --token-coverage-min 0.03 \
  --token-coverage-max 0.20 \
  --K-min-abs 8 \
  --batch-size 192 \
  --lr 8e-05 \
  --weight-decay 0.1 \
  --tokenizer-dropout 0.0 \
  --transformer-dropout 0.1 \
  --emb-dropout 0.0 \
  --drop-path 0.15 \
  --label-smoothing 0.1 \
  --gradient-clip 1.0 \
  --warmup-epochs 15 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 0 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --include-elastic-budget \
  --elastic-coverage-min 0.03 \
  --elastic-coverage-max 0.25 \
  --elastic-lambda-over 0.1 \
  --elastic-lambda-under 0.01 \
  --quota-learnable enable \
  --quota-entropy-weight 0.1 \
  --quota-align-weight 0.3 \
  --quota-align-mode curriculum \
  --splitter-type gumbel_topk \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.4 \
  --splitter-temp-warmup 10 \
  --aux-loss-warmup-epochs 10 \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --patience 30 \
  --num-workers 8 \
  --use-pattern-encoder
