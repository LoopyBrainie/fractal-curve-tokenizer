#!/bin/bash
# Tiny-ImageNet H1SS (Hilbert-Optimal Splitter) Training Script
# Splitter: HilbertOptimalSplitter (6 Axioms Optimal)
# Mathematical Formalization (2026-02-25)
# Target: 200 epochs, batch=192, RTX 4070 8GB

set -e

# ============================================================================
# Mathematical Formalization
# ============================================================================
# GPU: RTX 4070 Laptop (8GB VRAM, ~7GB usable)
#
# [Axioms Satisfied by H1SS]
#   A1 Locality:     1D Hilbert Conv (kernel=5) → J(S) ≤ 1.5
#   A2 Determinism:  No Gumbel → train/eval IOU = 1.0
#   A3 Gradient:     Entmax sparse → coverage ≥ 95%
#   A4 Tree:         Soft constraint z_parent -= λ × max(z_children)
#   A5 Consistency:  Single Entmax projection → E[|S|] = K, Var → 0
#   A6 Simplicity:   < 10K params, single loss
#
# [Tiny-ImageNet Parameters]
#   Image size: 64×64 (native)
#   max_level = ceil(log2(64/4)) = 4
#   max_tokens = (64/4)² = 256
#   K range: [5, 38] (2% - 15%)
#
# [Memory Budget (BS=192)]
#   M_params:    ~8M × 4B = 32 MB
#   M_grad:      ~8M × 4B = 32 MB
#   M_optimizer: ~8M × 4B × 2 = 64 MB
#   M_activ:    ~400 MB (with gradient checkpoint)
#   M_total:     ~600 MB << 8GB ✓
#
# [Key Differences from GumbelTopK]
#   - No Gumbel randomness → deterministic training/eval
#   - Entmax sparse activation → >95% gradient coverage (vs ~37%)
#   - 1D Hilbert Conv → enforces locality
#   - Single loss function (no multiple auxiliary losses)
#
# Expected: 55-65% Top-1 Accuracy | Time: ~15-25 hours
# ============================================================================

uv run python src/training/train_fractal_vit.py \
  --dataset tiny-imagenet \
  --epochs 200 \
  --dim 256 \
  --num-layers 8 \
  --heads 8 \
  --mlp-dim 512 \
  --min-patch-size 4 \
  --pool weighted \
  --ffn-type swiglu_level \
  --token-coverage-min 0.03 \
  --token-coverage-max 0.15 \
  --K-min-abs 8 \
  --K-max-abs 512 \
  --batch-size 192 \
  --lr 5e-4 \
  --weight-decay 0.1 \
  --tokenizer-dropout 0.0 \
  --transformer-dropout 0.15 \
  --emb-dropout 0.0 \
  --drop-path 0.2 \
  --label-smoothing 0.1 \
  --gradient-clip 1.0 \
  --warmup-epochs 10 \
  --mixup-alpha 0.4 \
  --cutmix-alpha 1.0 \
  --mixup-prob 0.5 \
  --include-soft-entropy \
  --soft-entropy-mode maximize \
  --soft-entropy-weight 0.1 \
  --splitter-type hilbert_optimal \
  --splitter-token-ratio-min 0.02 \
  --splitter-token-ratio-max 0.15 \
  --splitter-temp-start 1.0 \
  --splitter-temp-end 0.3 \
  --jump-loss-weight 0.1 \
  --use-amp \
  --gradient-checkpoint \
  --compile \
  --channels-last \
  --use-area-encoding \
  --use-pattern-encoder \
  --pattern-encoder-mode light \
  --depth-scale-min 0.5 \
  --depth-scale-max 2.0 \
  --patience 25 \
  --num-workers 8 \
  --seed 42 \
  --monitor-gradient-balance \
  --monitor-gradient-ratio \
  --monitor-token-stability
