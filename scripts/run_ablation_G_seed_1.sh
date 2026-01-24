#!/bin/bash
# I100-2 Ablation: ≈‰÷√ G - πÃ∂®Œ¬∂», seed=1
set -e
echo "[I100-2] ≈‰÷√ G seed=1"
OUTPUT_DIR="experiments/temp_ablation/G/seed_1"
mkdir -p "$OUTPUT_DIR"
uv run python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --seed 1 \
    --splitter-temp-start 1.0 \
    --splitter-temp-end 0.5 \
    --splitter-temp-warmup 10 \
    --temp-schedule cosine \
    --learnable-temperature false \
    --save-dir "$OUTPUT_DIR" \
    --batch-size 32 \
    --dim 256 \
    --depth 8 \
    --heads 8 \
    --use-amp \
    --gradient-checkpoint
echo "[ÕÍ≥…] ≈‰÷√ G seed=1"
