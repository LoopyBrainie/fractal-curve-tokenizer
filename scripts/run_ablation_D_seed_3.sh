#!/bin/bash
# I100-2 Ablation: ≈‰÷√ D - warmup=5, seed=3
set -e
echo "[I100-2] ≈‰÷√ D seed=3"
OUTPUT_DIR="experiments/temp_ablation/D/seed_3"
mkdir -p "$OUTPUT_DIR"
uv run python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --seed 3 \
    --splitter-temp-start 1.0 \
    --splitter-temp-end 0.5 \
    --splitter-temp-warmup 5 \
    --temp-schedule cosine \
    --learnable-temperature true \
    --save-dir "$OUTPUT_DIR" \
    --batch-size 32 \
    --dim 256 \
    --depth 8 \
    --heads 8 \
    --use-amp \
    --gradient-checkpoint
echo "[ÕÍ≥…] ≈‰÷√ D seed=3"
