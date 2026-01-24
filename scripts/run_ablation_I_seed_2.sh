#!/bin/bash
# I100-2 Ablation: ≈‰÷√ I - T_end=0.7, seed=2
set -e
echo "[I100-2] ≈‰÷√ I seed=2"
OUTPUT_DIR="experiments/temp_ablation/I/seed_2"
mkdir -p "$OUTPUT_DIR"
uv run python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --seed 2 \
    --splitter-temp-start 1.0 \
    --splitter-temp-end 0.7 \
    --splitter-temp-warmup 10 \
    --temp-schedule cosine \
    --learnable-temperature true \
    --save-dir "$OUTPUT_DIR" \
    --batch-size 32 \
    --dim 256 \
    --depth 8 \
    --heads 8 \
    --use-amp \
    --gradient-checkpoint
echo "[ÕÍ≥…] ≈‰÷√ I seed=2"
