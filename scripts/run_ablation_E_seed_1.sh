#!/bin/bash
# I100-2 Ablation: ÅäÖÃ E - linearÍË»ğ, seed=1
set -e
echo "[I100-2] ÅäÖÃ E seed=1"
OUTPUT_DIR="experiments/temp_ablation/E/seed_1"
mkdir -p "$OUTPUT_DIR"
uv run python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --seed 1 \
    --splitter-temp-start 1.0 \
    --splitter-temp-end 0.5 \
    --splitter-temp-warmup 10 \
    --temp-schedule linear \
    --learnable-temperature true \
    --save-dir "$OUTPUT_DIR" \
    --batch-size 32 \
    --dim 256 \
    --depth 8 \
    --heads 8 \
    --use-amp \
    --gradient-checkpoint
echo "[Íê³É] ÅäÖÃ E seed=1"
