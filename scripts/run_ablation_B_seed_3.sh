#!/bin/bash
# I100-2 Ablation: 配置 B - T_end=0.3更激进, seed=3
set -e
echo "[I100-2] 配置 B seed=3"
OUTPUT_DIR="experiments/temp_ablation/B/seed_3"
mkdir -p "$OUTPUT_DIR"
uv run python src/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --seed 3 \
    --splitter-temp-start 1.0 \
    --splitter-temp-end 0.3 \
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
echo "[完成] 配置 B seed=3"
