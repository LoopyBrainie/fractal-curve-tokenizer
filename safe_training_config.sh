#!/bin/bash
# 安全训练配置 - 4070 Laptop (8GB VRAM) + Tiny ImageNet

# ============================================================================
# 新增内存优化选项说明
# ============================================================================
# --use-gradient-checkpointing: 启用梯度检查点（节省 30-50% 激活值内存，慢 ~10%）
# --max-split-size-mb: CUDA 内存分配器最大块大小（降低可减少碎片，但稍慢）
#                      默认: 512, 建议: 256-128（内存紧张时）

# ============================================================================
# 配置 1: 保守配置（100% 能跑，推荐从这里开始）
# ============================================================================
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 50 \
    --batch-size 96 \
    --accum-steps 3 \
    --lr 5e-4 \
    --dim 320 \
    --depth 10 \
    --heads 8 \
    --dim-head 40 \
    --max-level 4 \
    --num-workers 10 \
    --use-amp

# 有效 batch size: 96 × 3 = 288 (比你的 256 还大！)
# 预计显存: ~4.5-5.5 GB
# 训练速度: ~4-5 min/epoch

# ============================================================================
# 配置 1+: 保守配置 + 梯度检查点（进一步降低显存）
# ============================================================================
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 50 \
    --batch-size 128 \
    --accum-steps 2 \
    --lr 5e-4 \
    --dim 320 \
    --depth 10 \
    --heads 8 \
    --dim-head 40 \
    --max-level 4 \
    --num-workers 10 \
    --use-amp \
    --use-gradient-checkpointing

# 有效 batch size: 128 × 2 = 256
# 预计显存: ~3.5-4.5 GB（梯度检查点节省 ~1GB）
# 训练速度: ~4.5-5.5 min/epoch（稍慢但更安全）

# ============================================================================
# 配置 2: 平衡配置（90% 能跑）
# ============================================================================
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 50 \
    --batch-size 128 \
    --accum-steps 2 \
    --lr 5e-4 \
    --dim 352 \
    --depth 11 \
    --heads 8 \
    --dim-head 44 \
    --max-level 4 \
    --num-workers 10 \
    --use-amp

# 有效 batch size: 128 × 2 = 256
# 预计显存: ~5.5-6.5 GB
# 训练速度: ~3.5-4.5 min/epoch

# ============================================================================
# 配置 2+: 平衡配置 + 梯度检查点（内存优化版）
# ============================================================================
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 50 \
    --batch-size 192 \
    --accum-steps 2 \
    --lr 5e-4 \
    --dim 384 \
    --depth 12 \
    --heads 8 \
    --dim-head 48 \
    --max-level 4 \
    --num-workers 10 \
    --use-amp \
    --use-gradient-checkpointing \
    --max-split-size-mb 256

# 有效 batch size: 192 × 2 = 384
# 预计显存: ~5-6 GB（梯度检查点 + 内存分配优化）
# 训练速度: ~4-5 min/epoch

# ============================================================================
# 配置 3: 激进配置（80% 能跑，接近极限）
# ============================================================================
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 50 \
    --batch-size 160 \
    --accum-steps 2 \
    --lr 5e-4 \
    --dim 384 \
    --depth 12 \
    --heads 8 \
    --dim-head 48 \
    --max-level 4 \
    --num-workers 10 \
    --use-amp

# 有效 batch size: 160 × 2 = 320
# 预计显存: ~6.5-7.5 GB
# 训练速度: ~3-4 min/epoch

# ============================================================================
# 配置 4: 禁用 torch.compile（牺牲 15% 速度，节省 1-2GB 显存）
# ============================================================================
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 50 \
    --batch-size 192 \
    --accum-steps 2 \
    --lr 5e-4 \
    --dim 384 \
    --depth 12 \
    --heads 8 \
    --dim-head 48 \
    --max-level 5 \
    --num-workers 10 \
    --use-amp \
    --quick-test  # 禁用 torch.compile

# 有效 batch size: 192 × 2 = 384
# 预计显存: ~5.5-6.5 GB
# 训练速度: ~4-5 min/epoch（但更稳定）

# ============================================================================
# 显存不足排查
# ============================================================================
# 1. 降低 batch-size: 256 → 128 → 96 → 64
# 2. 降低模型: dim=384 → 352 → 320 → 256
# 3. 降低 depth: 12 → 11 → 10 → 8
# 4. 降低 max-level: 5 → 4 → 3
# 5. 禁用 compile: 添加 --quick-test
