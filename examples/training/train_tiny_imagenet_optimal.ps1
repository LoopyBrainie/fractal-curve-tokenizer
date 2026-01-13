#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Tiny-ImageNet 最优训练配置 (RTX 4070 Laptop, 8GB VRAM)

.DESCRIPTION
    数学形式化推导的最优参数配置:
    
    模型配置:
    - dim=384, depth=12, heads=8, dim_head=48
    - 参数量: ~30M (适合 100K 训练样本)
    - SwiGLU FFN + LCA Hilbert Bias
    
    GumbelTopKSplitter (Scheme D):
    - num_scales=4 (max_depth=3, 85 candidates)
    - K_min=12, K_max=64
    - 温度退火: 1.0 → 0.5 (cosine)
    
    正则化 (防止过拟合):
    - dropout=0.2, drop_path=0.2
    - weight_decay=0.1
    - label_smoothing=0.1
    - mixup=0.4, cutmix=1.0
    
    优化:
    - gradient_checkpoint (显存优化)
    - torch.compile (速度优化)
    - channels_last (卷积优化)
    - AMP (FP16 混合精度)

.NOTES
    预计训练时间: 2-3 小时
    预计准确率: 52-56% Top-1
#>

$ErrorActionPreference = "Stop"

# 切换到项目根目录
Set-Location $PSScriptRoot\..\..

# 激活虚拟环境
& .\.venv\Scripts\Activate.ps1

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Tiny-ImageNet 最优训练 (RTX 4070 Laptop)" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""
Write-Host "配置:" -ForegroundColor Yellow
Write-Host "  - dim=384, depth=12, heads=8"
Write-Host "  - batch_size=192, epochs=100"
Write-Host "  - gradient_checkpoint + compile + channels_last"
Write-Host ""

$args = @(
    "examples/training/train_fractal_vit.py",
    
    # === 数据集 ===
    "--dataset", "tiny-imagenet",
    "--batch-size", "192",
    "--val-split", "0.05",
    
    # === 模型架构 ===
    "--dim", "384",
    "--depth", "12",
    "--heads", "8",
    "--dim-head", "48",
    "--num-scales", "4",
    "--pool", "cls",
    "--ffn-type", "swiglu_level",
    
    # === GumbelTopKSplitter (Scheme D) ===
    "--K-min", "12",
    "--K-max", "64",
    "--splitter-temp-start", "1.0",
    "--splitter-temp-end", "0.5",
    "--splitter-temp-warmup", "5",
    
    # === LCA Hilbert Bias (P6-2) ===
    "--lca-temperature", "1.5",
    
    # === 训练配置 ===
    "--epochs", "100",
    "--lr", "2.5e-4",
    "--weight-decay", "0.1",
    "--warmup-epochs", "10",
    "--gradient-clip", "1.0",
    
    # === 正则化 ===
    "--dropout", "0.2",
    "--emb-dropout", "0.15",
    "--drop-path", "0.2",
    "--label-smoothing", "0.1",
    
    # === 数据增强 ===
    "--mixup-alpha", "0.4",
    "--cutmix-alpha", "1.0",
    "--mixup-prob", "0.5",
    
    # === 早停 ===
    "--patience", "15",
    "--min-delta", "0.001",
    
    # === 性能优化 ===
    "--gradient-checkpoint",
    "--compile",
    "--channels-last",
    "--use-amp",
    
    # === 系统 ===
    "--seed", "42"
)

Write-Host "开始训练..." -ForegroundColor Green
Write-Host ""

uv run python @args

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "训练完成" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
