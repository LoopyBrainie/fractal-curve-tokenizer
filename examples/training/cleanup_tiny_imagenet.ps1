# Tiny ImageNet Dataset Cleanup and Re-download Script
# Usage: .\cleanup_tiny_imagenet.ps1 [-DataDir "path"]

param(
    [string]$DataDir = ".\data"
)

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Tiny ImageNet Cleanup Script" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "Data directory: $DataDir"
Write-Host ""

# Remove corrupted zip file
$zipPath = Join-Path $DataDir "tiny-imagenet-200.zip"
if (Test-Path $zipPath) {
    Write-Host "🗑️  Removing potentially corrupted zip file..." -ForegroundColor Yellow
    Remove-Item $zipPath -Force
    Write-Host "✓ Removed tiny-imagenet-200.zip" -ForegroundColor Green
}

# Remove incomplete extraction
$datasetPath = Join-Path $DataDir "tiny-imagenet-200"
if (Test-Path $datasetPath) {
    Write-Host "🗑️  Removing incomplete dataset directory..." -ForegroundColor Yellow
    Remove-Item $datasetPath -Recurse -Force
    Write-Host "✓ Removed tiny-imagenet-200/" -ForegroundColor Green
}

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "✓ Cleanup complete!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Next steps:" -ForegroundColor Yellow
Write-Host "1. Run your training command again"
Write-Host "2. The script will automatically re-download and extract the dataset"
Write-Host ""
Write-Host "Example:" -ForegroundColor Yellow
Write-Host "  uv run python examples/training/train_fractal_vit.py \"
Write-Host "      --dataset tiny-imagenet \"
Write-Host "      --epochs 100 \"
Write-Host "      --batch-size 96 \"
Write-Host "      --num-workers 8 \"
Write-Host "      --use-amp \"
Write-Host "      --ffn-type swiglu_level"
Write-Host ""
