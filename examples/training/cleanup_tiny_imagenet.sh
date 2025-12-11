#!/bin/bash
# Tiny ImageNet Dataset Cleanup and Re-download Script
# Usage: ./cleanup_tiny_imagenet.sh

set -e

DATA_DIR="${1:-/app/data}"

echo "======================================================================"
echo "Tiny ImageNet Cleanup Script"
echo "======================================================================"
echo "Data directory: $DATA_DIR"
echo ""

# Remove corrupted zip file
if [ -f "$DATA_DIR/tiny-imagenet-200.zip" ]; then
    echo "🗑️  Removing potentially corrupted zip file..."
    rm -f "$DATA_DIR/tiny-imagenet-200.zip"
    echo "✓ Removed tiny-imagenet-200.zip"
fi

# Remove incomplete extraction
if [ -d "$DATA_DIR/tiny-imagenet-200" ]; then
    echo "🗑️  Removing incomplete dataset directory..."
    rm -rf "$DATA_DIR/tiny-imagenet-200"
    echo "✓ Removed tiny-imagenet-200/"
fi

echo ""
echo "======================================================================"
echo "✓ Cleanup complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "1. Run your training command again"
echo "2. The script will automatically re-download and extract the dataset"
echo ""
echo "Example:"
echo "  uv run python examples/training/train_fractal_vit.py \\"
echo "      --dataset tiny-imagenet \\"
echo "      --epochs 100 \\"
echo "      --batch-size 96 \\"
echo "      --num-workers 8 \\"
echo "      --use-amp \\"
echo "      --ffn-type swiglu_level"
echo ""
