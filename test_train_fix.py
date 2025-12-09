#!/usr/bin/env python3
"""测试训练脚本修复是否正确"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# 测试导入
print("Testing imports...")
from vit_pytorch.fractal_vit import NextGenerationFractalViT, SimpleFractalViT
print("✓ Models imported successfully")

# 测试 SimpleFractalViT 创建
print("\nTesting SimpleFractalViT creation...")
try:
    model = SimpleFractalViT(
        image_size=32,
        num_classes=10,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        channels=3,
        dropout=0.1,
        emb_dropout=0.1,
        min_patch_size=(4, 4),
        max_level=3,
    )
    print(f"✓ SimpleFractalViT created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to create SimpleFractalViT: {e}")
    sys.exit(1)

# 测试 NextGenerationFractalViT 创建
print("\nTesting NextGenerationFractalViT creation...")
try:
    model = NextGenerationFractalViT(
        image_size=32,
        num_classes=10,
        dim=128,
        depth=4,
        heads=4,
        mlp_dim=256,
        channels=3,
        dropout=0.1,
        emb_dropout=0.1,
        min_patch_size=(4, 4),
        max_level=3,
        pool='cls',
        dim_head=32,
        learnable_split=False,
    )
    print(f"✓ NextGenerationFractalViT created successfully")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
except Exception as e:
    print(f"✗ Failed to create NextGenerationFractalViT: {e}")
    sys.exit(1)

print("\n✓ All tests passed!")
