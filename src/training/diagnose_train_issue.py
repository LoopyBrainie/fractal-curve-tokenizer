#!/usr/bin/env python
"""诊断训练问题的测试脚本"""
import sys
import os
# 确保路径正确
if os.path.exists('src'):
    sys.path.insert(0, os.path.abspath('src'))
if os.path.exists('src/training'):
    sys.path.insert(0, os.path.abspath('src/training'))

import torch
from torch.utils.data import DataLoader

print(f"Python path: {sys.path[:3]}...")
print(f"Working dir: {os.getcwd()}")

# 测试 1: DataLoader
print("="*60)
print("TEST 1: DataLoader")
print("="*60)

# 尝试多种导入方式
try:
    from dataset_factory import create_dataset
except ImportError:
    try:
        from training.dataset_factory import create_dataset
    except ImportError:
        import importlib.util
        spec = importlib.util.spec_from_file_location("dataset_factory", "src/training/dataset_factory.py")
        dataset_factory = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dataset_factory)
        create_dataset = dataset_factory.create_dataset

train_ds, val_ds, _ = create_dataset('tiny-imagenet', 'data', batch_size=192, num_workers=0, pin_memory=False)
train_loader = DataLoader(train_ds, batch_size=192, shuffle=True, num_workers=0, pin_memory=False)

print(f"Train loader batches: {len(train_loader)}")

print("\nTesting first 3 batches...")
for i, batch in enumerate(train_loader):
    imgs, labels = batch
    print(f"  Batch {i}: imgs={imgs.shape}, labels={labels.shape}")
    if i >= 2:
        break
print("DataLoader OK!")

# 测试 2: Model forward
print("\n" + "="*60)
print("TEST 2: Model Forward")
print("="*60)
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=64,
    num_classes=200,
    dim=256,
    depth=8,
    heads=8,
    mlp_dim=512,
    max_depth=6,
)
model.train()

print("Testing forward pass...")
batch = next(iter(train_loader))
imgs = batch[0][:8]  # 使用 8 张图片测试
print(f"  Input: {imgs.shape}")

with torch.no_grad():
    stats = model(imgs)
print(f"  Output logits: {stats.logits.shape}")
print(f"  Num tokens: {stats.num_tokens}")
print("Model forward OK!")

# 测试 3: 完整训练循环的一个 batch
print("\n" + "="*60)
print("TEST 3: Single Batch Training")
print("="*60)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

imgs, labels = next(iter(train_loader))
imgs = imgs.to(device)
labels = labels.to(device)

print("Running forward + backward...")
with torch.amp.autocast('cuda', enabled=torch.cuda.is_available() and scaler is not None):
    stats = model(imgs)
    loss = torch.nn.functional.cross_entropy(stats.logits, labels)
    print(f"  Loss: {loss.item():.4f}")

loss.backward()
optimizer.step()
optimizer.zero_grad()
print("Single batch training OK!")

print("\n" + "="*60)
print("ALL TESTS PASSED!")
print("="*60)
