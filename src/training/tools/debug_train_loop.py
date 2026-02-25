#!/usr/bin/env python
"""调试训练循环问题的脚本

性能优化说明:
- num_workers: 多进程并行加载数据，避免 GPU 等待 CPU
- pin_memory: 启用后数据从 CPU 到 GPU 使用 DMA 传输，减少拷贝开销
"""
import sys
import os

# 确保路径正确
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src', 'training'))

import torch
from torch.utils.data import DataLoader

print("="*60)
print("DIAGNOSTIC: 检查训练循环")
print("="*60)

# 测试 1: 检查 DataLoader
print("\n[TEST 1] DataLoader 测试")

# 直接从 train_fractal_vit.py 导入 create_dataloaders
from ..train_fractal_vit import create_dataloaders

# 创建最小配置
class QuickConfig:
    def __init__(self):
        self.batch_size = 32
        # P-OPT: 启用 num_workers 提高数据加载吞吐量
        # GPU 计算速度远快于 CPU 数据加载，需要多进程并行化
        import multiprocessing as mp
        self.num_workers = min(8, mp.cpu_count())
        # P-OPT: 启用 pin_memory 使用 DMA 传输，异步 GPU 加载
        self.pin_memory = True
        self.subset_size = 256
        self.val_split = 0.1
        self.seed = 42
        self.image_size = None  # 使用数据集默认
        self.use_area_encoding = False
        self.transform_mode = 'default'
        self.use_channels_last = False  # I139: 统一命名
        self.compile_model = False  # 添加 compile_model

config = QuickConfig()

# 获取数据集配置
from ..train_fractal_vit import DATASETS
spec = DATASETS['tiny-imagenet']

train_loader, val_loader, test_loader = create_dataloaders(spec, config)

print(f"  DataLoader 长度: {len(train_loader)} batches")
print(f"  Batch size: {train_loader.batch_size}")

# 测试 2: 检查模型
print("\n[TEST 2] 模型 Forward 测试")
from vit_pytorch import FractalCurveViT

model = FractalCurveViT(
    image_size=64,
    num_classes=200,
    dim=128,
    num_layers=4,
    heads=4,
    mlp_dim=256,
)
model.train()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device, non_blocking=True)

print(f"  模型设备: {next(model.parameters()).device}")

# 测试 3: 模拟训练循环的一个迭代
print("\n[TEST 3] 模拟训练迭代")

# 获取一个 batch
batch = next(iter(train_loader))
imgs, labels = batch
imgs = imgs.to(device, non_blocking=True)
labels = labels.to(device, non_blocking=True)

print(f"  输入形状: {imgs.shape}")
print(f"  标签形状: {labels.shape}")

# Forward
try:
    with torch.no_grad():
        stats = model(imgs)
    logits = stats.logits if hasattr(stats, 'logits') else stats
    print(f"  输出形状: {logits.shape}")
    print(f"  Forward 成功!")
except Exception as e:
    print(f"  Forward 失败: {e}")
    import traceback
    traceback.print_exc()

# 测试 4: 检查 torch.compile 是否可用
print("\n[TEST 4] torch.compile 检查")
print(f"  PyTorch 版本: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA 版本: {torch.version.cuda}")
    print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")

try:
    @torch.compile(mode="reduce-overhead")
    def dummy_fn(x):
        return x + 1
    x = torch.randn(10, device=device)
    result = dummy_fn(x)
    print("  torch.compile: 可用")
except Exception as e:
    print(f"  torch.compile: 不可用 ({e})")

# 测试 5: 完整训练循环测试
print("\n[TEST 5] 完整训练循环测试")
print("  模拟 3 个 epoch，每个 2 个 batch...")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

for epoch in range(1, 4):
    model.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    for batch_idx in range(2):  # 只运行 2 个 batch
        batch = next(iter(train_loader))
        imgs, labels = batch
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            stats = model(imgs)
            logits = stats.logits if hasattr(stats, 'logits') else stats
            loss = torch.nn.functional.cross_entropy(logits, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=1)
        epoch_correct += (preds == labels).sum().item()
        epoch_total += labels.numel()

        print(f"    Epoch {epoch}, Batch {batch_idx}: loss={loss.item():.4f}")

    acc = 100. * epoch_correct / epoch_total if epoch_total > 0 else 0
    print(f"  Epoch {epoch} 完成: avg_loss={epoch_loss/2:.4f}, acc={acc:.1f}%")

print("\n" + "="*60)
print("所有测试完成!")
print("="*60)
