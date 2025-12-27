# Chapter 9: Training System

## 9.1 Overview

This chapter describes the training infrastructure, including dataset support, optimization strategies, and best practices for training `FractalCurveViT`.

---

## 9.2 Dataset Support

### Supported Datasets

| Dataset | Classes | Size | Auto-Download |
|:--------|:--------|:-----|:--------------|
| MNIST | 10 | 28×28 | ✓ |
| CIFAR-10 | 10 | 32×32 | ✓ |
| CIFAR-100 | 100 | 32×32 | ✓ |
| Tiny-ImageNet | 200 | 64×64 | ✓ |
| ImageNet | 1000 | 224×224 | ✗ |
| COCO | Variable | Variable | ✗ |
| Caltech-256 | 256 | Variable | ✗ |

### Data Augmentation

| Dataset | Augmentation Strategy |
|:--------|:---------------------|
| MNIST | Resize, RandomRotation, Normalize |
| CIFAR-10/100 | AutoAugment (CIFAR10 policy) |
| ImageNet/COCO | AutoAugment (ImageNet policy) |
| General | RandomCrop, RandomHorizontalFlip, RandomErasing |

---

## 9.3 Model Configuration

### Recommended Hyperparameters

| Parameter | CIFAR-10 | ImageNet | Notes |
|:----------|:---------|:---------|:------|
| `dim` | 192 | 512 | Embedding dimension |
| `depth` | 9 | 12 | Transformer layers |
| `heads` | 6 | 8 | Attention heads |
| `mlp_dim` | 384 | 2048 | FFN hidden dim |
| `patch_size` | 4 | 16 | Base patch size |
| `dropout` | 0.1 | 0.1 | Dropout rate |
| `drop_path` | 0.1 | 0.1 | DropPath rate |

### P0 Training Fixes

| Parameter | Old Default | New Default | Impact |
|:----------|:------------|:------------|:-------|
| `mlp_dim` | dim × 2 | dim × 4 | +3-5% accuracy |
| `dropout` | 0.3 | 0.1 | +2-3% accuracy |
| `drop_path` | 0.2 | 0.1 | +1-2% accuracy |
| `mixup_alpha` | 0.8 | 0.4 | +1-2% accuracy |
| `weight_decay` | 0.05 | 0.03 | +0.5-1% accuracy |

---

## 9.4 Training Loop

### Basic Training

```python
def train_one_epoch(model, dataloader, optimizer, criterion, scaler):
    model.train()
    
    for images, labels in dataloader:
        images = images.cuda()
        labels = labels.cuda()
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        # Clear tokenizer cache
        model.clear_tokenizer_cache()
```

### Mixed Precision Training

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

---

## 9.5 Learning Rate Schedule

### Warmup + Cosine Annealing

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Warmup for first 5 epochs
warmup = LinearLR(optimizer, start_factor=0.1, total_iters=5)

# Cosine annealing for remaining epochs
cosine = CosineAnnealingLR(optimizer, T_max=epochs - 5)

# Combine schedulers
scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[5])
```

### Schedule Visualization

```
LR
 │
 │     ╱‾‾‾‾‾‾‾‾╲
 │    ╱          ╲
 │   ╱            ╲
 │  ╱              ╲
 │ ╱                ╲
 │╱                  ╲
 └────────────────────── Epoch
   5    Warmup   Cosine
```

---

## 9.6 V3 Training Characteristics

### Variable Depth Tokens

V3 uses content-adaptive quadtree splitting without temperature annealing:

```python
for epoch in range(epochs):
    for images, labels in dataloader:
        output = model(images)
        loss = criterion(output, labels)
        
        # Optional: Monitor split statistics
        if epoch % 10 == 0:
            stats = model.tokenizer.get_split_stats()
            print(f"Mean tokens: {np.mean(stats['num_tokens']):.1f}")
            print(f"Depth entropy: {model.tokenizer.get_scale_entropy():.3f}")
        
        loss.backward()
        optimizer.step()
```

### Diagnostics

```python
# Split statistics
stats = tokenizer.get_split_stats()
# {
#     'num_tokens': [48, 52, 64, ...],      # Tokens per image
#     'depth_distributions': [
#         {0: 1, 1: 4, 2: 16, 3: 27},       # Per-image depth counts
#         ...
#     ],
#     'mean_complexity': 0.42,
# }

# Depth diversity (higher is better)
entropy = tokenizer.get_scale_entropy()  # Target: > 1.5
```

---

## 9.7 Experiment Management

### Directory Structure

```
experiments/
└── fractal_vit_simple_20251214_123456/
    ├── checkpoints/
    │   ├── best.pth
    │   └── last.pth
    ├── logs/
    │   └── training.log
    ├── visualizations/
    │   ├── training_curves.png
    │   └── attention_maps.png
    └── training_history.json
```

### Checkpoint Format

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'best_accuracy': best_acc,
    'config': config.__dict__,
}
torch.save(checkpoint, 'best.pth')
```

### Loading Checkpoint

```python
checkpoint = torch.load('best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
start_epoch = checkpoint['epoch'] + 1
```

---

## 9.8 Training Script

### Command Line Interface

```bash
python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --image-size 32 \
    --dim 192 \
    --depth 9 \
    --heads 6 \
    --mlp-dim 384 \
    --tokenizer-type streaming_v3 \
    --bias-mode lca \
    --ffn-type swiglu_level \
    --epochs 100 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 0.03
```

### Key Arguments

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--tokenizer-type` | `streaming_v3` | Tokenizer type (V3 only) |
| `--bias-mode` | `lca` | Hilbert bias mode |
| `--ffn-type` | `swiglu_level` | FFN type |
| `--dim` | 384 | Model dimension |
| `--depth` | 6 | Transformer layers |
| `--heads` | 6 | Attention heads |
| `--lr` | 5e-4 | Learning rate |
| `--weight-decay` | 0.03 | Weight decay |

---

## 9.9 Best Practices

### Memory Optimization

1. **Clear tokenizer cache**: Call `model.clear_tokenizer_cache()` after each batch
2. **Mixed precision**: Use `torch.cuda.amp` for 2× memory reduction
3. **Gradient checkpointing**: Enable for large models

### Training Stability

1. **Gradient clipping**: `max_norm=1.0`
2. **Learning rate warmup**: 5-10 epochs
3. **Weight decay**: 0.03 for most configurations

### Monitoring

1. **Depth distribution entropy**: Should be > 1.5
2. **Token count variance**: Some variation is healthy
3. **Gradient norms**: Watch for explosions

> **Next**: [10_testing_qa.md](10_testing_qa.md) - Testing and QA
