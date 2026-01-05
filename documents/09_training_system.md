# Chapter 9: Training System

## 9.1 Overview

This chapter describes the modular training infrastructure for `FractalCurveViT`, including the new `fractal_training` module that provides class-balanced sampling, focal loss, FLOPS budgeting, and a fully decoupled trainer architecture.

### Key Design Principles

1. **Model-Agnostic**: Trainer does not depend on specific model implementation
2. **Pluggable Components**: Sampler, Loss, Metrics, Callbacks are independently replaceable
3. **Configuration-Driven**: All hyperparameters via dataclass configs and YAML files
4. **Mathematically Verifiable**: All components have formal definitions with unit tests

### Module Structure

```
src/fractal_training/
├── samplers/      # ClassBalancedSampler, ProgressiveSampler
├── losses/        # FocalLoss, ClassBalancedCE, CompositeLoss
├── metrics/       # ClassificationMetrics (MCA, per-class, head/tail)
├── schedulers/    # FLOPSBudgetLoss, BudgetScheduler
├── trainer/       # ModularTrainer, Callbacks
└── config/        # ExperimentConfig, ConfigLoader
```

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

### Class-Balanced Sampling

For imbalanced datasets (e.g., Tiny-ImageNet), use `ClassBalancedSampler`:

```python
from fractal_training import ClassBalancedSampler

# Get labels from dataset
labels = [label for _, label in train_dataset]

# Create sampler with β=0.9 (near inverse-frequency)
sampler = ClassBalancedSampler(labels, beta=0.9)

# Use in DataLoader
train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
```

**Mathematical Formulation**:
$$P(\text{sample } i) = \frac{w_i}{\sum_j w_j}, \quad w_i = \frac{1}{n_{c_i}^\beta}$$

Where:
- $n_{c_i}$: Number of samples in class $c_i$
- $\beta \in [0, 1]$: Balance strength (0=uniform, 1=inverse frequency)

---

## 9.3 Loss Functions

### Focal Loss

For hard sample mining:

```python
from fractal_training import FocalLoss

loss_fn = FocalLoss(gamma=2.0, alpha=None)  # alpha=None for auto-compute
```

**Mathematical Formulation**:
$$\mathcal{L}_{focal} = -\alpha_c (1 - p_c)^\gamma \log(p_c)$$

### Class-Balanced Cross-Entropy

Based on effective number of samples (Cui et al., CVPR 2019):

```python
from fractal_training import ClassBalancedCE

loss_fn = ClassBalancedCE(class_counts=class_counts, beta=0.9999)
```

**Mathematical Formulation**:
$$E_c = \frac{1 - \beta^{n_c}}{1 - \beta}, \quad w_c = \frac{1}{E_c}$$

### Combined Focal + Class-Balanced

```python
from fractal_training import FocalClassBalancedLoss

loss_fn = FocalClassBalancedLoss(
    num_classes=200,
    class_counts=class_counts,
    gamma=2.0,
    cb_beta=0.9999,
)
```

---

## 9.4 Model Configuration

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

---

## 9.5 Modular Trainer

### Basic Usage

```python
from fractal_training import (
    ModularTrainer,
    TrainerConfig,
    FocalLoss,
    ClassificationMetrics,
    EarlyStoppingCallback,
    CheckpointCallback,
)

# Create trainer
trainer = ModularTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    loss_fn=FocalLoss(gamma=2.0),
    metrics=ClassificationMetrics(num_classes=200),
    callbacks=[
        EarlyStoppingCallback(patience=20),
        CheckpointCallback(checkpoint_dir="./checkpoints"),
    ],
    config=TrainerConfig(
        num_epochs=100,
        gradient_clip_norm=1.0,
        use_amp=True,
    ),
)

# Train
history = trainer.fit()
```

### Training Loop

The `ModularTrainer` implements:

```python
def train_epoch(self):
    self.model.train()
    for batch_idx, (inputs, targets) in enumerate(self.train_loader):
        # Forward with AMP
        with torch.amp.autocast('cuda', enabled=self.config.use_amp):
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
        
        # Backward with gradient clipping
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Callbacks
        self.callbacks.on_batch_end(self, ctx)
```

---

## 9.6 FLOPS Budget Constraint

### FLOPS Computation

```python
from fractal_training import FLOPSConfig, compute_transformer_flops

config = FLOPSConfig(embed_dim=384, num_layers=12, num_heads=8)
flops = compute_transformer_flops(n_tokens=197, config=config)
# 4.54 GFLOPS
```

**Mathematical Formulation**:
$$\text{FLOPS} = L \cdot (12ND^2 + 2N^2D)$$

Where:
- $L$: Number of layers
- $N$: Number of tokens  
- $D$: Embedding dimension

### FLOPS Budget Loss

```python
from fractal_training import FLOPSBudgetLoss

budget_loss = FLOPSBudgetLoss(
    budget=5e9,  # 5 GFLOPS
    lambda_weight=0.1,
)

loss = budget_loss(actual_flops)
```

**Mathematical Formulation**:
$$\mathcal{L}_{FLOPS} = \lambda \cdot \text{ReLU}\left(\frac{\text{FLOPS}_{actual}}{\text{Budget}} - 1\right)^2$$

### Budget Scheduler

Dynamic budget annealing (cosine or linear):

```python
from fractal_training import BudgetScheduler

scheduler = BudgetScheduler(
    budget_max=7.5e9,  # Initial relaxed
    budget_min=5.0e9,  # Final strict
    total_epochs=100,
    schedule="cosine",
)

for epoch in range(100):
    current_budget = scheduler.step(epoch)
```

---

## 9.7 Learning Rate Schedule

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

## 9.8 YAML Configuration System

### Configuration Files

```yaml
# configs/base.yaml
name: base
data:
  batch_size: 128
  sampler_type: default
model:
  embed_dim: 384
  depth: 10
loss:
  type: cross_entropy
budget:
  enabled: false
optimizer:
  type: adamw
  lr: 0.0001
training:
  num_epochs: 100
```

### Configuration Inheritance

```yaml
# configs/tiny_imagenet_balanced.yaml
base: base.yaml  # Inherit from base

name: tiny_imagenet_class_balanced
data:
  sampler_type: class_balanced
  sampler_beta: 0.9
loss:
  type: focal_cb
  focal_gamma: 2.0
budget:
  enabled: true
  flops_budget: 5000000000
```

### Loading Configuration

```python
from fractal_training import ConfigLoader

loader = ConfigLoader(config_dir='configs')
config = loader.load('tiny_imagenet_balanced.yaml')

# Command-line overrides
config = loader.load('base.yaml', overrides={
    'training.num_epochs': 50,
    'optimizer.lr': 5e-5,
})
```

---

## 9.9 Metrics

### Classification Metrics

```python
from fractal_training import ClassificationMetrics

metrics = ClassificationMetrics(num_classes=200, topk=(1, 5))

# During training
for outputs, targets in dataloader:
    metrics.update(outputs, targets)

# Compute results
result = metrics.compute()
print(f"Top-1: {result.top1_accuracy:.2%}")
print(f"MCA: {result.mean_class_accuracy:.2%}")
print(f"Head Acc: {result.head_accuracy:.2%}")
print(f"Tail Acc: {result.tail_accuracy:.2%}")
```

**Mean Class Accuracy (MCA)**:
$$\text{MCA} = \frac{1}{C} \sum_{c=1}^C \frac{\text{TP}_c}{n_c}$$

---

## 9.10 V3 Training Characteristics

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

## 9.11 Experiment Management

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

## 9.12 Training Script

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

## 9.13 Best Practices

### Memory Optimization

1. **Clear tokenizer cache**: Call `model.clear_tokenizer_cache()` after each batch
2. **Mixed precision**: Use `torch.cuda.amp` for 2× memory reduction
3. **Gradient checkpointing**: Enable for large models

### Training Stability

1. **Gradient clipping**: `max_norm=1.0`
2. **Learning rate warmup**: 5-10 epochs
3. **Weight decay**: 0.03 for most configurations

### Class Imbalance

1. **Use ClassBalancedSampler**: `beta=0.9` for strong balance
2. **Use FocalLoss**: `gamma=2.0` for hard sample focus
3. **Monitor MCA**: Not just Top-1 accuracy

### Monitoring

1. **Depth distribution entropy**: Should be > 1.5
2. **Token count variance**: Some variation is healthy
3. **Gradient norms**: Watch for explosions
4. **Per-class accuracy**: Check head vs tail classes

---

## 9.14 Implementation Status

| Component | Status | Location |
|:----------|:-------|:---------|
| ClassBalancedSampler | ✅ Complete | `fractal_training.samplers` |
| ProgressiveSampler | ✅ Complete | `fractal_training.samplers` |
| FocalLoss | ✅ Complete | `fractal_training.losses` |
| ClassBalancedCE | ✅ Complete | `fractal_training.losses` |
| ClassificationMetrics | ✅ Complete | `fractal_training.metrics` |
| FLOPSBudgetLoss | ✅ Complete | `fractal_training.schedulers` |
| BudgetScheduler | ✅ Complete | `fractal_training.schedulers` |
| ModularTrainer | ✅ Complete | `fractal_training.trainer` |
| ConfigLoader | ✅ Complete | `fractal_training.config` |
| W&B Integration | ⏳ Pending | - |
| Visualization Panel | ⏳ Pending | - |

**Tests**: 31/31 passing

> **Next**: [10_testing_qa.md](10_testing_qa.md) - Testing and QA
