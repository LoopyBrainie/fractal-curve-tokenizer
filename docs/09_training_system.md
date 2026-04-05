# Chapter 9: Training System

## 9.1 Overview

This chapter describes the modular training infrastructure for `FractalCurveViT`, including gradient monitoring, numerical stability defense, mixed-precision training, and a fully decoupled trainer architecture.

### Key Design Principles

1. **Model-Agnostic**: Trainer functions do not depend on specific model implementation
2. **Numerical Safety First**: Comprehensive NaN/Inf detection and gradient validation
3. **Configuration-Driven**: All hyperparameters via dataclass configs
4. **Mathematically Verifiable**: All components have formal definitions with unit tests

### Module Structure

```
src/training/
├── config.py           # Config, TrainingHyperparams, NumericalConfig, etc.
├── train_fractal_vit.py # Main training entry point
├── trainer/
│   ├── epoch_train.py  # train_one_epoch, train_one_epoch_simple
│   ├── epoch_eval.py   # evaluate, evaluate_simple
│   ├── loss.py         # MixupCutmixLoss, compute_loss
│   └── state.py       # TrainingState, EpochMetrics
├── scheduler/
│   └── lr_scheduler.py # WarmupCosineScheduler, create_scheduler
├── monitor/
│   ├── gradient_monitor.py   # GradientMonitor, GradientStatisticsTracker
│   ├── loss_monitor.py        # LossMonitor, LossTracker
│   └── numerical_defense.py   # NumericalDefender, GradientValidator
├── checkpoint/
│   ├── saver.py        # save_checkpoint, save_epoch_stats
│   └── loader.py       # load_checkpoint, find_latest_checkpoint
└── training_logs/
    ├── epoch_logger.py  # EpochLogger
    └── metrics.py       # MetricsTracker, compute_* functions
```

---

## 9.2 Training Entry Point

### Main Script

```bash
python -m src.training.train_fractal_vit [arguments]
```

Or directly:

```bash
python src/training/train_fractal_vit.py [arguments]
```

### Quick Test

```bash
python src/training/train_fractal_vit.py --quick-test --use-amp
```

### Key CLI Arguments

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--splitter-type` | `gumbel_topk` | Splitter type: `hilbert_optimal` (H1SS), `hilbert_entmax`, `gumbel_topk` |
| `--dim` | 384 | Model embedding dimension |
| `--num-layers` | 8 | Number of transformer layers |
| `--heads` | 6 | Number of attention heads |
| `--mlp-dim` | 1536 | FFN hidden dimension |
| `--lr` | 8e-5 | Learning rate |
| `--weight-decay` | 0.1 | Weight decay |
| `--batch-size` | 128 | Batch size |
| `--num-epochs` | 100 | Number of training epochs |
| `--use-amp` | False | Use automatic mixed precision |
| `--compile` | False | Use torch.compile |
| `--dataset` | `cub200` | Dataset name |

> **Note**: Use `--splitter-type hilbert_optimal` for H1SS (recommended) or `hilbert_entmax` for H-entmax.

---

## 9.3 Configuration System

### Config Dataclasses

```python
from src.training import (
    Config,
    TrainingHyperparams,
    NumericalConfig,
    MixedPrecisionConfig,
    create_config,
)

# Create default config
config = create_config(
    num_epochs=100,
    batch_size=128,
    base_lr=8e-5,
    weight_decay=0.1,
)

# Access hyperparameters
print(config.hyperparams.num_epochs)
print(config.numerical.detect_anomaly)
```

### TrainingHyperparams

Core training hyperparameters:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `num_epochs` | 100 | Total training epochs |
| `batch_size` | 128 | Batch size per iteration |
| `accumulation_steps` | 1 | Gradient accumulation steps |
| `gradient_clip_norm` | 1.0 | Gradient clipping threshold |
| `base_lr` | 5e-4 | Base learning rate |
| `warmup_epochs` | 5 | LR warmup epochs |
| `min_lr` | 1e-6 | Minimum learning rate |
| `warmup_start_lr` | 1e-7 | Starting LR during warmup |
| `weight_decay` | 0.05 | Weight decay coefficient |
| `label_smoothing` | 0.0 | Label smoothing factor |
| `mixup_alpha` | 0.8 | Mixup alpha parameter |
| `cutmix_alpha` | 1.0 | CutMix alpha parameter |
| `mixup_cutmix_prob` | 0.5 | Probability of applying Mixup/Cutmix |
| `log_interval` | 50 | Logging interval |
| `eval_interval` | 1 | Evaluation interval |
| `checkpoint_interval` | 10 | Checkpoint save interval |

### NumericalConfig

Numerical stability configuration:

| Parameter | Default | Description |
|:----------|:--------|:------------|
| `detect_anomaly` | False | Enable PyTorch anomaly detection |
| `check_gradients` | True | Check gradients for NaN/Inf |
| `skip_on_nan_grad` | True | Skip optimizer step on NaN/Inf gradient |
| `record_grad_norms` | True | Record gradient norms |
| `record_layer_grad_norms` | True | Record per-layer gradient norms |
| `record_loss_components` | True | Record loss component breakdown |

---

## 9.4 Training Functions

### Basic Training Loop

```python
from src.training import (
    train_one_epoch,
    evaluate,
    TrainingState,
    create_scheduler,
    save_checkpoint,
)

# Create model, optimizer, dataloaders
model = FractalCurveViT(...)
optimizer = torch.optim.AdamW(model.parameters(), lr=8e-5)
scheduler = create_scheduler(optimizer, config)
scaler = torch.cuda.amp.GradScaler()

# Training state
state = TrainingState(
    epoch=0,
    global_step=0,
    best_metric=0.0,
)

# Train for one epoch
for epoch in range(config.hyperparams.num_epochs):
    state.epoch = epoch

    # Train
    metrics = train_one_epoch(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        state=state,
        config=config,
        device=device,
    )

    # Evaluate
    eval_metrics = evaluate(
        model=model,
        val_loader=val_loader,
        device=device,
    )

    # Save checkpoint
    if epoch % 10 == 0:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            metrics=metrics,
            epoch=epoch,
            is_best=(eval_metrics['top1'] > state.best_metric),
        )
```

### TrainingState

```python
from src.training import TrainingState

state = TrainingState(
    epoch=0,
    global_step=0,
    best_metric=0.0,
    is_best=False,
    optimizer_state=None,
    scheduler_state=None,
    scaler_state=None,
    sampler_state=None,
    metrics_history={},
    warning_count=0,
    nan_skip_count=0,
)
```

---

## 9.5 Loss Functions

### MixupCutmixLoss

Combines Mixup and CutMix augmentation:

```python
from src.training.trainer.loss import MixupCutmixLoss

loss_fn = MixupCutmixLoss(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    mixup_prob=0.5,  # Probability of applying augmentation
    num_classes=200,
)

# In training loop
for images, labels in train_loader:
    # Apply Mixup/Cutmix automatically
    mixed_images, mixed_labels = loss_fn(images, labels)

    outputs = model(mixed_images)
    loss = compute_loss(outputs, mixed_labels, loss_fn)
```

**Mixup Mathematical Formulation**:

$$x̃ = \lambda \cdot x_i + (1 - \lambda) \cdot x_j$$
$$ỹ = \lambda \cdot y_i + (1 - \lambda) \cdot y_j$$
$$\lambda \sim \text{Beta}(\alpha, \alpha)$$

**CutMix Mathematical Formulation**:

$$x̃ = \text{Mix}(x_i, x_j, \text{region})$$
$$\lambda = 1 - \frac{\text{region\_area}}{\text{total\_area}}$$

### compute_loss

```python
from src.training.trainer.loss import compute_loss

# For integer labels
loss = compute_loss(logits, labels, None)

# For soft labels (Mixup/Cutmix)
loss = compute_loss(logits, mixed_labels, loss_fn)
```

---

## 9.6 Numerical Defense System

### GradientValidator

Checks gradients for numerical issues:

```python
from src.training.monitor import GradientValidator

validator = GradientValidator(
    model=model,
    skip_on_issue=True,
    log_warnings=True,
)

# After backward
should_skip = validator.check_gradients()
if not should_skip:
    optimizer.step()
```

### NumericalDefender

Comprehensive numerical stability protection:

```python
from src.training.monitor import NumericalDefender

defender = NumericalDefender(
    model=model,
    skip_on_nan=True,
    detect_anomaly=False,
)

# In training loop
with defender:
    outputs = model(images)
    loss = criterion(outputs, labels)
    scaler.scale(loss).backward()
```

### GradientMonitor

Per-layer gradient monitoring:

```python
from src.training.monitor import GradientMonitor

monitor = GradientMonitor(
    model=model,
    log_interval=100,
)

# In training loop
monitor.record_gradients()
if state.global_step % 100 == 0:
    stats = monitor.get_statistics()
    print(stats)
```

---

## 9.7 Learning Rate Scheduling

### WarmupCosineScheduler

```python
from src.training.scheduler import create_scheduler

scheduler = create_scheduler(
    optimizer,
    schedule_type='warmup_cosine',
    num_epochs=100,
    warmup_epochs=15,
    base_lr=8e-5,
    min_lr=1e-6,
)

# In training loop
scheduler.step()
```

**Mathematical Formulation**:

**Warmup** ($t \leq T_{warmup}$):
$$lr(t) = lr_{base} \cdot \frac{t}{T_{warmup}}$$

**Cosine Decay** ($t > T_{warmup}$):
$$lr(t) = lr_{min} + (lr_{base} - lr_{min}) \cdot \frac{1}{2}(1 + \cos(\pi \cdot \frac{t - T_{warmup}}{T_{total} - T_{warmup}}))$$

---

## 9.8 Checkpoint Management

### Save Checkpoint

```python
from src.training.checkpoint import save_checkpoint

save_checkpoint(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
    metrics=metrics,
    epoch=epoch,
    checkpoint_dir='./checkpoints',
    is_best=True,
)
```

### Load Checkpoint

```python
from src.training.checkpoint import load_checkpoint

checkpoint = load_checkpoint(
    checkpoint_path='./checkpoints/best.pth',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
)

start_epoch = checkpoint['epoch'] + 1
```

### Find Latest/Best Checkpoint

```python
from src.training.checkpoint import find_latest_checkpoint, find_best_checkpoint

latest = find_latest_checkpoint('./checkpoints')
best = find_best_checkpoint('./checkpoints')
```

---

## 9.9 Metrics and Logging

### MetricsTracker

```python
from src.training.training_logs import MetricsTracker, compute_accuracy

tracker = MetricsTracker()

# During training
for outputs, targets in dataloader:
    tracker.update(outputs=outputs, targets=targets)

# Compute metrics
metrics = tracker.compute()
# {
#     'top1_accuracy': 0.85,
#     'top5_accuracy': 0.98,
#     'num_samples': 1000,
# }
```

### compute_accuracy

```python
from src.training.training_logs import compute_accuracy

top1, top5 = compute_accuracy(logits, targets, topk=(1, 5))
```

### EpochLogger

```python
from src.training.training_logs import EpochLogger

logger = EpochLogger(
    log_dir='./logs',
    experiment_name='fractal_vit_exp',
)

logger.log_epoch(epoch, metrics, lr=scheduler.get_last_lr()[0])
```

---

## 9.10 H1SS Training Characteristics

### Hilbert Splitter with Stable Selection (H1SS)

H1SS is the recommended splitter type, based on six axioms:

| Axiom | Description |
|:------|:------------|
| A1 | 1D Hilbert manifold convolution |
| A2 | No Gumbel perturbation |
| A3 | Entmax sparse activation |
| A4 | Tree consistency soft constraint |
| A5 | Single Entmax projection |
| A6 | < 10K parameters |

### Training with H1SS

```bash
python -m src.training.train_fractal_vit \
    --splitter-type hilbert_optimal \
    --dataset cub200 \
    --image-size 224 \
    --dim 384 \
    --num-layers 8 \
    --heads 6 \
    --use-amp \
    --compile
```

### Diagnostics

```python
# Get model diagnostics
if hasattr(model, 'get_diagnostics'):
    diagnostics = model.get_diagnostics()
    print(f"Splitter type: {diagnostics.get('splitter_type')}")
    print(f"Gradient coverage: {diagnostics.get('gradient_coverage')}")
```

---

## 9.11 Mixed Precision Training

### Enable AMP

```bash
python -m src.training.train_fractal_vit --use-amp
```

### Manual AMP

```python
scaler = torch.cuda.amp.GradScaler()

for images, labels in train_loader:
    with torch.cuda.amp.autocast():
        outputs = model(images)
        loss = criterion(outputs, labels)

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
```

---

## 9.12 Best Practices

### Memory Optimization

1. **Clear tokenizer cache**: Call `model.clear_tokenizer_cache()` after each batch
2. **Mixed precision**: Use `--use-amp` for 2× memory reduction
3. **Gradient checkpointing**: Use `--gradient-checkpoint` for large models
4. **Channels last**: Use `--channels-last` for faster convolution

### Training Stability

1. **Gradient clipping**: `gradient_clip_norm=1.0`
2. **Learning rate warmup**: 15 epochs
3. **Weight decay**: 0.1 for most configurations
4. **Numerical defense**: Enable `skip_on_nan=True` to handle NaN gradients

### Monitoring

1. **Gradient norms**: Watch for explosions using `GradientMonitor`
2. **Loss trends**: Use `LossMonitor` to track loss components
3. **Token count variance**: Some variation is healthy for adaptive tokenization

---

## 9.13 Implementation Status

| Component | Status | Location |
|:----------|:-------|:---------|
| `train_one_epoch` | ✅ Complete | `src/training/trainer/epoch_train.py` |
| `evaluate` | ✅ Complete | `src/training/trainer/epoch_eval.py` |
| `MixupCutmixLoss` | ✅ Complete | `src/training/trainer/loss.py` |
| `TrainingState` | ✅ Complete | `src/training/trainer/state.py` |
| `NumericalDefender` | ✅ Complete | `src/training/monitor/numerical_defense.py` |
| `GradientMonitor` | ✅ Complete | `src/training/monitor/gradient_monitor.py` |
| `LossMonitor` | ✅ Complete | `src/training/monitor/loss_monitor.py` |
| `WarmupCosineScheduler` | ✅ Complete | `src/training/scheduler/lr_scheduler.py` |
| `save_checkpoint` | ✅ Complete | `src/training/checkpoint/saver.py` |
| `load_checkpoint` | ✅ Complete | `src/training/checkpoint/loader.py` |
| `EpochLogger` | ✅ Complete | `src/training/logging/epoch_logger.py` |
| `MetricsTracker` | ✅ Complete | `src/training/logging/metrics.py` |

---

> **Next**: [10_testing_qa.md](10_testing_qa.md) - Testing and QA
