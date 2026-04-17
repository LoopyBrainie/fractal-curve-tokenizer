# Chapter 9: Training System

## 9.1 Overview

The **Training System** provides a robust, modular infrastructure for optimizing `FractalCurveViT`. It handles unique challenges of fractal tokenization: non-deterministic sequence lengths, complex gradient flow through token splitters, and numerical stability across varying spatial scales.

**Key Design Principles**:

1. **Model-Agnostic**: Trainer functions do not depend on specific model implementation
2. **Numerical Safety First**: Comprehensive NaN/Inf detection and gradient validation
3. **Decoupled Architecture**: Training logic separated from model definitions via `TrainingState`
4. **BPE Schedule**: Budget, Power, Exploration schedule for fractal tree adaptation

`★ Insight ─────────────────────────────────────`
The BPE (Budget, Power, Exploration) schedule is central to fractal tokenization training — it manages the transition from high-entropy exploration to structural consolidation of the fractal tree. This is crucial because the splitter must learn which regions to split without collapsing into a single scale.
`─────────────────────────────────────────────────`

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLI Layer: train_fractal_vit.py                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Config (config) │  │ Scheduler (LR)   │  │ NumericalDefense │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                    │                    │              │
│           ▼                    ▼                    ▼              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │               train_one_epoch + GradBalancer                 │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │   │
│  │  │ MixupCutmix  │→ │ Backward+   │→ │   AMP        │      │   │
│  │  │ Loss         │  │ GradClip    │  │   GradScaler │      │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│           │                    │                    │              │
│           ▼                    ▼                    ▼              │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │ GradientMonitor  │  │ CheckpointSaver │  │   EpochLogger   │  │
│  └──────────────────┘  └──────────────────┘  └──────────────────┘  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Structure

| Module | Purpose | Key Files |
|:-------|:--------|:---------|
| **CLI Layer** | Entry point and argument parsing | `train_fractal_vit.py`, `main.py` |
| **Config** | Hyperparameter management | `config.py` |
| **Trainer** | Training/evaluation loops | `epoch_train.py`, `epoch_eval.py`, `loss.py` |
| **Scheduler** | Learning rate schedules | `lr_scheduler.py` |
| **Monitor** | Gradient and numerical health | `gradient_monitor.py`, `numerical_defense.py` |
| **Checkpoint** | State persistence | `saver.py`, `loader.py` |

---

## 9.2 Training Script and BPE Schedule

### 9.2.1 Training Entry Point

```bash
# Standard entry
python -m src.training.train_fractal_vit [arguments]

# Direct execution
python src/training/train_fractal_vit.py [arguments]
```

### 9.2.2 BPE (Budget, Power, Exploration) Schedule

The BPE schedule manages the transition from high-entropy exploration to structural consolidation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      3-Stage BPE Schedule                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Stochastic Exploration (Epochs 0-30)                     │
│  ├── Temperature τ = 2.0 (high entropy)                            │
│  ├── Budget penalty = 0 (no compression)                           │
│  └── Purpose: Explore potential tokenization paths                 │
│                                                                      │
│  Stage 2: Power-Law Annealing (Epochs 30-70)                       │
│  ├── Temperature τ: 2.0 → 0.3 (annealing)                         │
│  ├── Budget penalty: 0 → 0.1 (gradual compression)                 │
│  └── Purpose: Balance exploration with compression                  │
│                                                                      │
│  Stage 3: Structural Consolidation (Epochs 70-100)                  │
│  ├── Temperature τ = 0.3 (fixed)                                   │
│  ├── Budget penalty = 0.1 (fixed)                                   │
│  └── Purpose: Stabilize learned fractal patterns                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2.3 Splitter Parameter Groups

The system uses **separate learning rates** for the token splitter (typically 5× the backbone LR):

```python
# From src/training/config.py
param_groups = [
    {'params': backbone_params, 'lr': base_lr},           # e.g., 8e-5
    {'params': splitter_params, 'lr': base_lr * 5},      # e.g., 4e-4
]
```

This ensures the splitter can adapt quickly to the changing feature manifold.

### 9.2.4 Key CLI Arguments

| Argument | Default | Description |
|:---------|:--------|:------------|
| `--splitter-type` | `hilbert_optimal` | Splitter: `hilbert_optimal` (H1SS), `hilbert_entmax`, `gumbel_topk` |
| `--dim` | 384 | Model embedding dimension |
| `--num-layers` | 8 | Number of transformer layers |
| `--heads` | 6 | Number of attention heads |
| `--mlp-dim` | 1536 | FFN hidden dimension |
| `--lr` | 8e-5 | Base learning rate |
| `--splitter-lr` | 4e-4 | Splitter learning rate (typically 5× base_lr) |
| `--weight-decay` | 0.1 | Weight decay |
| `--batch-size` | 128 | Batch size |
| `--num-epochs` | 100 | Number of training epochs |
| `--use-amp` | False | Use automatic mixed precision |
| `--compile` | False | Use torch.compile |
| `--dataset` | `cub200` | Dataset: `cub200`, `tiny_imagenet` |

### 9.2.5 Training with H1SS

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

---

## 9.3 Loss Functions and Evaluation

### 9.3.1 FractalViTLoss System

The `FractalViTLoss` combines standard classification objectives with geometric and structural constraints:

| Loss Component | Purpose | Code Entity |
|:---------------|:--------|:------------|
| **Classification** | Cross-entropy with Label Smoothing/Mixup | `MixupCutmixLoss` |
| **Budget Loss** | Penalizes exceeding target token ratio | `budget_loss_weight` |
| **SDS Regularization** | Spatial-distance-scale consistency | `use_sds_regularization` |
| **Tree Constraint** | Soft constraint on parent-child quadtree logic | `tree_constraint_weight` |

### 9.3.2 Mixup and CutMix

```python
from src.training.trainer.loss import MixupCutmixLoss

loss_fn = MixupCutmixLoss(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    mixup_prob=0.5,
    num_classes=200,
)
```

**Mixup Formulation**:

$$x̃ = \lambda \cdot x_i + (1 - \lambda) \cdot x_j$$
$$ỹ = \lambda \cdot y_i + (1 - \lambda) \cdot y_j$$
$$\lambda \sim \text{Beta}(\alpha, \alpha)$$

**CutMix Formulation**:

$$x̃ = \text{Mix}(x_i, x_j, \text{region})$$
$$\lambda = 1 - \frac{\text{region\_area}}{\text{total\_area}}$$

### 9.3.3 Evaluation Metrics

| Metric | Description |
|:-------|:------------|
| **Top-1 Accuracy** | Standard classification accuracy |
| **Top-5 Accuracy** | Whether correct class in top 5 predictions |
| **ECE** | Expected Calibration Error |
| **Hilbert Locality Score (J)** | Measures spatial distance preservation |

---

## 9.4 Training Loop and Gradient Management

### 9.4.1 GradBalancer

**Problem**: The splitter's auxiliary gradients (budget loss, tree constraint) can vanish or explode relative to classification gradients during training.

**Solution**: `GradBalancer` dynamically adjusts the `budget_loss_weight` using an EMA of gradient norms:

```python
# From src/training/trainer/epoch_train.py
class GradBalancer:
    def __init__(
        self,
        budget_loss_weight: float = 0.1,
        grad_norm_ema: float = 0.95,
        target_ratio: float = 0.1,
    ):
        self.budget_loss_weight = budget_loss_weight
        self.grad_norm_ema = grad_norm_ema
        self.target_ratio = target_ratio
        self.backbone_grad_norm_ema = None
        self.splitter_grad_norm_ema = None

    def update(
        self,
        backbone_grad_norm: float,
        splitter_grad_norm: float,
    ) -> float:
        # Dynamically adjust budget_loss_weight based on gradient ratio
        ratio = splitter_grad_norm / (backbone_grad_norm + 1e-8)
        adjustment = self.target_ratio / (ratio + 1e-8)
        self.budget_loss_weight *= adjustment
        return self.budget_loss_weight
```

### 9.4.2 Memory Management

Implements `expandable_segments` to mitigate memory fragmentation caused by dynamic token counts:

```python
# From train_fractal_vit.py
torch.backends.cuda.expandable_segments = True

# OOM mitigation
if OOM occurred:
    torch.cuda.empty_cache()
    # Retry with smaller batch
```

### 9.4.3 Training State

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
)
```

### 9.4.4 Gradient Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Gradient Control Flow                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Forward Pass: images → backbone → splitter → tokens → logits       │
│                                                                      │
│  Backward Pass:                                                     │
│  ├── grad_norm_backbone = clip(gradients)                           │
│  ├── grad_norm_splitter = clip(gradients)                           │
│  ├── GradBalancer.update(backbone_grad_norm, splitter_grad_norm)     │
│  └── budget_loss_weight = adjusted_weight                           │
│                                                                      │
│  Optimization:                                                       │
│  ├── scaler.scale(loss + budget_loss_weight * budget_loss).backward()│
│  ├── scaler.unscale_(optimizer)                                     │
│  ├── clip_grad_norm_(model.parameters(), max_norm=1.0)              │
│  └── scaler.step(optimizer)                                         │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9.5 Monitoring, Checkpointing, and Logging

### 9.5.1 NumericalDefender

Intercepts NaNs and Infs during training:

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

**NaNAutoInvestigation**: When a NaN is detected, the system triggers investigation to pinpoint whether the failure occurred in:
- Hilbert sort
- Attention weights
- Splitter logits

### 9.5.2 GradientMonitor

Tracks backbone-to-splitter gradient norm ratio:

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
    # ratio = splitter_grad_norm / backbone_grad_norm
```

**Health Check**: A healthy ratio indicates the BPE schedule is working — if the ratio drifts far from the target (typically 0.1), it signals that the splitter's training is becoming unstable.

### 9.5.3 Checkpoint Management

```python
from src.training.checkpoint import save_checkpoint, load_checkpoint

# Save
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

# Load
checkpoint = load_checkpoint(
    checkpoint_path='./checkpoints/best.pth',
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    scaler=scaler,
)
start_epoch = checkpoint['epoch'] + 1
```

**Checkpoint Types**:

| File | Purpose |
|:-----|:--------|
| `best.pth` | Best model based on validation metric |
| `last.pth` | Most recent checkpoint |
| `epoch_*.pth` | Periodic snapshots |

---

## 9.6 Learning Rate Scheduling

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
```

**Mathematical Formulation**:

**Warmup** ($t \leq T_{warmup}$):
$$lr(t) = lr_{base} \cdot \frac{t}{T_{warmup}}$$

**Cosine Decay** ($t > T_{warmup}$):
$$lr(t) = lr_{min} + (lr_{base} - lr_{min}) \cdot \frac{1}{2}(1 + \cos(\pi \cdot \frac{t - T_{warmup}}{T_{total} - T_{warmup}}))$$

---

## 9.7 Mixed Precision Training

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

## 9.8 Best Practices

### Memory Optimization

| Technique | Benefit |
|:----------|:--------|
| `--use-amp` | 2× memory reduction via FP16 |
| `expandable_segments` | Reduces fragmentation from dynamic token counts |
| Gradient checkpointing | Trade compute for memory |

### Training Stability

| Technique | Purpose |
|:----------|:--------|
| Gradient clipping | `max_norm=1.0` prevents explosions |
| GradBalancer | Prevents splitter gradient vanishing/exploding |
| Warmup (15 epochs) | Stabilizes early training |
| `skip_on_nan=True` | Handles NaN gradients gracefully |

### Monitoring

1. **GradientMonitor**: Watch for ratio drift from target (0.1)
2. **LossMonitor**: Track loss component breakdown
3. **Token count variance**: Some variation is healthy for adaptive tokenization

---

## 9.9 Implementation Status

| Component | Status | Location |
|:----------|:-------|:---------|
| `train_one_epoch` | ✅ Complete | `src/training/trainer/epoch_train.py` |
| `GradBalancer` | ✅ Complete | `src/training/trainer/epoch_train.py` |
| `evaluate` | ✅ Complete | `src/training/trainer/epoch_eval.py` |
| `MixupCutmixLoss` | ✅ Complete | `src/training/trainer/loss.py` |
| `NumericalDefender` | ✅ Complete | `src/training/monitor/numerical_defense.py` |
| `GradientMonitor` | ✅ Complete | `src/training/monitor/gradient_monitor.py` |
| `WarmupCosineScheduler` | ✅ Complete | `src/training/scheduler/lr_scheduler.py` |
| `CheckpointSaver` | ✅ Complete | `src/training/checkpoint/saver.py` |

---

## 9.10 Document Navigation

| Chapter | Content |
|:--------|:--------|
| [08_fractal_vit_model](08_fractal_vit_model.md) | Complete model architecture |
| [09_training_system](09_training_system.md) | Training infrastructure (this chapter) |
| [10_testing_qa](10_testing_qa.md) | Testing and benchmarking |

> **Next**: [10_testing_qa.md](10_testing_qa.md) - Testing and Benchmarking
