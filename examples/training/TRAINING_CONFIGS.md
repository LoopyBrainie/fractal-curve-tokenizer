# Training Configuration Examples

## RTX 4070 Laptop - Tiny ImageNet (6 hours)

**Hardware Specs:**
- GPU: RTX 4070 Laptop (8GB VRAM, Ampere architecture)
- Shared Memory: 8GB
- CPU Workers: 12 available

**Dataset Setup:**
Tiny ImageNet will be **automatically downloaded** (~237 MB) on first run. The script will:
1. Download `tiny-imagenet-200.zip` from Stanford
2. Extract and organize the dataset structure
3. Reorganize validation set for PyTorch ImageFolder compatibility

**Recommended Configuration (Memory Safe):**

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --batch-size 64 \
    --num-workers 8 \
    --warmup-epochs 10 \
    --lr 8e-4 \
    --weight-decay 0.05 \
    --dropout 0.15 \
    --emb-dropout 0.15 \
    --gradient-clip 1.0 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 224 \
    --depth 9 \
    --heads 8 \
    --dim-head 28 \
    --max-level 3 \
    --pool cls
```

**Expected Performance:**
- Training Time: ~6 hours
- Throughput: ~200-250 images/sec
- Peak VRAM: ~6.5GB (safe for 8GB card)
- Parameters: ~5.8M
- Expected Final Accuracy: 42-48% (top-1), 68-73% (top-5)

**Key Optimizations:**
- ✅ **TF32 Auto-enabled**: ~8× matrix multiplication speedup
- ✅ **Mixed Precision (AMP)**: 30-40% faster training
- ✅ **Optimal Batch Size**: 64 maximizes GPU utilization without OOM
- ✅ **8 Workers**: Good balance for 8GB shm-size (avoids deadlock)
- ✅ **SwiGLU FFN**: -5.8% parameters vs GELU, better convergence
- ✅ **10 Warmup Epochs**: Stabilizes training on larger model
- ✅ **torch.compile**: Optional 10-30% speedup (PyTorch 2.0+)

**With torch.compile (experimental, ~10-30% faster):**

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --batch-size 64 \
    --num-workers 8 \
    --warmup-epochs 10 \
    --use-amp \
    --compile \
    --compile-mode reduce-overhead \
    --ffn-type swiglu_level \
    --dim 224 \
    --depth 9 \
    --max-level 3
```

> ⚠️ First epoch will be slower due to compilation. Speedup visible from epoch 2+.

---

## Alternative Configurations

### 🔬 Feasibility Validation (Quick Test)

**Purpose**: Quickly validate Fractal ViT works correctly before long training runs.

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar10 \
    --epochs 20 \
    --batch-size 64 \
    --num-workers 4 \
    --warmup-epochs 3 \
    --use-amp \
    --gradient-checkpoint \
    --ffn-type swiglu_level \
    --dim 128 \
    --depth 6 \
    --heads 4 \
    --dim-head 32 \
    --max-level 3
```

**Expected:**
- Time: ~5-10 minutes on RTX 4070
- VRAM: ~2.5GB (with gradient checkpoint)
- Accuracy: 60-70% (validates model is learning)
- Parameters: ~1.5M

### Memory-Constrained (4-5GB VRAM)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --batch-size 32 \
    --accum-steps 2 \
    --num-workers 6 \
    --use-amp \
    --gradient-checkpoint \
    --ffn-type swiglu_level \
    --dim 160 \
    --depth 8 \
    --heads 8 \
    --warmup-epochs 10
```

### Ultra-Low Memory (2-3GB VRAM)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar100 \
    --epochs 100 \
    --batch-size 16 \
    --accum-steps 4 \
    --num-workers 4 \
    --use-amp \
    --gradient-checkpoint \
    --ffn-type swiglu \
    --dim 128 \
    --depth 6 \
    --heads 4 \
    --max-level 2
```

**Expected:** ~3GB VRAM, 65-70% accuracy on CIFAR-100

### Speed-Focused (4 hours, ~40% accuracy)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 60 \
    --batch-size 64 \
    --num-workers 10 \
    --warmup-epochs 8 \
    --lr 1e-3 \
    --use-amp \
    --ffn-type swiglu \
    --dim 192 \
    --depth 8 \
    --max-level 3
```

### High-End GPU (12-16GB VRAM, 8 hours, higher accuracy)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 150 \
    --batch-size 96 \
    --num-workers 8 \
    --warmup-epochs 15 \
    --lr 6e-4 \
    --weight-decay 0.08 \
    --dropout 0.2 \
    --emb-dropout 0.2 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 320 \
    --depth 12 \
    --heads 10 \
    --max-level 4
```

---

## CIFAR-100 Quick Benchmark (30 min)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset cifar100 \
    --epochs 100 \
    --batch-size 128 \
    --num-workers 8 \
    --warmup-epochs 10 \
    --use-amp \
    --ffn-type swiglu_level
```

**Expected:** 70-75% accuracy in ~25-30 minutes

---

## Tips for RTX 4070 Laptop

1. **Power Management**: Ensure laptop is plugged in and set to "High Performance" mode
2. **Cooling**: Good ventilation is critical for sustained performance
3. **Background Tasks**: Close unnecessary applications to free RAM
4. **Monitoring**: Use `nvidia-smi -l 1` in another terminal to monitor GPU usage
5. **TF32**: Automatically enabled for Ampere GPUs, provides ~8× speedup for matmul

**Check GPU Before Training:**
```bash
# Verify TF32 support
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Compute: {torch.cuda.get_device_capability(0)}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Compute: (8, 9)  # Ampere architecture
```

---

## Troubleshooting

### Tiny ImageNet Download Issues

**Problem: "File is not a zip file" or corrupted download**

Solution:
```bash
# Remove corrupted files
rm -f /app/data/tiny-imagenet-200.zip
rm -rf /app/data/tiny-imagenet-200

# Re-run training (will automatically re-download)
uv run python examples/training/train_fractal_vit.py --dataset tiny-imagenet ...
```

The script now automatically:
- Validates file size before extraction
- Tests zip file integrity
- Removes corrupted files and re-downloads
- Shows extraction progress

**Problem: Out of Memory (OOM) during training**

**Common causes for 8GB RTX 4070 Laptop:**
- Batch size too large (>64 for Tiny ImageNet)
- Model too large (dim >224 or depth >9)
- max_level too high (>3 increases memory exponentially)

**Solutions (try in order):**

```bash
# Step 1: Reduce batch size (keeps effective batch via accumulation)
--batch-size 32 --accum-steps 2  # Effective batch = 64

# Step 2: Reduce model size (most effective)
--dim 192 --depth 8 --heads 8 --dim-head 24 --max-level 3

# Step 3: Reduce max fractal level (critical for memory)
--max-level 2  # Reduces tokenization memory significantly

# Step 4: Use gradient checkpointing (if available)
# Add to model initialization in code

# Emergency: Ultra-low memory config
--batch-size 16 --accum-steps 4 --dim 128 --depth 6 --max-level 2
```

**Memory-safe config for 8GB VRAM:**
```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --batch-size 64 \
    --dim 224 \
    --depth 9 \
    --max-level 3 \
    --use-amp \
    --num-workers 8
```

**Problem: Training too slow**

Checklist:
- ✅ Is `--use-amp` enabled?
- ✅ Is GPU utilization >90%? (check with `nvidia-smi`)
- ✅ Are you using `--num-workers 8`?
- ✅ Is laptop plugged in and on high-performance mode?
- ✅ Check for thermal throttling (GPU temp should be <85°C)
- ✅ Try `--compile` for PyTorch 2.0+ compilation speedup

**Performance monitoring in training:**
The script now outputs performance stats every 5 epochs:
```
📊 Perf: 245.3 samples/s, mem=5.82GB
```

**Expected throughput for RTX 4070 Laptop:**
| Config | Batch Size | Throughput | VRAM |
|--------|------------|------------|------|
| dim=192, depth=8 | 64 | ~280 samples/s | ~5GB |
| dim=224, depth=9 | 64 | ~200 samples/s | ~6GB |
| dim=256, depth=10 | 32 | ~120 samples/s | ~7GB |
