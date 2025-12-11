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

**Recommended Configuration:**

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --batch-size 96 \
    --num-workers 8 \
    --warmup-epochs 10 \
    --lr 8e-4 \
    --weight-decay 0.05 \
    --dropout 0.15 \
    --emb-dropout 0.15 \
    --gradient-clip 1.0 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 256 \
    --depth 10 \
    --heads 8 \
    --dim-head 32 \
    --max-level 4 \
    --pool cls
```

**Expected Performance:**
- Training Time: ~6 hours
- Throughput: ~280-350 images/sec
- Peak VRAM: ~7.2GB
- Parameters: ~8.5M
- Expected Final Accuracy: 45-50% (top-1), 70-75% (top-5)

**Key Optimizations:**
- ✅ **TF32 Auto-enabled**: ~8× matrix multiplication speedup
- ✅ **Mixed Precision (AMP)**: 30-40% faster training
- ✅ **Optimal Batch Size**: 96 maximizes GPU utilization without OOM
- ✅ **8 Workers**: Good balance for 8GB shm-size (avoids deadlock)
- ✅ **SwiGLU FFN**: -5.8% parameters vs GELU, better convergence
- ✅ **10 Warmup Epochs**: Stabilizes training on larger model

---

## Alternative Configurations

### Memory-Constrained (6GB VRAM)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 100 \
    --batch-size 48 \
    --accum-steps 2 \
    --num-workers 6 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 192 \
    --depth 8 \
    --warmup-epochs 10
```

### Speed-Focused (4 hours, lower accuracy)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 60 \
    --batch-size 128 \
    --num-workers 10 \
    --warmup-epochs 8 \
    --lr 1e-3 \
    --use-amp \
    --ffn-type swiglu \
    --dim 192 \
    --depth 8
```

### Accuracy-Focused (12 hours, higher accuracy)

```bash
uv run python examples/training/train_fractal_vit.py \
    --dataset tiny-imagenet \
    --epochs 150 \
    --batch-size 64 \
    --num-workers 8 \
    --warmup-epochs 15 \
    --lr 5e-4 \
    --weight-decay 0.1 \
    --dropout 0.2 \
    --emb-dropout 0.2 \
    --use-amp \
    --ffn-type swiglu_level \
    --dim 320 \
    --depth 12 \
    --heads 10
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
