# Hilbert Curve ViT 消融实验指南

本文档详细说明 Hilbert Curve ViT 的消融实验设计、运行方式和结果解读。

## 📋 实验目的

验证 Hilbert Curve ViT 的三个核心设计假设：

| 假设 | 描述 | 验证方法 |
|------|------|----------|
| **H1** | Hilbert 排序优于光栅排序 | E1-Base vs E2-Hilbert |
| **H2** | LCA Bias 优于 Low-Rank Bias | E3-LowRank vs E4-LCA |
| **H3** | 多尺度自适应优于固定尺度 | E4-LCA vs E5-Adaptive |

## 🧪 实验矩阵

```
┌─────────────┬───────────┬────────────┬─────────────┬─────────────────────┐
│ 实验        │ Hilbert   │ Bias Mode  │ 尺度选择    │ 验证假设            │
├─────────────┼───────────┼────────────┼─────────────┼─────────────────────┤
│ E1-Base     │ Raster ❌  │ None       │ Fixed 16×16 │ Baseline            │
│ E2-Hilbert  │ Hilbert ✅ │ None       │ Fixed       │ H1: Hilbert 排序    │
│ E3-LowRank  │ Hilbert ✅ │ low_rank   │ Fixed       │ Bias 基准           │
│ E4-LCA      │ Hilbert ✅ │ lca ⭐     │ Fixed       │ H2: LCA vs Low-Rank │
│ E5-Adaptive │ Hilbert ✅ │ lca ⭐     │ Adaptive ⭐ │ H3: 自适应尺度      │
└─────────────┴───────────┴────────────┴─────────────┴─────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 确保在项目根目录
cd fractal-curve-tokenizer

# 安装依赖 (如果使用 uv)
uv sync

# 或使用 pip
pip install -e .
pip install torchvision
```

### 2. 运行实验

#### 快速测试 (推荐首次使用)

```bash
# 快速测试: 3 epochs, 5000 样本
uv run python tests/benchmarks/ablation_hilbert_curve.py --quick
```

预计时间: ~5-10 分钟 (取决于 GPU)

#### 完整消融实验

```bash
# 完整实验: 20 epochs, CIFAR-10 全集
uv run python tests/benchmarks/ablation_hilbert_curve.py --epochs 20
```

预计时间: ~2-4 小时 (取决于 GPU)

#### 运行特定实验

```bash
# 只验证 H2 (LCA vs Low-Rank)
uv run python tests/benchmarks/ablation_hilbert_curve.py --experiments E3-LowRank E4-LCA

# 只验证 H1 (Hilbert 排序)
uv run python tests/benchmarks/ablation_hilbert_curve.py --experiments E1-Base E2-Hilbert
```

### 3. 保存结果

```bash
# 保存到 JSON 文件
uv run python tests/benchmarks/ablation_hilbert_curve.py \
    --epochs 20 \
    --output workspace/results/ablation_results.json
```

## 📊 结果解读

### 输出示例

```
========================================================================
消融实验分析报告
========================================================================

📊 实验结果摘要:
----------------------------------------------------------------------
实验            最佳验证Acc      总参数     Bias参数       吞吐量
----------------------------------------------------------------------
E1-Base              65.32%      312,458          0     1234.5/s
E2-Hilbert           67.45%      312,458          0     1201.3/s
E3-LowRank           69.12%      361,962     49,504     1156.8/s
E4-LCA               68.87%      312,494         36     1245.2/s
E5-Adaptive          71.23%      312,494         36     1089.4/s
----------------------------------------------------------------------

🔬 假设检验结果:

  H1_hilbert_ordering:
    结论: Hilbert 排序有效

  H2_lca_vs_lowrank:
    结论: LCA 参数减少 99.9%，精度下降 0.25%

  H3_adaptive_scale:
    结论: 自适应有效

💡 优化建议:
  ✅ 保留 Hilbert 排序：提升 2.13% 验证准确率
  ✅ 使用 LCA Bias：参数减少 99.9%，精度变化 -0.25%
```

### 关键指标解读

| 指标 | 含义 | 期望值 |
|------|------|--------|
| 最佳验证Acc | 训练过程中最高验证准确率 | 越高越好 |
| Bias参数 | Hilbert Bias 参数量 | LCA ~36, Low-Rank ~50K |
| 吞吐量 | 每秒处理图像数 | 越高越好 |
| complexity_correlation | Token数与图像复杂度相关性 | >0.3 表示自适应有效 |

### 假设验证标准

| 假设 | 成功标准 | 失败标准 |
|------|----------|----------|
| H1 (Hilbert) | E2 > E1 + 1% | E2 < E1 - 1% |
| H2 (LCA) | E4 ≥ E3 - 0.5% 且参数减少 >90% | E4 < E3 - 2% |
| H3 (Adaptive) | E5 > E4 + 1% 且 correlation > 0.3 | E5 < E4 |

## 🔬 数学背景

### H1: Hilbert 曲线局部性

Hilbert 曲线满足 Hölder 连续性：

$$\|H(d_1) - H(d_2)\|_2 \leq C \cdot |d_1 - d_2|^{1/2}$$

这意味着 1D 序列中相邻的 token 在 2D 空间中也倾向于相邻。

**验证方式**: 对比使用 Hilbert 排序 vs 光栅 (row-major) 排序的模型性能。

### H2: LCA 距离等价性

LCA (Lowest Common Ancestor) 深度直接编码空间距离：

$$\text{LCA}(i, j) = \ell \implies \|pos_i - pos_j\|_\infty \leq \frac{N}{2^\ell}$$

LCA Bias 只需 $O(\log N)$ 参数，而 Low-Rank Bias 需要 $O(d \times rank)$ 参数。

**验证方式**: 对比 LCA Bias vs Low-Rank Bias 的精度和参数效率。

### H3: 自适应 Token 分配

复杂区域应分配更多 token，简单区域分配更少：

$$n_{tokens} \propto \text{complexity}(image)$$

**验证方式**: 
1. 对比固定尺度 vs 自适应尺度的精度
2. 检查 token 数量与图像复杂度的相关性

## ⚙️ 高级配置

### 自定义实验配置

```python
from ablation_hilbert_curve import ExperimentConfig, run_experiment

# 创建自定义配置
custom_config = ExperimentConfig(
    name="Custom-Exp",
    description="自定义实验",
    use_hilbert_order=True,
    bias_mode='lca',
    use_hilbert_bias=True,
    adaptive_scale=True,
    dim=128,          # 更大的模型维度
    depth=6,          # 更深的网络
    heads=8,          # 更多的注意力头
    epochs=30,        # 更长的训练
)

# 运行实验
result = run_experiment(custom_config, train_loader, val_loader, num_classes, device)
```

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--quick` | False | 快速测试模式 |
| `--epochs` | 10 | 训练轮数 |
| `--batch-size` | 64 | 批次大小 |
| `--experiments` | 全部 | 指定运行的实验 |
| `--output` | None | 结果输出路径 |
| `--device` | auto | 计算设备 |

## 📁 输出文件结构

```
workspace/results/
├── ablation_results.json     # 完整实验结果
│   ├── results/              # 每个实验的详细结果
│   │   ├── E1-Base/
│   │   │   ├── train_losses
│   │   │   ├── val_accs
│   │   │   └── ...
│   │   └── ...
│   └── analysis/             # 假设检验和建议
│       ├── summary
│       ├── hypothesis_tests
│       └── recommendations
```

## 🔧 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```bash
   # 减小批次大小
   uv run python tests/benchmarks/ablation_hilbert_curve.py --batch-size 32
   ```

2. **torchvision 未安装**
   ```bash
   pip install torchvision
   ```

3. **CIFAR-10 下载失败**
   - 检查网络连接
   - 手动下载并放置到 `workspace/data/cifar-10-batches-py/`

### 性能优化

- 使用 `--batch-size 128` 提高 GPU 利用率 (需要足够显存)
- 确保使用 CUDA 加速: 检查 `torch.cuda.is_available()`
- 使用 `--quick` 进行初步验证后再运行完整实验

## 📝 实验记录模板

建议使用以下模板记录实验结果：

```markdown
## 消融实验记录

**日期**: YYYY-MM-DD
**设备**: GPU 型号, CUDA 版本
**运行命令**: `uv run python tests/benchmarks/ablation_hilbert_curve.py --epochs 20`

### 结果摘要

| 实验 | 最佳验证Acc | 参数量 | 时间 |
|------|-------------|--------|------|
| E1-Base | XX.XX% | XXX | XX min |
| ... | ... | ... | ... |

### 假设验证

- [ ] H1 (Hilbert 排序): 支持/拒绝
- [ ] H2 (LCA Bias): 支持/拒绝
- [ ] H3 (Adaptive): 支持/拒绝

### 观察与结论

...
```

## 📚 相关文件

- [ablation_hilbert_curve.py](ablation_hilbert_curve.py) - 消融实验主程序
- [benchmark_fractal_vit.py](benchmark_fractal_vit.py) - 性能基准测试
- [ablation_ffn_variants.py](ablation_ffn_variants.py) - FFN 变体消融
- [compare_fractal_vs_standard.py](compare_fractal_vs_standard.py) - 与标准 ViT 对比
