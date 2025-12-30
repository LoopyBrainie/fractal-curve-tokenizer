# 🎯 Tiny-ImageNet 4070 Laptop 最优训练参数配置

> **推导版本**: v1.0 (基于 TRAINING_PARAMETERS_ANALYSIS.md v2.4 批判性分析)  
> **目标硬件**: RTX 4070 Laptop (8GB VRAM)  
> **目标约束**: batch_size ≤ 196, 参数量 > 40M, epochs = 100  
> **优化选项**: gradient-checkpoint + compile + channels-last  
> **推导日期**: 2025-12-30

---

## 一、模型架构参数推导

### 1.1 参数量目标约束

**目标**: $P_{total} > 40M$

根据文档的实测系数：
$$P_{layer} \approx 16.71 D^2$$

$$P_{total} \approx N_{depth} \times 16.71 D^2 + P_{other}$$

其中 $P_{other} \approx 2.1M$ (Tokenizer + Position + Head + CLS)

设 $P_{total} > 40M$：
$$N_{depth} \times 16.71 D^2 > 38M$$

### 1.2 候选配置验算

#### 配置 A: dim=512, depth=10 (文档实测 46.84M)

$$P_{layer} = 16.71 \times 512^2 = 16.71 \times 262,144 = 4,380,627$$
$$P_{transformer} = 10 \times 4,380,627 = 43,806,270$$
$$P_{total} \approx 43.8M + 2.1M = 45.9M \approx 46.84M \text{ (文档实测)} \checkmark$$

**结论**: dim=512, depth=10 → **~46.84M** ✓ 满足 >40M

#### 配置 B: dim=448, depth=12

$$P_{layer} = 16.71 \times 448^2 = 16.71 \times 200,704 = 3,353,762$$
$$P_{transformer} = 12 \times 3,353,762 = 40,245,144$$
$$P_{total} \approx 40.2M + 2.1M = 42.3M \checkmark$$

**结论**: dim=448, depth=12 → **~42.3M** ✓ 满足 >40M

#### 配置 C: dim=512, depth=12

$$P_{layer} = 16.71 \times 512^2 = 4,380,627$$
$$P_{transformer} = 12 \times 4,380,627 = 52,567,524$$
$$P_{total} \approx 52.6M + 2.1M = 54.7M \checkmark$$

**结论**: dim=512, depth=12 → **~54.7M** ✓ 满足 >40M (最大)

### 1.3 配置选择: dim=512, depth=12

**推荐配置**: dim=512, depth=12, heads=8, dim_head=64

验证 heads × dim_head = dim:
$$8 \times 64 = 512 \checkmark$$

**实测参数量**: 53,469,399 (**53.47M**) ✓ 远超 40M 目标

**模块分布**:
| 模块 | 参数量 | 占比 |
|------|--------|------|
| transformer | 50.74M | 94.9% |
| tokenizer | 1.73M | 3.2% |
| mlp_head | 0.73M | 1.4% |
| pos_embedding | 0.27M | 0.5% |

---

## 二、VRAM 显存验算

### 2.1 显存公式

使用 gradient checkpoint + AMP 时的峰值显存：

$$V_{peak} = V_{activations} + V_{params} + V_{gradients} + V_{optimizer} + V_{intermediate}$$

### 2.2 各组件计算 (batch_size=196)

**Tiny-ImageNet Token 数量**:
$$N_{tokens} = \frac{64 \times 64}{4 \times 4} + 1 = 256 + 1 = 257$$

**FP16 Activations**:
$$V_{act} = B \times N_{tokens} \times D \times 2 = 196 \times 257 \times 512 \times 2$$
$$V_{act} = 51,574,784 \text{ bytes} \approx 49.2 \text{ MB}$$

**FP32 Parameters** (~54.7M):
$$V_{params} = 54.7M \times 4 = 218.8 \text{ MB}$$

**FP32 Gradients**:
$$V_{grads} = 54.7M \times 4 = 218.8 \text{ MB}$$

**AdamW States (m, v)**:
$$V_{adam} = 54.7M \times 4 \times 2 = 437.6 \text{ MB}$$

**Intermediate (gradient checkpoint 重计算)**:

启用 gradient checkpoint 时，仅存储每层边界的激活值：
$$V_{intermediate} \approx \frac{V_{full}}{N_{depth}} \times 3 = \frac{B \times N \times D \times N_{depth} \times 4}{N_{depth}} \times 3$$
$$V_{intermediate} \approx 196 \times 257 \times 512 \times 4 \times 3 = 309 \text{ MB} \times 3 \approx 927 \text{ MB}$$

但实际上 gradient checkpoint 的显存使用更复杂，根据经验公式：
$$V_{gc} \approx 2 \times B \times N \times D \times 4 + \text{overhead}$$
$$V_{gc} \approx 2 \times 196 \times 257 \times 512 \times 4 = 412 \text{ MB} + 1.5\text{GB overhead}$$

### 2.3 总显存估算

$$V_{peak} \approx 49 + 219 + 219 + 438 + 2000 = 2.9 \text{ GB} + \text{compile overhead}$$

**torch.compile 额外开销**: ~1-2 GB (编译缓存 + 临时张量)

$$V_{total} \approx 2.9 + 1.5 = 4.4 \text{ GB}$$

**安全余量**: 8GB - 4.4GB = 3.6GB (充足)

### 2.4 Batch Size 极限验算

若 batch_size = 196：
- 数据加载: $196 \times 3 \times 64 \times 64 \times 4 = 9.2 \text{ MB}$
- 临时张量峰值: 预估 ~1.5 GB

**结论**: batch_size=196 在 8GB GPU 上**安全可行** ✓

---

## 三、学习率参数推导

### 3.1 线性缩放法则

$$lr = lr_{base} \times \frac{B_{effective}}{B_{reference}}$$

| 变量 | 值 | 说明 |
|------|-----|------|
| $B_{effective}$ | 196 | 实际 batch size |
| $B_{reference}$ | 256 | ViT 原始论文基准 |
| $lr_{base}$ | $1 \times 10^{-3}$ | AdamW 标准基准 |

$$lr = 1 \times 10^{-3} \times \frac{196}{256} = 7.66 \times 10^{-4}$$

### 3.2 参数量调整

大模型通常需要更小的学习率：
$$lr_{adjusted} = lr \times \sqrt{\frac{P_{ref}}{P_{model}}}$$

对于 54.7M 模型 vs ViT-Base (86M):
$$lr_{adjusted} = 7.66 \times 10^{-4} \times \sqrt{\frac{86M}{54.7M}} = 7.66 \times 10^{-4} \times 1.25 \approx 9.6 \times 10^{-4}$$

### 3.3 小图像 (64×64) 保守调整

Tiny-ImageNet 的 64×64 图像信息密度低于标准 224×224，需要更谨慎的学习率：
$$lr_{final} = lr_{adjusted} \times 0.8 = 9.6 \times 10^{-4} \times 0.8 = 7.7 \times 10^{-4}$$

**取整**: $\boxed{lr = 8 \times 10^{-4}}$

---

## 四、Weight Decay 参数推导

### 4.1 逆参数量缩放

$$\lambda = \lambda_{base} \times \sqrt{\frac{P_{ref}}{P_{model}}}$$

对于 54.7M 模型:
$$\lambda = 0.05 \times \sqrt{\frac{86M}{54.7M}} = 0.05 \times 1.25 = 0.0625$$

### 4.2 Batch Size 调整

大 batch 通常需要更强的正则化:
$$\lambda_{adjusted} = \lambda \times \left(\frac{B}{B_{ref}}\right)^{0.25} = 0.0625 \times \left(\frac{196}{128}\right)^{0.25}$$
$$\lambda_{adjusted} = 0.0625 \times 1.11 = 0.069$$

**取整**: $\boxed{\lambda = 0.07}$

---

## 五、Dropout 与 Drop Path 参数推导

### 5.1 有效梯度保留率目标

目标: 保留 ~5-10% 有效梯度

$$\eta_{grad} = (1-p_{dropout})^{2 \times depth} \times \prod_{i=1}^{depth}(1-p_{drop\_path}^{(i)})$$

### 5.2 深层模型 (depth=12) 的参数调整

对于 depth=12，设 $p_{dropout}=0.1$, $p_{drop\_path}=0.2$：

**Dropout 贡献**:
$$(1-0.1)^{24} = 0.9^{24} \approx 0.0798$$

**Drop Path 贡献** (线性递增):
$$\prod_{i=1}^{12}\left(1 - 0.2 \times \frac{i}{12}\right) = \prod_{i=1}^{12}(1 - 0.0167i)$$

$$= (1-0.0167)(1-0.0333)...(1-0.2)$$
$$\approx 0.983 \times 0.967 \times ... \times 0.8 \approx 0.82$$

**总有效梯度**:
$$\eta_{grad} \approx 0.0798 \times 0.82 \approx 6.5\%$$

**结论**: 在目标范围 5-10% 内 ✓

$\boxed{p_{dropout} = 0.1, \quad p_{drop\_path} = 0.2}$

---

## 六、Label Smoothing 参数推导

### 6.1 K=200 类别的交叉熵增量

$$\Delta H = -\epsilon \log\left(\frac{\epsilon}{K}\right) - (1-\epsilon)\log(1-\epsilon)$$

对于 $\epsilon = 0.1$, $K = 200$:
$$\Delta H = -0.1 \times \log(0.0005) - 0.9 \times \log(0.9)$$
$$= -0.1 \times (-7.6) - 0.9 \times (-0.105)$$
$$= 0.76 + 0.095 = 0.855 \text{ nats}$$

### 6.2 大模型正则化加强

大模型 (54.7M) 更容易过拟合，可考虑稍高的 label smoothing:

$$\epsilon_{adjusted} = \epsilon_{base} \times \left(\frac{P_{model}}{P_{ref}}\right)^{0.2}$$
$$= 0.1 \times \left(\frac{54.7M}{30M}\right)^{0.2} = 0.1 \times 1.13 = 0.113$$

**取整**: $\boxed{\epsilon = 0.1}$ (标准值，避免过度平滑)

---

## 七、Mixup/CutMix 参数推导

### 7.1 Beta 分布参数

对于 64×64 小图像，较小的 mixup_alpha 更适合（保留更多原始信息）:

$$\alpha_{mixup} = 0.4 \text{ (偏向纯样本)}$$
$$\alpha_{cutmix} = 1.0 \text{ (均匀分布)}$$

### 7.2 混合概率

$$p_{mix} = 0.5 \text{ (平衡原始/混合样本)}$$

$\boxed{\alpha_{mixup} = 0.4, \quad \alpha_{cutmix} = 1.0, \quad p_{mix} = 0.5}$

---

## 八、Warmup 参数推导

### 8.1 Warmup 步数计算

Tiny-ImageNet 数据集大小:
- 训练集: ~100,000 样本
- 每 epoch 步数: $\lceil 100,000 / 196 \rceil = 511$ 步

设 warmup_epochs = 10:
$$T_{warmup} = 511 \times 10 = 5,110 \text{ steps}$$

总训练步数:
$$T_{total} = 511 \times 100 = 51,100 \text{ steps}$$

Warmup 比例:
$$\frac{T_{warmup}}{T_{total}} = \frac{5,110}{51,100} = 10\% \checkmark$$

$\boxed{warmup\_epochs = 10}$

---

## 九、Early Stopping 参数推导

### 9.1 Patience 计算

$$patience = \lfloor E_{total} \times \rho \rfloor$$

取 $\rho = 0.15$ (15% 训练时间容忍度):
$$patience = \lfloor 100 \times 0.15 \rfloor = 15$$

### 9.2 Min Delta

对于分类任务，验证准确率改进阈值:
$$min\_delta = 0.001 \text{ (0.1% 最小改进)}$$

$\boxed{patience = 15, \quad min\_delta = 0.001}$

---

## 十、Gradient Clipping 参数推导

### 10.1 标准范数约束

$$\|\nabla\|_2 \leq \tau$$

对于大 batch (196) + 大模型 (54.7M)，梯度可能更不稳定:
$$\tau = 1.0 \text{ (标准值)}$$

$\boxed{gradient\_clip = 1.0}$

---

## 十一、可学习分割器参数推导

### 11.1 温度退火

**起始温度**: 
- Gumbel-Softmax 需要高温度以探索
- $T_{start} = 1.0$

**终止温度**:
- 趋近离散采样
- $T_{end} = 0.1$

**Warmup 阶段**:
- 与学习率 warmup 同步
- $warmup = 10$ epochs

### 11.2 辅助损失权重

**熵损失** (鼓励尺度多样性):
$$\lambda_{entropy} = 0.1$$

**预算约束** (控制 token 数量):
$$\lambda_{budget} = 0.01$$

**Token 预算**:
对于 64×64 图像，目标 token 数:
$$N_{budget} = 64 \text{ (每图 64 个 token)}$$

---

## 十二、P14 长尾效应优化参数

### 12.1 Focal Loss

考虑到 Tiny-ImageNet 有 200 类，可能存在类别不平衡:

$$\gamma = 2.0 \text{ (标准聚焦参数)}$$

### 12.2 Class-Balanced Loss

$$\beta = 0.9999 \text{ (高平滑度)}$$

### 12.3 渐进式增强

$$progressive\_aug = \text{True}$$

---

## 十三、最终推荐命令

### PowerShell 格式

```powershell
python examples/training/train_fractal_vit.py `
    --exp-name tiny-imagenet-optimal-54M `
    --dataset tiny-imagenet `
    --batch-size 196 `
    --epochs 100 `
    --dim 512 `
    --depth 12 `
    --heads 8 `
    --dim-head 64 `
    --num-scales 3 `
    --tokenizer-type streaming_v3 `
    --ffn-type swiglu_level `
    --pool cls `
    --lr 8e-4 `
    --warmup-epochs 10 `
    --weight-decay 0.07 `
    --dropout 0.1 `
    --drop-path 0.2 `
    --label-smoothing 0.1 `
    --mixup-alpha 0.4 `
    --cutmix-alpha 1.0 `
    --mixup-prob 0.5 `
    --gradient-clip 1.0 `
    --patience 15 `
    --min-delta 0.001 `
    --splitter-temp-start 1.0 `
    --splitter-temp-end 0.1 `
    --splitter-temp-warmup 10 `
    --lambda-splitter-entropy 0.1 `
    --lambda-splitter-budget 0.01 `
    --splitter-token-budget 64 `
    --use-focal-loss `
    --focal-gamma 2.0 `
    --use-class-balanced `
    --class-balance-beta 0.9999 `
    --progressive-aug `
    --gradient-checkpoint `
    --compile `
    --channels-last `
    --use-amp `
    --seed 42
```

### Bash 格式

```bash
python examples/training/train_fractal_vit.py \
    --exp-name tiny-imagenet-optimal-54M \
    --dataset tiny-imagenet \
    --batch-size 196 \
    --epochs 100 \
    --dim 512 \
    --depth 12 \
    --heads 8 \
    --dim-head 64 \
    --num-scales 3 \
    --tokenizer-type streaming_v3 \
    --ffn-type swiglu_level \
    --pool cls \
    --lr 8e-4 \
    --warmup-epochs 10 \
    --weight-decay 0.07 \
    --dropout 0.1 \
    --drop-path 0.2 \
    --label-smoothing 0.1 \
    --mixup-alpha 0.4 \
    --cutmix-alpha 1.0 \
    --mixup-prob 0.5 \
    --gradient-clip 1.0 \
    --patience 15 \
    --min-delta 0.001 \
    --splitter-temp-start 1.0 \
    --splitter-temp-end 0.1 \
    --splitter-temp-warmup 10 \
    --lambda-splitter-entropy 0.1 \
    --lambda-splitter-budget 0.01 \
    --splitter-token-budget 64 \
    --use-focal-loss \
    --focal-gamma 2.0 \
    --use-class-balanced \
    --class-balance-beta 0.9999 \
    --progressive-aug \
    --gradient-checkpoint \
    --compile \
    --channels-last \
    --use-amp \
    --seed 42
```

---

## 十四、参数验算总结表

| 参数 | 值 | 数学验算依据 |
|------|-----|-------------|
| **架构参数** |
| `--dim` | 512 | $16.71 \times 512^2 \times 12 \approx 52.6M$ |
| `--depth` | 12 | 标准深度，表达能力强 |
| `--heads` | 8 | $512 / 64 = 8$ ✓ |
| `--dim-head` | 64 | $8 \times 64 = 512$ ✓ |
| **实测参数量** | **53.47M** | 实际验证 ✓ (远超 40M 目标) |
| **优化参数** |
| `--lr` | 8e-4 | $\frac{196}{256} \times 1e-3 \times 1.25 \times 0.8 = 7.7e-4 → 8e-4$ |
| `--weight-decay` | 0.07 | $0.05 \times 1.25 \times 1.11 = 0.069 → 0.07$ |
| `--warmup-epochs` | 10 | $\frac{10}{100} = 10\%$ 标准 warmup |
| **正则化参数** |
| `--dropout` | 0.1 | $0.9^{24} = 7.98\%$ 保留 |
| `--drop-path` | 0.2 | 总有效梯度 ~6.5% |
| `--label-smoothing` | 0.1 | $\Delta H = 0.855$ nats |
| **数据增强** |
| `--mixup-alpha` | 0.4 | Beta(0.4, 0.4) 偏向纯样本 |
| `--cutmix-alpha` | 1.0 | Beta(1.0, 1.0) 均匀分布 |
| `--mixup-prob` | 0.5 | 50% 应用概率 |
| **训练控制** |
| `--batch-size` | 196 | VRAM ~4.4GB < 8GB ✓ |
| `--patience` | 15 | $100 \times 0.15 = 15$ epochs |
| `--gradient-clip` | 1.0 | 标准梯度裁剪 |
| **性能优化** |
| `--gradient-checkpoint` | ✓ | 显存节省 ~50% |
| `--compile` | ✓ | ~15% 加速 |
| `--channels-last` | ✓ | 卷积加速 |
| `--use-amp` | ✓ | FP16 混合精度 |
| **P14 长尾优化** |
| `--use-focal-loss` | ✓ | γ=2.0 聚焦困难样本 |
| `--use-class-balanced` | ✓ | β=0.9999 类别平衡 |
| `--progressive-aug` | ✓ | 渐进式增强 |

---

## 十五、预期结果

| 指标 | 预期值 |
|------|--------|
| **参数量** | **53.47M** (实测验证) ✓ |
| **显存峰值** | ~4-5GB (gradient checkpoint + AMP) |
| **训练时间** | ~4-5 小时 (RTX 4070 Laptop) |
| **预期准确率** | 62-68% (Tiny-ImageNet SOTA ~70%) |
| **每 epoch 时间** | ~2-3 分钟 |

---

*文档结束 - Tiny-ImageNet 4070 Laptop 最优配置推导 v1.0*
