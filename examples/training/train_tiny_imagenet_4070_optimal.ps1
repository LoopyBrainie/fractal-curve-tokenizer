# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-08)
# ===========================
#
# 1. 模型容量 vs 数据集规模
#    -------------------------
#    Tiny-ImageNet: N = 100,000 samples, C = 200 classes
#    经验法则: θ_optimal ∈ [N/10, N/5] = [10M, 20M]
#    
#    选择: dim=320, depth=12, heads=5 → θ = 21.14M
#    理由: 略高于上界，因为 200 类需要足够容量区分细粒度特征
#
# 2. 几何极限约束 (I16-2)
#    ----------------------
#    image_size=64, min_patch_size=4 → grid_size=16
#    max_depth = ⌊log₂(grid_size)⌋ = 4
#    num_scales = max_depth + 1 = 5
#    max_tokens = (4^(max_depth+1) - 1) / 3 = (4^5 - 1) / 3 = 341
#    
#    验证: elastic_N_max=192 < max_tokens=341 ✓
#
# 3. 学习率缩放 (Linear Scaling Rule)
#    ---------------------------------
#    lr_base = 5e-4 @ batch_size=64
#    lr = lr_base × (batch_size / 64) × 0.8 = 5e-4 × (196/64) × 0.8 ≈ 1.2e-3
#    
#    保守调整因子 0.8 防止小数据集上的不稳定
#
# 4. VRAM 预算 (8GB - 4070 Laptop)
#    ------------------------------
#    Model (fp32): 21.14M × 4B = 84.6 MB
#    Optimizer (AdamW): 21.14M × 8B = 169.1 MB
#    Gradients: 84.6 MB
#    Activations (AMP + Checkpoint): ~0.5 GB (batch_size=196)
#    Buffer: ~0.15 GB
#    Total: ~0.9 GB << 8 GB ✓
#    
#    结论: batch_size=196 可行，充分利用 GPU 并行能力
#
# 5. 训练时间估算
#    -------------
#    样本数: 100,000
#    batch_size: 196
#    迭代/epoch: ⌈100,000 / 196⌉ = 511
#    假设速度: 1.8 s/iter (含 compile 优化)
#    epoch 时间: 511 × 1.8s ≈ 15.3 min
#    100 epochs: ~25.5 小时
#
# 6. 最新架构特性
#    -------------
#    ✓ Gumbel-Softmax + STE (端到端可微分)
#    ✓ Soft Entropy Loss (尺度多样性)
#    ✓ Elastic Budget Loss (Dead Zone 惩罚)
#    ✓ Warmup Forced Split (初始化稳定性)
#    ✓ I10-19 Continuous Relaxation (启用 - 解决 Splitter 崩塌)
#
# ============================================================================

uv run python examples/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --epochs 100 `
  `
  <# ====================================================================== #> `
  <# 模型架构 (21.14M 参数 - Tiny-ImageNet 最优容量)                        #> `
  <# ====================================================================== #> `
  --dim 320 `
  --depth 12 `
  --heads 5 `
  --dim-head 64 `
  --max-level 4 `
  --pool cls `
  --ffn-type swiglu_level `
  `
  <# ====================================================================== #> `
  <# LearnableSplitter 配置 (I16-2 几何极限修复)                            #> `
  <#   num_scales=5 → max_depth=4 → max_tokens=341                          #> `
  <#   target_tokens=80 (经验值，~25% 利用率)                                #> `
  <# ====================================================================== #> `
  --tokenizer-type streaming_v3 `
  --num-scales 5 `
  --target-tokens 80 `
  --split-tau0 0.0 `
  --split-gamma 0.6 `
  --enforce-balance `
  `
  <# ====================================================================== #> `
  <# 训练配置 (batch_size=196 最大化 GPU 利用率)                            #> `
  <#   lr = 1.2e-3 (线性缩放 + 小数据集保守调整 0.8×)                       #> `
  <# ====================================================================== #> `
  --batch-size 196 `
  --num-workers 4 `
  --lr 1.2e-3 `
  --weight-decay 0.05 `
  --warmup-epochs 10 `
  `
  <# ====================================================================== #> `
  <# 正则化 (中等强度 - 21M 参数 + 100k 样本)                               #> `
  <# ====================================================================== #> `
  --dropout 0.1 `
  --emb-dropout 0.1 `
  --drop-path 0.15 `
  --label-smoothing 0.1 `
  `
  <# ====================================================================== #> `
  <# 数据增强 (Mixup + CutMix 标准配置)                                     #> `
  <# ====================================================================== #> `
  --mixup-alpha 0.8 `
  --cutmix-alpha 1.0 `
  --mixup-prob 0.5 `
  `
  <# ====================================================================== #> `
  <# P10 辅助损失 (推荐启用)                                                #> `
  <#   Soft Entropy: 最大化尺度多样性 (防止崩塌)                             #> `
  <#   Elastic Budget: Dead Zone [48, 192] 内零惩罚                         #> `
  <# ====================================================================== #> `
  --include-soft-entropy `
  --soft-entropy-mode maximize `
  --soft-entropy-weight 0.1 `
  --include-elastic-budget `
  --elastic-N-min 48 `
  --elastic-N-max 192 `
  --elastic-lambda-over 0.1 `
  --elastic-lambda-under 0.01 `
  --elastic-lambda-collapse 1.0 `
  `
  <# ====================================================================== #> `
  <# Splitter 温度退火 (从 1.0 线性衰减到 0.3)                              #> `
  <# ====================================================================== #> `
  --splitter-temp-start 1.0 `
  --splitter-temp-end 0.3 `
  --splitter-temp-warmup 5 `
  `
  <# ====================================================================== #> `
  <# I10-19 连续松弛 (解决 Splitter 崩塌问题)                               #> `
  <#   启用完全可微分前向传播，让主损失梯度也能流向 Splitter                  #> `
  <# ====================================================================== #> `
  --use-continuous-relaxation `
  --continuous-max-depth 3 `
  `
  <# ====================================================================== #> `
  <# 性能优化 (4070 Laptop 最大化)                                          #> `
  <#   AMP: FP16 计算，减少 VRAM 和加速                                      #> `
  <#   Gradient Checkpoint: 用计算换内存                                     #> `
  <#   Compile: torch.compile 动态形状优化                                   #> `
  <#   Channels-Last: 卷积优化内存布局                                       #> `
  <# ====================================================================== #> `
  --use-amp `
  --gradient-checkpoint `
  --compile `
  --channels-last `
  --accum-steps 1 `
  --gradient-clip 1.0 `
  `
  <# ====================================================================== #> `
  <# 早停策略 (patience=15 防止过拟合)                                      #> `
  <# ====================================================================== #> `
  --patience 15 `
  --min-delta 0.001 `
  --exp-name tiny_imagenet_4070_optimal_320d_12l_bs196
