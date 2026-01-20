# ============================================================================
# Tiny-ImageNet 最优训练脚本 (RTX 4070 Laptop)
# ============================================================================
#
# 数学形式化分析 (2026-01-20) - 基于第一性原理推导
# ============================================================================
#
# 1. 模型架构参数计算
#    --------------------
#    Tiny-ImageNet: N_train = 100,000 samples, C = 200 classes
#
#    参数公式 (SwiGLU FFN):
#      P_total = depth * 16 * dim^2 + 2.5 * dim * num_classes
#      P_attn = 4 * dim^2 (Q, K, V, O)
#      P_ffn = 12 * dim^2 (SwiGLU: up, gate, down)
#      每层总计: 16 * dim^2
#
#    推荐配置 (dim=384, depth=10, heads=6):
#      P_total = 10 * 16 * 384^2 + 2.5 * 384 * 200
#              = 10 * 2,359,296 + 192,000
#              = 41.39M 参数
#
#    P/N 比率: 41.39M / 100K = 413.86 (ViT 可接受范围: [150, 500])
#
# 2. Tokenizer 配置 (I33 相对预算)
#    ----------------
#    image_size=64 (Tiny-ImageNet 原始尺寸)
#    min_patch_size=4
#    max_depth = log2(64/4) = 4
#    候选区域: 1+4+16+64+256 = 341 (5 尺度)
#
#    相对预算公式 (自动计算):
#      K_min = max(8, 0.005 * N) = max(8, 1.7) = 8
#      K_max = min(4096, 0.0535 * N) = min(4096, 18.2) = 19
#      实际 tokens: ~8-19 (覆盖率 2.35%-5.57%)
#
#    Elastic Budget 相对覆盖率:
#      ELASTIC_COVERAGE_MAX = 0.08 → deadzone 上界 ~27 tokens
#      ELASTIC_COVERAGE_MIN = 0.005 → 崩溃阈值 ~2 tokens
#
#    注意: --K-min 8 --K-max 64 已废弃，代码使用 constants.py 中的相对预算设计
#
# 3. 学习率缩放 (Linear Scaling Rule)
#    ---------------------------------
#    lr_base = 3e-4 @ batch_size=256 (ViT 标准)
#    lr = 3e-4 * (192/256) = 2.25e-4 (线性缩放)
#
# 4. VRAM 预算 (8GB - RTX 4070 Laptop)
#    ----------------------------------
#    M_params (FP16):  90.2 MB  (41.39M * 2)
#    M_gradients (FP32): 180.3 MB  (41.39M * 4)
#    M_optimizer:      360.6 MB  (41.39M * 8)
#    M_activations:    ~2.8 MB (checkpoint × AMP)
#    M_data:           ~18.0 MB
#    CUDA overhead:    ~500 MB
#    总计:             ~1.1 GB << 8GB ✓
#
#    优化因子:
#      AMP:        0.5× (FP16)
#      Checkpoint: 0.35× (65% 节省)
#      channels-last: 0.9×
#
# 5. 正则化参数
#    ------------
#    drop_path = 0.15 (ViT 标准)
#    dropout = 0.1 (embedding dropout)
#    label_smoothing = 0.1
#    weight_decay = 0.1 (AdamW decoupled)
#
# ============================================================================

$script = @"
uv run python examples/training/train_fractal_vit.py `
  --dataset tiny-imagenet `
  --image-size 64 `
  --epochs 100 `
  `
  <# ====================================================================== #> `
  <# 模型架构 (41.39M 参数 - 数学推导最优)                                      #> `
  <# 验证: dim=384, depth=10, heads=6 → P = 41.39M                              #> `
  <# dim_head = dim / heads = 384 / 6 = 64                                      #> `
  <# mlp_dim = 4 × dim = 1536 (SwiGLU 标准)                                     #> `
  <# P/N ratio = 413.86 (ViT 可接受范围: [150, 500])                            #> `
  <# ====================================================================== #> `
  --dim 384 `
  --depth 10 `
  --heads 6 `
  --mlp-dim 1536 `
  --pool cls `
  --ffn-type swiglu_level `
  `
  <# ====================================================================== #> `
  <# I33 相对预算: K_min/K_max 由代码自动计算 (constants.py)          #> `
  <#   K_min = max(8, 0.005 * N_candidates)                             #> `
  <#   K_max = min(4096, 0.0535 * N_candidates), 自适应                  #> `
  <#   64x64 图像: 实际 tokens = 8-19 (覆盖率 2.35%-5.57%)              #> `
  <# ====================================================================== #> `
  --tokenizer-type streaming_v3 `
  --min-patch-size 4 `
  `
  <# ====================================================================== #> `
  <# 训练配置 (batch_size=192)                                                  #> `
  <#   lr = 3e-4 × (192/256) = 2.25e-4 (线性缩放)                               #> `
  <# ====================================================================== #> `
  --batch-size 192 `
  --num-workers 4 `
  --lr 2.25e-4 `
  --weight-decay 0.1 `
  --warmup-epochs 10 `
  `
  <# ====================================================================== #> `
  <# 正则化参数                                                                 #> `
  <#   drop_path = 0.15 (ViT 标准)                                             #> `
  <# ====================================================================== #> `
  --dropout 0.1 `
  --emb-dropout 0.1 `
  --drop-path 0.15 `
  --label-smoothing 0.1 `
  `
  <# ====================================================================== #> `
  <# 数据增强 (Mixup + CutMix)                                                  #> `
  <# ====================================================================== #> `
  --mixup-alpha 0.4 `
  --cutmix-alpha 1.0 `
  --mixup-prob 0.5 `
  `
  <# ====================================================================== #> `
  <# 辅助损失配置                                                               #> `
  <#   Soft Entropy: 最大化尺度多样性                                           #> `
  <#   Elastic Budget: 使用相对预算 (constants.py)                             #> `
  <#     ELASTIC_COVERAGE_MAX = 0.08 → 上界 ≈ 0.08×N tokens                   #> `
  <#     ELASTIC_COVERAGE_MIN = 0.005 → 下界 ≈ 0.005×N tokens                 #> `
  <# ====================================================================== #> `
  --include-soft-entropy `
  --soft-entropy-mode maximize `
  --soft-entropy-weight 0.1 `
  --include-elastic-budget `
  --elastic-lambda-over 0.1 `
  --elastic-lambda-under 0.01 `
  --elastic-lambda-collapse 1.0 `
  `
  <# ====================================================================== #> `
  <# Splitter 温度退火 (cosine schedule)                                        #> `
  <#   T(t) = T_end + (T_start - T_end) × (1 + cos(πt/T)) / 2                   #> `
  <#   T_end=0.5 保持梯度流 (Jang et al., 2017)                                 #> `
  <# ====================================================================== #> `
  --splitter-temp-start 1.0 `
  --splitter-temp-end 0.5 `
  --splitter-temp-warmup 10 `
  `
  <# ====================================================================== #> `
  <# LCA Hilbert Bias (τ=1.5)                                                   #> `
  <#   B[i,j] = τ · LCAEmbed(LCA(i,j)) 提供位置偏置                             #> `
  <# ====================================================================== #> `
  --lca-temperature 1.5 `
  `
  <# ====================================================================== #> `
  <# 性能优化 (RTX 4070 Laptop 最大化)                                          #> `
  <#   AMP: FP16 计算，减少 VRAM 和加速                                          #> `
  <#   Gradient Checkpoint: 用计算换内存 (÷√12)                                 #> `
  <#   Compile: torch.compile 动态形状优化                                       #> `
  <#   Channels-Last: 卷积优化内存布局                                           #> `
  <#   TF32: Ampere+ GPU 加速                                                   #> `
  <# ====================================================================== #> `
  --use-amp `
  --gradient-checkpoint `
  --compile `
  --channels-last `
  --tf32 `
  --accum-steps 1 `
  --gradient-clip 1.0 `
  `
  <# ====================================================================== #> `
  <# 早停策略 (patience=15 防止过拟合)                                          #> `
  <# ====================================================================== #> `
  --patience 15 `
  --min-delta 0.001 `
  --exp-name tiny_imagenet_optimal_384d_10l_bs192
"@

# 执行训练脚本
Invoke-Expression $script
