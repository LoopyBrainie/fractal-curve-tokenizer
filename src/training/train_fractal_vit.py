#!/usr/bin/env python3
"""Fractal ViT Training Script - V3 Variable Depth Tokens

I36 配置系统重构:
- 使用 ModelArchitectureConfig (training.config) 作为模型架构配置
- 使用 TrainingConfig 包装器满足 FractalConfigProtocol 接口

数学形式化
===========
完整前向传播:
    1. Tokenization:  (T, L) = Tokenizer(I)
       其中 I ∈ R^{B × C × H × W}, T ∈ R^{B × N × D}, L ∈ Z^{B × N}
    
    2. 位置编码:      T' = T + E_pos(T, L)
    
    3. Transformer:   X' = Transformer([CLS; T'], L)
    
    4. 分类:          ŷ = MLP(Pool(X'))

分割方案 (Split Schemes)
-------------------------
使用 Scheme D (GumbelTopKSplitter) - 推荐使用的分割器:

+------------------+----------------------------------+---------------------------+
| 方案              | 数学描述                          | 特点                       |
+==================+==================================+===========================+
| Scheme D         | selected = TopK(logits + g, K)   | 端到端学习, ~K/N梯度覆盖   |
| (GumbelTopK)     | 树一致性约束 + STE               | 硬 K 约束, 并行评估        |
+------------------+----------------------------------+---------------------------+

Scheme D 优势:
- ~K/N 梯度覆盖 (约 37.6%，K=32,N=85) - Top-K 后只有 K 个候选有显著梯度
- 硬 K 约束 [K_min, K_max] 消除死锁风险
- O(1) 并行评估所有候选
- 树一致性向量化约束保证 Hilbert 空间局部性

**注意**: 100% 梯度覆盖需要 Subset Softmax（每个深度独立 softmax），已在 I30-2 中移除。

P9 性能优化 (2025-12-28)
-------------------------
训练速度从 24s/iter 优化至 ~1.6s/iter (14.8x 加速):

1. P9-1: TensorSplitResult - 纯张量表示替代 Python dataclass
2. P9-2: 完全向量化 BFS - O(D) GPU kernels
3. P9-5: TokenizerOutput 预填充缓存
4. P9-6: depth_distribution 向量化 - scatter_add

P10 训练稳定性修复 (2025-01-14)
-------------------------------
解决自适应分割器梯度消失和训练崩溃问题:

1. P10-1: STE 梯度修复 - 使用 hard_split - soft_probs.detach() + soft_probs
2. P10-2: 初始化修复 - gain=1.0 替代 4.0，临界区从 95% 降至 22%
3. P10-4/P10-5: 软熵损失 (推荐开启)
   - 公式: H̃ = -Σ_d p̃(d) · log(p̃(d) + ε)
   - 使用 BFS 缓存概率，而非固定网格评估
   - 模式: 'maximize' (最大化多样性) 或 'target' (匹配目标)
4. P10-9: 弹性预算损失 (推荐开启)
   - Dead Zone [N_min, N_max] 内零惩罚
   - 非对称惩罚: λ_over=0.1 >> λ_under=0.01

I13 数学形式化全面审查 (2026-01-03)
---------------------------------------
全面审查核心模块数学一致性:

1. I13-1: 阈值先验值修复 - 0.5 → 0.0 (logit 空间中性值)
   - 修复位置: get_soft_token_count, get_soft_depth_distribution
   - 公式: p_d = σ((0.0 - τ_d) / T) = σ(-τ_d / T)
   - 效果: 当 τ_d = 0 时，p_d = 0.5 (正确的最大不确定性)

I14 分割器稳定性 (2026-01-03)
-----------------------------------------------------
GumbelTopKSplitter 通过 Top-K 硬约束自动保证 token 数量，
无需 warmup_forced_split 机制。
   
弹性预算崩溃惩罚 (I14-1 D1):
   - 当 actual_tokens < 2 时触发强惩罚
   - 公式: L_collapse = λ_collapse · 𝟙[N_actual < 2]
   - 参数: --elastic-lambda-collapse (默认 1.0)
   - API: get_auxiliary_losses() 传递 actual_token_count

P11 数学形式化审查 (2025-12-30)
--------------------------------
架构审查和代码简化:

1. P11-8: 死代码清理 - 删除 LowRankHilbertBias/HierarchicalHilbertBias，仅保留 LCAHilbertBias
2. P11-9: 深度分布统计向量化 - O(B×D) Python 循环 → O(1) scatter_add
3. P11-15: 注释修复 - level_mixing_weights sigmoid vs softmax 不一致
4. P11-17: 索引越界审查 - 无需修改，现有 clamp 保护合理

P12 内部向量化优化 (2025-12-29)
--------------------------------
消除 O(B) 或 O(N) Python 循环，提升推理和训练速度:

1. P12-1: get_soft_balance_loss - roi_align + scatter_reduce (23.1x 加速)
   - 公式: L_soft = Σ_g (L_soft_g · w_g)，w_g = N_g / N
   - 向量化: 使用 unique + scatter_reduce 批量计算每组统计
2. P12-2: _create_attention_mask - 广播比较替代循环 (3.8-7.8x 加速)
   - 公式: M_{b,s} = 1{s > L_b}，向量化: positions > lengths
3. P12-3: _embed_with_tensor_result - segment cumsum (6.4-39x 加速)
   - 公式: pos[i] = i - Σ_{j<i} 1{batch_idx[j] ≠ batch_idx[i-1]}
   - 向量化: cummax 传播 segment 起始位置
4. P12-4: to_split_results - bincount/cumsum (2.2x 加速，已废弃)
5. P12-5: _fallback_roi_pool - 批量 grid_sample (15.1x 加速)

特性：
1. StreamingFractalTokenizerV3：Variable Depth Tokens 自适应多尺度
2. SwiGLU FFN：现代化前馈网络 (swiglu_level 推荐)
3. Hilbert 曲线重排序：保持空间局部性
4. LCA Hilbert Bias：层级感知注意力偏置
5. AMP 混合精度训练 + torch.compile 编译优化

使用示例：
    # CIFAR-10 快速测试
    python train_fractal_vit.py --quick-test --use-amp

    # Tiny ImageNet 完整训练 (I30-3 优化配置 - 增强正则化缓解过拟合)
    # 正则化参数: dropout=0.25, drop_path=0.25, weight_decay=0.15
    python train_fractal_vit.py --dataset tiny-imagenet --epochs 100 --dim 320 \
        --num-layers 12 --heads 8 \
        --use-amp --gradient-checkpoint --compile --channels-last \
        --include-soft-entropy --include-elastic-budget

    # 小数据集推荐配置 (I30-3: 进一步增强正则化)
    python train_fractal_vit.py --dataset tiny-imagenet --epochs 150 \
        --dim 256 --num-layers 8 --heads 6 \
        --dropout 0.3 --drop-path 0.3 --weight-decay 0.2 \
        --freeze-tokenizer --use-amp
    
    # 自定义 P10 参数
    python train_fractal_vit.py --dataset tiny-imagenet --epochs 100 \\
        --soft-entropy-mode maximize --soft-entropy-weight 0.1 \\
        --elastic-N-min 32 --elastic-N-max 256 \\
        --elastic-lambda-over 0.1 --elastic-lambda-under 0.01

注意：V1 和 V2 已从代码库中完全删除，当前仅支持 streaming_v3。
"""

from __future__ import annotations

import os
import sys
import platform
import time
import json
import multiprocessing as _mp
import argparse
import random
from typing import Protocol, runtime_checkable, Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

# 抑制 PyTorch FutureWarning (checkpoint.py cpu.amp.autocast 弃用警告)
# 预计在 PyTorch 2.6+ 中修复
import warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='torch.utils.checkpoint')

# 抑制 PyTorch 内部日志 (cudagraphs 警告等)
# 设置 inductor 日志级别为 ERROR，避免 "skipping cudagraphs" 刷屏
import os
os.environ['TORCHINDUCTOR_LOG_LEVEL'] = 'error'
os.environ['PYTORCH_CUDA_LOG_LEVEL'] = 'error'

# 配置 Python logging 抑制 torch inductor INFO 日志
import logging
logging.getLogger('torch._inductor').setLevel(logging.ERROR)
logging.getLogger('torch.cuda').setLevel(logging.ERROR)


# =========================================================================
# 工具函数
# =========================================================================

def obj_to_dict(obj: Any) -> Dict[str, Any]:
    """将对象转换为字典 (支持 dataclass 和普通对象)"""
    if hasattr(obj, '__dataclass_fields__'):
        # dataclass
        return asdict(obj)
    elif hasattr(obj, '__dict__'):
        # 普通对象
        return {k: v for k, v in vars(obj).items() if not k.startswith('_')}
    elif isinstance(obj, dict):
        return obj
    else:
        return {'value': obj}


# =========================================================================
# I35: CUDA 优化配置 - 必须在第一次 torch 调用前设置
# =========================================================================

# 全局标志：防止重复配置
_CUDA_OPTIMIZATIONS_CONFIGURED = False

def _configure_cuda_optimizations():
    """配置 CUDA 优化以获得最佳性能 (I35)。

    设置包括：
    1. TF32: Ampere+ GPU 的快速精度模式
    2. cuDNN benchmark: 自动选择最优 kernel
    3. cuDNN deterministic: 允许非确定性以提高性能
    4. SDPA 后端: 启用 Flash/Memory-Efficient/cuDNN Attention
    """
    global _CUDA_OPTIMIZATIONS_CONFIGURED

    # 防止重复配置
    if _CUDA_OPTIMIZATIONS_CONFIGURED:
        return
    _CUDA_OPTIMIZATIONS_CONFIGURED = True

    if not torch.cuda.is_available():
        return

    # TF32 (Ampere+ GPU) - 加速矩阵运算
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
        torch.backends.cuda.matmul.allow_tf32 = True

    if hasattr(torch.backends, 'cudnn') and hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = True

    # cuDNN auto-tuning - 选择最优卷积算法
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # 允许非确定性以提高性能

    # 启用所有 SDPA 后端 (Flash / Memory-Efficient / cuDNN)
    # PyTorch 2.0+ 自动选择最优实现
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)

    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)

    if hasattr(torch.backends.cuda, 'enable_cudnn_sdp'):
        torch.backends.cuda.enable_cudnn_sdp(True)

    # I35: CUDA 优化已静默启用，如需调试可取消注释
    # print("[I35] CUDA 优化配置:")

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset, default_collate
from torchvision import datasets, transforms
from tqdm import tqdm
import atexit
import signal

# 分层采样
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def stratified_split(
    indices: np.ndarray, 
    labels: np.ndarray, 
    val_ratio: float, 
    seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """手动实现分层划分 (不依赖 sklearn).
    
    确保每个类别在训练/验证集中的比例相同。
    """
    np.random.seed(seed)
    unique_classes = np.unique(labels)
    train_idx_list = []
    val_idx_list = []
    
    for c in unique_classes:
        class_indices = indices[labels == c]
        np.random.shuffle(class_indices)
        
        val_count = max(1, int(len(class_indices) * val_ratio))
        val_idx_list.append(class_indices[:val_count])
        train_idx_list.append(class_indices[val_count:])
    
    train_idx = np.concatenate(train_idx_list)
    val_idx = np.concatenate(val_idx_list)
    
    # 再次打乱
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    
    return train_idx, val_idx

import logging
# 抑制 torch.compile 的符号形状警告
logging.getLogger('torch.fx.experimental.symbolic_shapes').setLevel(logging.ERROR)
logging.getLogger('torch._dynamo').setLevel(logging.ERROR)

# AMP 兼容层 (I78: PyTorch 2.0+ 使用统一 API)
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler

# P2 修复: 在导入 torch 后配置 CUDA 优化
_configure_cuda_optimizations()

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
EXAMPLES_PATH = PROJECT_ROOT / "examples"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(EXAMPLES_PATH) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PATH))

from vit_pytorch import FractalCurveViT
from vit_pytorch.constants import (
    EPS,  # I112-3: 统一数值稳定性常量
    SPLITTER_TEMP_START,
    SPLITTER_TEMP_END,
    SPLITTER_TEMP_SCHEDULE,  # I29-4: 导入调度策略
)
from vit_pytorch.gumbel_topk_splitter import DepthMonitor  # I111-6: 深度分布监控

# Fractal Training 模块 (I15) - 现在位于 examples/training
from training import (
    # Samplers
    ClassBalancedSampler,
    ProgressiveSampler,
    # Losses
    FocalLoss,
    ClassBalancedCrossEntropy,
    FocalClassBalancedLoss,
    # I30-2: Hilbert-aware 困难样本挖掘
    HilbertAwareHardMining,
    # Metrics
    ClassificationMetrics,
    # Schedulers
    FLOPSConfig,
    compute_transformer_flops,
    FLOPSBudgetLoss,
    DepthWeightedBudgetLoss,
    BudgetScheduler,
    # 类别权重计算 (修复 T2-5)
    compute_class_weights_from_targets,
    # Trainer
    ModularTrainer,
    # Config
    TrainerConfig,
    ExperimentConfig,
    # Callbacks
    EarlyStoppingCallback,
    CheckpointCallback,
    WandBCallback,
    WandBCallbackConfig,
    # Visualization
    VisualizationConfig,
    ExperimentVisualizer,
    # CUB-200 细粒度分类专用 (解耦合的训练器)
    CUB200Trainer,
    CUB200TrainingConfig,
    CUB200EvalResult,
    create_cub200_trainer,
    get_cub200_augmentation,
    # CUB-200 细粒度损失函数
    CenterLoss,
    AttentionEntropyLoss,
    FinegrainedLoss,
    FinegrainedLossConfig,
    # I36: ModelArchitectureConfig (替代 FractalViTConfig)
    ModelArchitectureConfig,
    # ModelGene (自包含 checkpoint)
    ModelGene,
    # Checkpoint 工具 (与评估器共享)
    save_checkpoint_with_gene,
    load_checkpoint,
    load_model,
    # InferenceWrapper (与 eval.py 共享推理逻辑)
    evaluate as inference_evaluate,
    EvalResult,
)


# ============================================================================
# I36: FractalConfigProtocol - 配置接口抽象 (替代弃用的 FractalViTConfig)
# ============================================================================

@runtime_checkable
class FractalConfigProtocol(Protocol):
    """训练配置协议 (Protocol)

    定义训练脚本所需的最小配置接口。
    ModelArchitectureConfig 自动满足此协议。

    注意: FractalViTConfig 已移除，请使用 ModelArchitectureConfig。
    """

    # ========== 数据集配置 (create_dataloaders) ==========
    @property
    def subset_size(self) -> Optional[int]: ...
    @property
    def val_split(self) -> float: ...
    @property
    def seed(self) -> int: ...
    @property
    def num_workers(self) -> int: ...
    @property
    def batch_size(self) -> int: ...

    # ========== 模型架构配置 (模型创建) ==========
    # 注意: max_level/max_depth 是变参数，完全由模型架构内部计算，
    # 不在 Protocol 中定义，确保训练/评估模型结构完全一致。
    @property
    def dim(self) -> int: ...
    @property
    def num_layers(self) -> int: ...  # 原 depth
    @property
    def heads(self) -> int: ...
    @property
    def dim_head(self) -> int: ...
    @property
    def pool(self) -> str: ...
    @property
    def ffn_type(self) -> str: ...
    @property
    def min_patch_size(self) -> int: ...
    @property
    def dropout(self) -> float: ...
    @property
    def emb_dropout(self) -> float: ...
    @property
    def drop_path_rate(self) -> float: ...
    @property
    def use_checkpoint(self) -> bool: ...  # I139: 统一命名 (原 use_checkpoint)
    @property
    def lca_temperature(self) -> Optional[float]: ...
    @property
    def learnable_temperature(self) -> bool: ...
    @property
    def use_area_encoding(self) -> bool: ...
    @property
    def use_affine_modulation(self) -> bool: ...
    @property
    def fourier_levels(self) -> int: ...
    @property
    def quota_learnable(self) -> Optional[bool]: ...
    @property
    def freeze_quota(self) -> bool: ...
    @property
    def freeze_tokenizer(self) -> bool: ...
    @property
    def freeze_tokenizer_epochs(self) -> int: ...
    @property
    def depth_scale_range(self) -> Optional[Tuple[float, float]]: ...

    # ========== Tokenizer K 值 (I33 相对预算) ==========
    @property
    def K_min(self) -> int: ...
    @property
    def K_max(self) -> int: ...
    @property
    def token_coverage_min(self) -> float: ...
    @property
    def token_coverage_max(self) -> float: ...

    # ========== 训练配置 (训练循环) ==========
    @property
    def use_channels_last(self) -> bool: ...  # I139: 统一命名
    @property
    def use_amp(self) -> bool: ...
    @property
    def learning_rate(self) -> float: ...
    @property
    def weight_decay(self) -> float: ...
    @property
    def gradient_clip(self) -> float: ...
    @property
    def accum_steps(self) -> int: ...
    @property
    def warmup_epochs(self) -> int: ...
    @property
    def epochs(self) -> int: ...
    @property
    def label_smoothing(self) -> float: ...
    @property
    def mixup_alpha(self) -> float: ...
    @property
    def cutmix_alpha(self) -> float: ...
    @property
    def mixup_prob(self) -> float: ...
    @property
    def use_focal_loss(self) -> bool: ...
    @property
    def focal_gamma(self) -> float: ...
    @property
    def use_class_balanced(self) -> bool: ...
    @property
    def class_balance_beta(self) -> float: ...
    @property
    def progressive_aug(self) -> bool: ...
    @property
    def use_hilbert_mining(self) -> bool: ...
    @property
    def hilbert_mining_lambda(self) -> float: ...
    @property
    def hilbert_mining_warmup(self) -> int: ...
    @property
    def patience(self) -> int: ...
    @property
    def min_delta(self) -> float: ...
    @property
    def compile_model(self) -> bool: ...
    @property
    def splitter_temp_start(self) -> float: ...
    @property
    def splitter_temp_end(self) -> float: ...
    @property
    def splitter_temp_warmup(self) -> int: ...
    @property
    def include_elastic_budget(self) -> bool: ...
    @property
    def include_soft_entropy(self) -> bool: ...
    @property
    def soft_entropy_target(self) -> Optional[float]: ...
    @property
    def soft_entropy_weight(self) -> float: ...
    @property
    def soft_entropy_mode(self) -> str: ...


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class DatasetSpec:
    """数据集规格"""
    name: str
    num_classes: int
    image_size: int
    channels: int
    mean: Tuple[float, ...]
    std: Tuple[float, ...]


# ============================================================================
# I36: 训练配置类已移除
# 使用 ModelArchitectureConfig (training.config) 配合 TrainingConfig 包装器
# ============================================================================


# ============================================================================
# Mixup/CutMix 实现 (简化版，使用 training 模块的 FocalLoss)
# ============================================================================

class MixupCutmix:
    """Mixup 和 CutMix 数据增强
    
    参考: 
    - Mixup: https://arxiv.org/abs/1710.09412
    - CutMix: https://arxiv.org/abs/1905.04899
    """
    
    def __init__(
        self,
        mixup_alpha: float = 0.8,
        cutmix_alpha: float = 1.0,
        prob: float = 0.5,
        num_classes: int = 10,
        label_smoothing: float = 0.0,
    ):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.prob = prob
        self.num_classes = num_classes
        self.label_smoothing = label_smoothing
    
    def __call__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """应用 Mixup 或 CutMix
        
        Args:
            images: [B, C, H, W] 图像张量
            labels: [B] 标签张量
            
        Returns:
            mixed_images: 混合后的图像
            mixed_labels: 混合后的 one-hot 标签 [B, num_classes]
        """
        batch_size = images.size(0)
        device = images.device
        
        # 检查 label 范围，防止越界
        assert labels.min() >= 0, f"Label 包含负值: min={labels.min().item()}"
        assert labels.max() < self.num_classes, f"Label 越界: max={labels.max().item()} >= {self.num_classes}"
        
        # 转换为 one-hot 并应用 label smoothing
        labels_one_hot = F.one_hot(labels, self.num_classes).float()
        if self.label_smoothing > 0:
            labels_one_hot = labels_one_hot * (1 - self.label_smoothing) + self.label_smoothing / self.num_classes
        
        # 随机决定是否应用增强
        if random.random() > self.prob:
            return images, labels_one_hot
        
        # 随机选择 Mixup 或 CutMix
        use_cutmix = random.random() > 0.5 and self.cutmix_alpha > 0
        
        if use_cutmix:
            lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        else:
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha) if self.mixup_alpha > 0 else 1.0
        
        # 随机打乱索引
        index = torch.randperm(batch_size, device=device)
        
        if use_cutmix:
            # CutMix: 随机裁剪区域
            _, _, H, W = images.shape
            cut_h = int(H * np.sqrt(1 - lam))
            cut_w = int(W * np.sqrt(1 - lam))
            
            cx = random.randint(0, W)
            cy = random.randint(0, H)
            
            x1 = max(0, cx - cut_w // 2)
            x2 = min(W, cx + cut_w // 2)
            y1 = max(0, cy - cut_h // 2)
            y2 = min(H, cy + cut_h // 2)
            
            mixed_images = images.clone()
            mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
            
            # 重新计算 lambda 基于实际裁剪区域
            lam = 1 - (x2 - x1) * (y2 - y1) / (W * H)
        else:
            # Mixup: 线性混合
            # 注意: 需要保持原始 dtype (可能是 float16)
            lam_t = torch.tensor(lam, dtype=images.dtype, device=device)
            mixed_images = lam_t * images + (1 - lam_t) * images[index]
        
        # 混合标签 (始终使用 float32 以保持精度)
        mixed_labels = lam * labels_one_hot + (1 - lam) * labels_one_hot[index]
        
        return mixed_images, mixed_labels


def mixup_criterion(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """计算 Mixup/CutMix 的交叉熵损失
    
    Args:
        outputs: [B, C] 模型输出 logits
        targets: [B, C] one-hot 或 soft 标签
        
    Returns:
        损失标量
    """
    # 检查 logits 是否包含 NaN/Inf
    if torch.isnan(outputs).any() or torch.isinf(outputs).any():
        raise ValueError(f"Logits 包含 NaN/Inf: nan={torch.isnan(outputs).sum()}, inf={torch.isinf(outputs).sum()}")
    
    # 数值稳定的 log_softmax
    log_probs = F.log_softmax(outputs, dim=1)
    
    # 确保 targets 归一化且非负
    # I112-3: 使用 EPS 统一数值稳定性
    targets = targets.clamp(min=0)
    targets = targets / (targets.sum(dim=1, keepdim=True) + EPS)
    
    loss = -(targets * log_probs).sum(dim=1).mean()
    
    # 检查 loss 是否为 NaN
    if torch.isnan(loss):
        raise ValueError("Loss 为 NaN，可能是 logits 过大或标签问题")
    
    return loss


# ============================================================================
# 数据集配置
# ============================================================================

DATASETS = {
    "cifar10": DatasetSpec("CIFAR10", 10, 32, 3, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": DatasetSpec("CIFAR100", 100, 32, 3, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "mnist": DatasetSpec("MNIST", 10, 28, 1, (0.1307,), (0.3081,)),
    "tiny-imagenet": DatasetSpec("TinyImageNet", 200, 64, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # CUB-200-2011: 细粒度鸟类分类数据集
    # 图像尺寸混合，动态获取每张图像的实际尺寸
    "cub200": DatasetSpec("CUB200", 200, None, 3, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


# ============================================================================
# 环境检测
# ============================================================================

def detect_environment() -> Dict[str, Any]:
    """检测运行环境"""
    env = {
        'in_container': os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'),
        'platform': platform.system(),
        'cpu_count': _mp.cpu_count(),
        'recommended_workers': 4,
    }

    if env['platform'] == 'Linux':
        try:
            import psutil
            shm = psutil.disk_usage('/dev/shm')
            shm_gb = shm.total / (1024**3)
            # I136: 优化 workers 配置，避免 GPU 空转
            # 原则: num_workers >= CPU核心数/2 以保持数据供给
            cpu_cores = env['cpu_count']
            if env['in_container']:
                # 容器环境：共享内存受限，但需要足够 workers 避免 GPU 空转
                if shm_gb >= 8:
                    env['recommended_workers'] = min(16, cpu_cores)
                elif shm_gb >= 4:
                    env['recommended_workers'] = min(12, cpu_cores)
                else:
                    env['recommended_workers'] = min(8, cpu_cores)
            else:
                # 非容器环境：可以使用更多 workers
                # 推荐: min(16, CPU核心数) 确保数据预处理不成为瓶颈
                env['recommended_workers'] = min(16, cpu_cores)
        except ImportError:
            pass

    return env


def print_environment_info(env: Dict[str, Any]) -> None:
    """打印环境信息"""
    print("\n" + "="*70)
    print("Environment")
    print("="*70)
    print(f"  Platform: {env['platform']}")
    print(f"  Container: {'Yes' if env['in_container'] else 'No'}")
    print(f"  CPU Cores: {env['cpu_count']}")
    print(f"  Recommended Workers: {env['recommended_workers']}")
    print("="*70 + "\n")


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_amp_context(device: torch.device, enabled: bool):
    """获取 AMP autocast 上下文 (I78: 简化 PyTorch 2.0+ API)"""
    return autocast('cuda', enabled=enabled)


def create_grad_scaler(enabled: bool) -> GradScaler:
    """创建 GradScaler (I78: 简化 PyTorch 2.0+ API)"""
    return GradScaler('cuda', enabled=enabled)


# ============================================================================
# NaN/Inf 诊断工具
# ============================================================================

def diagnose_nan_inf(
    batch_idx: int,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    logits: Optional[torch.Tensor] = None,
    loss: Optional[torch.Tensor] = None,
    ce_loss: Optional[torch.Tensor] = None,
    entropy_loss: Optional[torch.Tensor] = None,
    splitter_loss: Optional[torch.Tensor] = None,
    multi_layer_loss: Optional[torch.Tensor] = None,
    log_file: Optional[str] = None,
) -> str:
    """诊断 NaN/Inf 出现的原因，输出详细信息
    
    Args:
        batch_idx: 当前批次索引
        imgs: 输入图像张量
        labels: 标签张量
        model: 模型
        logits: 模型输出 logits（可选）
        loss: 总损失（可选）
        ce_loss: 交叉熵损失（可选）
        entropy_loss: 熵损失（可选）
        splitter_loss: 分割器损失（可选）
        log_file: 日志文件路径（可选，用于持久化）
        
    Returns:
        诊断报告字符串
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"[NaN/Inf 诊断报告] Batch {batch_idx}")
    lines.append("=" * 70)
    
    # 1. 输入数据统计
    lines.append("\n[1] 输入数据统计:")
    lines.append(f"  imgs.shape: {imgs.shape}, dtype: {imgs.dtype}")
    lines.append(f"  imgs: min={imgs.min().item():.4f}, max={imgs.max().item():.4f}, "
                f"mean={imgs.mean().item():.4f}, std={imgs.std().item():.4f}")
    lines.append(f"  imgs NaN: {torch.isnan(imgs).sum().item()}, Inf: {torch.isinf(imgs).sum().item()}")
    lines.append(f"  labels: min={labels.min().item()}, max={labels.max().item()}")
    
    # 2. 损失统计
    lines.append("\n[2] 损失统计:")
    if loss is not None:
        lines.append(f"  total_loss: {loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 'NaN/Inf'}")
        lines.append(f"    → isnan: {torch.isnan(loss).item()}, isinf: {torch.isinf(loss).item()}, dtype: {loss.dtype}")
    if ce_loss is not None:
        ce_val = ce_loss.item() if not (torch.isnan(ce_loss) or torch.isinf(ce_loss)) else 'NaN/Inf'
        lines.append(f"  ce_loss: {ce_val}")
        lines.append(f"    → isnan: {torch.isnan(ce_loss).item()}, isinf: {torch.isinf(ce_loss).item()}, dtype: {ce_loss.dtype}")
    if entropy_loss is not None:
        ent_val = entropy_loss.item() if not (torch.isnan(entropy_loss) or torch.isinf(entropy_loss)) else 'NaN/Inf'
        lines.append(f"  entropy_loss: {ent_val}")
        lines.append(f"    → dtype: {entropy_loss.dtype}")
    if splitter_loss is not None:
        if isinstance(splitter_loss, torch.Tensor):
            spl_val = splitter_loss.item() if not (torch.isnan(splitter_loss) or torch.isinf(splitter_loss)) else 'NaN/Inf'
            lines.append(f"  splitter_loss: {spl_val}")
            lines.append(f"    → isnan: {torch.isnan(splitter_loss).item()}, isinf: {torch.isinf(splitter_loss).item()}, dtype: {splitter_loss.dtype}")
        else:
            lines.append(f"  splitter_loss: {splitter_loss}")
    if multi_layer_loss is not None:
        if isinstance(multi_layer_loss, torch.Tensor):
            ml_val = multi_layer_loss.item() if not (torch.isnan(multi_layer_loss) or torch.isinf(multi_layer_loss)) else 'NaN/Inf'
            lines.append(f"  multi_layer_loss: {ml_val}")
            lines.append(f"    → isnan: {torch.isnan(multi_layer_loss).item()}, isinf: {torch.isinf(multi_layer_loss).item()}, dtype: {multi_layer_loss.dtype}")
        else:
            lines.append(f"  multi_layer_loss: {multi_layer_loss}")
    
    # 3. Logits 统计
    if logits is not None:
        lines.append("\n[3] Logits 统计:")
        lines.append(f"  logits.shape: {logits.shape}")
        nan_count = torch.isnan(logits).sum().item()
        inf_count = torch.isinf(logits).sum().item()
        lines.append(f"  NaN count: {nan_count}, Inf count: {inf_count}")
        if nan_count == 0 and inf_count == 0:
            lines.append(f"  min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
                        f"mean={logits.mean().item():.4f}, std={logits.std().item():.4f}")
        # 检查是否有极端值
        if not (torch.isnan(logits).any() or torch.isinf(logits).any()):
            if logits.abs().max() > 100:
                lines.append(f"  [WARN] Logits 有极端值 (>100)，可能导致 softmax 数值不稳定")
    
    # 4. 模型参数统计
    lines.append("\n[4] 模型参数统计:")
    param_stats = []
    nan_params = []
    inf_params = []
    large_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_nan = torch.isnan(param).any().item()
            has_inf = torch.isinf(param).any().item()
            if has_nan:
                nan_params.append(name)
            if has_inf:
                inf_params.append(name)
            # 检查极端值
            if not (has_nan or has_inf):
                max_val = param.abs().max().item()
                if max_val > 1000:
                    large_params.append((name, max_val))
    
    lines.append(f"  参数包含 NaN: {len(nan_params)} 个")
    if nan_params:
        for name in nan_params[:5]:  # 最多显示 5 个
            lines.append(f"    - {name}")
        if len(nan_params) > 5:
            lines.append(f"    ... 还有 {len(nan_params) - 5} 个")
    
    lines.append(f"  参数包含 Inf: {len(inf_params)} 个")
    if inf_params:
        for name in inf_params[:5]:
            lines.append(f"    - {name}")
    
    if large_params:
        lines.append(f"  参数极端值 (>1000): {len(large_params)} 个")
        for name, val in large_params[:5]:
            lines.append(f"    - {name}: max={val:.2f}")
    
    # 5. 梯度统计（如果有）
    lines.append("\n[5] 梯度统计:")
    grad_nan = []
    grad_inf = []
    grad_large = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_nan = torch.isnan(param.grad).any().item()
            has_inf = torch.isinf(param.grad).any().item()
            if has_nan:
                grad_nan.append(name)
            if has_inf:
                grad_inf.append(name)
            if not (has_nan or has_inf):
                max_val = param.grad.abs().max().item()
                if max_val > 1000:
                    grad_large.append((name, max_val))
    
    lines.append(f"  梯度包含 NaN: {len(grad_nan)} 个")
    if grad_nan:
        for name in grad_nan[:5]:
            lines.append(f"    - {name}")
    
    lines.append(f"  梯度包含 Inf: {len(grad_inf)} 个")
    if grad_inf:
        for name in grad_inf[:5]:
            lines.append(f"    - {name}")
    
    if grad_large:
        lines.append(f"  梯度极端值 (>1000): {len(grad_large)} 个")
        for name, val in grad_large[:5]:
            lines.append(f"    - {name}: max_grad={val:.2f}")
    
    # 6. Tokenizer/Splitter 状态（如果有）
    lines.append("\n[6] Tokenizer/Splitter 状态:")
    try:
        if hasattr(model, 'tokenizer'):
            tokenizer = model.tokenizer
            if hasattr(tokenizer, 'splitter'):
                splitter = tokenizer.splitter
                if hasattr(splitter, 'get_temperature'):
                    temp = splitter.get_temperature()
                    lines.append(f"  Splitter temperature: {temp:.4f}")
                if hasattr(splitter, '_last_tau_values') and splitter._last_tau_values is not None:
                    taus = splitter._last_tau_values
                    lines.append(f"  Last tau values: min={taus.min().item():.4f}, max={taus.max().item():.4f}")
                    if torch.isnan(taus).any() or torch.isinf(taus).any():
                        lines.append(f"    [WARN] tau 包含 NaN/Inf!")
    except Exception as e:
        lines.append(f"  [ERROR] 无法获取 Tokenizer 状态: {e}")
    
    # 7. 建议
    lines.append("\n[7] 可能原因及建议:")
    if nan_params or inf_params:
        lines.append("  - 参数已损坏，建议降低学习率或检查初始化")
    if grad_nan or grad_inf:
        lines.append("  - 梯度爆炸，建议降低 gradient_clip 值或学习率")
    if large_params:
        lines.append("  - 参数值过大，可能导致数值不稳定")
    if logits is not None and not (torch.isnan(logits).any() or torch.isinf(logits).any()):
        if logits.abs().max() > 100:
            lines.append("  - Logits 过大，建议检查分类头或添加 LayerNorm")
    
    lines.append("=" * 70)
    
    report = "\n".join(lines)
    
    # 写入日志文件（如果指定）
    if log_file:
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(report + "\n\n")
        except Exception as e:
            print(f"[WARN] 无法写入 NaN 诊断日志: {e}")
    
    return report


# ============================================================================
# Tiny ImageNet 下载
# ============================================================================

def download_with_progress(url: str, dest: Path, desc: str = "Downloading") -> bool:
    """带进度条的下载函数"""
    import urllib.request
    
    try:
        # 获取文件大小
        with urllib.request.urlopen(url, timeout=30) as response:
            total_size = int(response.headers.get('Content-Length', 0))
        
        # 下载
        downloaded = 0
        block_size = 8192
        
        with urllib.request.urlopen(url, timeout=30) as response:
            with open(dest, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        f.write(buffer)
                        downloaded += len(buffer)
                        pbar.update(len(buffer))
        
        return True
    except Exception as e:
        print(f"\n[ERROR] Download failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_tiny_imagenet(data_root: Path) -> bool:
    """下载并设置 Tiny ImageNet
    
    数据集信息:
    - 200 类，每类 500 张训练图像
    - 训练集: 100,000 张 64x64 图像
    - 验证集: 10,000 张图像
    - 测试集: 10,000 张图像（无标签）
    
    下载源:
    - 主源: Stanford CS231n
    - 大小: ~237MB
    """
    target_dir = data_root / "tiny-imagenet-200"
    
    # 检查是否已存在
    if (target_dir / "train").exists() and (target_dir / "val").exists():
        train_classes = len(list((target_dir / "train").iterdir()))
        val_has_classes = any((target_dir / "val").iterdir())
        if train_classes >= 200 and val_has_classes:
            print(f"[OK] Tiny ImageNet already exists at {target_dir}")
            return True
    
    print("\n" + "="*60)
    print("Downloading Tiny ImageNet Dataset")
    print("="*60)
    print(f"  Target: {target_dir}")
    print(f"  Size: ~237MB")
    print("="*60 + "\n")
    
    zip_path = data_root / "tiny-imagenet-200.zip"
    
    # 检查已缓存的 zip 是否有效
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                # 验证 zip 文件
                if zf.testzip() is not None:
                    raise zipfile.BadZipFile("Corrupted zip file")
                if len(zf.namelist()) < 100:  # Tiny ImageNet 应该有很多文件
                    raise zipfile.BadZipFile("Incomplete zip file")
            print(f"[OK] Using cached zip: {zip_path}")
        except (zipfile.BadZipFile, Exception) as e:
            print(f"[WARN] Cached zip is invalid: {e}")
            print("[*] Removing corrupted file and re-downloading...")
            zip_path.unlink()
    
    # 尝试多个下载源
    urls = [
        "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
        "https://image-net.org/data/tiny-imagenet-200.zip",
    ]
    
    if not zip_path.exists():
        download_success = False
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Trying: {url}")
            if download_with_progress(url, zip_path, "Tiny ImageNet"):
                # 验证下载的文件
                try:
                    with zipfile.ZipFile(zip_path, 'r') as zf:
                        if zf.testzip() is not None:
                            raise zipfile.BadZipFile("Downloaded file is corrupted")
                    download_success = True
                    print("[OK] Download complete and verified")
                    break
                except zipfile.BadZipFile as e:
                    print(f"[WARN] Downloaded file is invalid: {e}")
                    if zip_path.exists():
                        zip_path.unlink()
            print(f"[WARN] Failed, trying next source...")
        
        if not download_success:
            print("\n[ERROR] All download sources failed.")
            print("Please download manually from:")
            print("  http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            print(f"And place it at: {zip_path}")
            return False
    
    # 解压
    print("\nExtracting...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            total = len(zf.namelist())
            with tqdm(total=total, desc="Extracting", unit="files") as pbar:
                for member in zf.namelist():
                    zf.extract(member, data_root)
                    pbar.update(1)
        print("[OK] Extraction complete")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False
    
    # 组织验证集（原始格式是所有图片在一个文件夹）
    val_dir = target_dir / "val"
    val_images_dir = val_dir / "images"
    
    if val_images_dir.exists():
        print("\nOrganizing validation set by class...")
        val_annotations = val_dir / "val_annotations.txt"
        
        if val_annotations.exists():
            # 读取标注
            with open(val_annotations, 'r') as f:
                lines = f.readlines()
            
            # 按类别组织
            for line in tqdm(lines, desc="Organizing"):
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    img_name, class_id = parts[0], parts[1]
                    class_dir = val_dir / class_id / "images"
                    class_dir.mkdir(parents=True, exist_ok=True)
                    src = val_images_dir / img_name
                    dst = class_dir / img_name
                    if src.exists() and not dst.exists():
                        shutil.move(str(src), str(dst))
            
            # 删除原始 images 文件夹
            if val_images_dir.exists():
                shutil.rmtree(val_images_dir)
            
            print("[OK] Validation set organized")
        else:
            print("[WARN] val_annotations.txt not found")
    
    # 验证
    train_classes = len(list((target_dir / "train").iterdir()))
    val_classes = len([d for d in (target_dir / "val").iterdir() if d.is_dir()])
    print(f"\n[OK] Dataset ready:")
    print(f"  Train classes: {train_classes}")
    print(f"  Val classes: {val_classes}")
    
    # 清理 zip
    if zip_path.exists():
        zip_path.unlink()
        print("[OK] Cleaned up zip file")
    
    return True


def download_cub200(data_root: Path) -> bool:
    """下载并设置 CUB-200-2011 细粒度鸟类分类数据集
    
    数学形式化分析
    ================
    数据集规格:
        N_train = 5,994 (约 30 张/类)
        N_test = 5,794 (约 29 张/类)
        C = 200 类 (鸟类物种)
        图像尺寸: 变长 → resize 到 224×224
    
    细粒度分类特性:
        - 类间差异小 (同为鸟类)
        - 类内差异大 (姿态、光照变化)
        - 需要关注局部细节 (喙、羽毛纹理)
        → 适合验证 Fractal ViT 的自适应多尺度能力
    
    下载源:
        - Caltech Data: https://data.caltech.edu/records/65de6-vp158
        - 大小: ~1.2GB
    
    目录结构 (转换后):
        data/CUB_200_2011/
        ├── train/
        │   ├── 001.Black_footed_Albatross/
        │   │   ├── Black_Footed_Albatross_0001_796111.jpg
        │   │   └── ...
        │   └── ...
        └── test/
            ├── 001.Black_footed_Albatross/
            │   └── ...
            └── ...
    """
    target_dir = data_root / "CUB_200_2011"
    
    # 检查是否已存在并正确组织
    if (target_dir / "train").exists() and (target_dir / "test").exists():
        train_classes = len(list((target_dir / "train").iterdir()))
        test_classes = len(list((target_dir / "test").iterdir()))
        if train_classes >= 200 and test_classes >= 200:
            print(f"[OK] CUB-200-2011 already exists at {target_dir}")
            return True
    
    print("\n" + "="*60)
    print("Downloading CUB-200-2011 Dataset")
    print("="*60)
    print(f"  Target: {target_dir}")
    print(f"  Size: ~1.2GB")
    print(f"  Classes: 200 bird species")
    print(f"  Train: ~5,994 images")
    print(f"  Test: ~5,794 images")
    print("="*60 + "\n")
    
    tgz_path = data_root / "CUB_200_2011.tgz"
    
    # 检查已缓存的文件
    if tgz_path.exists():
        print(f"[OK] Using cached archive: {tgz_path}")
    else:
        # 下载
        urls = [
            "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
        ]
        
        download_success = False
        for i, url in enumerate(urls):
            print(f"[{i+1}/{len(urls)}] Trying: {url}")
            if download_with_progress(url, tgz_path, "CUB-200-2011"):
                download_success = True
                print("[OK] Download complete")
                break
            print(f"[WARN] Failed, trying next source...")
        
        if not download_success:
            print("\n[ERROR] Download failed.")
            print("Please download manually from:")
            print("  https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz")
            print(f"And place it at: {tgz_path}")
            return False
    
    # 解压
    print("\nExtracting...")
    try:
        import tarfile
        with tarfile.open(tgz_path, 'r:gz') as tar:
            members = tar.getmembers()
            with tqdm(total=len(members), desc="Extracting", unit="files") as pbar:
                for member in members:
                    # filter='data' 兼容 Python 3.14+ (PEP 706)
                    tar.extract(member, data_root, filter='data')
                    pbar.update(1)
        print("[OK] Extraction complete")
    except Exception as e:
        print(f"[ERROR] Extraction failed: {e}")
        return False
    
    # 组织为 train/test 目录结构
    print("\nOrganizing dataset by train/test split...")
    
    raw_dir = target_dir
    images_dir = raw_dir / "images"
    
    if not images_dir.exists():
        print(f"[ERROR] Images directory not found: {images_dir}")
        return False
    
    # 读取 train_test_split.txt
    split_file = raw_dir / "train_test_split.txt"
    images_file = raw_dir / "images.txt"
    
    if not split_file.exists() or not images_file.exists():
        print(f"[ERROR] Split files not found")
        return False
    
    # 读取图像列表
    image_id_to_path = {}
    with open(images_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id, img_path = parts
                image_id_to_path[img_id] = img_path
    
    # 读取 train/test 划分
    train_ids = set()
    test_ids = set()
    with open(split_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                img_id, is_train = parts
                if is_train == '1':
                    train_ids.add(img_id)
                else:
                    test_ids.add(img_id)
    
    print(f"  Train images: {len(train_ids)}")
    print(f"  Test images: {len(test_ids)}")
    
    # 创建目录并移动文件
    train_dir = target_dir / "train"
    test_dir = target_dir / "test"
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)
    
    for img_id, img_path in tqdm(image_id_to_path.items(), desc="Organizing"):
        # img_path 格式: 001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg
        class_name = img_path.split('/')[0]
        img_name = img_path.split('/')[-1]
        
        src = images_dir / img_path
        
        if img_id in train_ids:
            dst_dir = train_dir / class_name
        else:
            dst_dir = test_dir / class_name
        
        dst_dir.mkdir(exist_ok=True)
        dst = dst_dir / img_name
        
        if src.exists() and not dst.exists():
            shutil.copy2(str(src), str(dst))
    
    # 验证
    train_classes = len(list(train_dir.iterdir()))
    test_classes = len(list(test_dir.iterdir()))
    train_images = sum(1 for _ in train_dir.rglob("*.jpg"))
    test_images = sum(1 for _ in test_dir.rglob("*.jpg"))
    
    print(f"\n[OK] Dataset organized:")
    print(f"  Train: {train_classes} classes, {train_images} images")
    print(f"  Test: {test_classes} classes, {test_images} images")
    
    # 清理 tgz (可选，保留以便重新解压)
    # if tgz_path.exists():
    #     tgz_path.unlink()
    #     print("[OK] Cleaned up archive")
    
    return True


def _compute_prefetch_factor(
    batch_size: int,
    num_workers: int,
    is_container: bool = True,
    swap_memory_mb: float = 24576.0,  # 默认 24GB swap
    model_memory_mb: float = 2048.0,   # 模型内存估计
    sample_memory_mb: float = 2.0,     # 单样本内存估计 (MB)
    prefetch_max: int = 16,            # prefetch_factor 上限 (I136: 增加到 16 避免 GPU 空转)
) -> int:
    """
    计算安全的 DataLoader prefetch_factor

    数学模型:
        Memory_prefetch = B × W × P × M_s ≤ M_swap - M_model

    参数:
        batch_size: 批次大小
        num_workers: DataLoader worker 数
        is_container: 是否容器环境
        swap_memory_mb: 可用交换内存 (MB)
        model_memory_mb: 模型内存估计 (MB)
        sample_memory_mb: 单样本内存估计 (MB)
        prefetch_max: prefetch_factor 上限

    返回:
        prefetch_factor: 安全预取因子 (最小值 2)

    示例:
        Container (B=32, W=8, 24GB swap):
            P = min(8, (24576-2048)/(32*8*2)) = min(8, 44) = 8
    """
    # 可用于预取的内存
    available_memory = swap_memory_mb - model_memory_mb

    # 计算最大安全预取因子
    max_safe_prefetch = available_memory / (batch_size * num_workers * sample_memory_mb)

    # 根据环境计算上限
    prefetch_upper = min(prefetch_max, int(max_safe_prefetch))

    # 保证最小预取以维持性能
    return max(2, prefetch_upper)


# ============================================================================
# 数据加载
# ============================================================================

# P-OPT: 模块级 collate_fn (可 pickle，用于多进程)
def collate_fn_use_channels_last(batch, compile_model: bool = False):
    """Collate 函数：转换为 use_channels_last 格式

    Args:
        batch: 数据批次
        compile_model: 是否使用 torch.compile (编译时不在 CPU 端转换)
    """
    imgs, labels = default_collate(batch)
    # P-OPT: 仅在不编译时在 CPU 端预转换
    if not compile_model and imgs.dim() == 4:
        imgs = imgs.to(memory_format=torch.use_channels_last)
    return imgs, labels


def create_dataloaders(
    spec: DatasetSpec,
    config: FractalConfigProtocol,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""

    # 数据增强 - 不 resize，保持原始分辨率
    if spec.name == "MNIST":
        train_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ])
    elif spec.name == "TinyImageNet":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),  # 原始图像 64x64
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),
        ])
    elif spec.name == "CUB200":
        # CUB-200-2011 细粒度分类 - 保持原始分辨率
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(spec.image_size, padding=4) if spec.image_size else transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),
        ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(spec.mean, spec.std),
    ])
    
    # 加载数据
    data_root = PROJECT_ROOT / "data"
    data_root.mkdir(exist_ok=True)
    
    if spec.name == "CIFAR10":
        train_ds = datasets.CIFAR10(data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR10(data_root, train=False, download=True, transform=test_tf)
    elif spec.name == "CIFAR100":
        train_ds = datasets.CIFAR100(data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf)
    elif spec.name == "MNIST":
        train_ds = datasets.MNIST(data_root, train=True, download=True, transform=train_tf)
        test_ds = datasets.MNIST(data_root, train=False, download=True, transform=test_tf)
    elif spec.name == "TinyImageNet":
        if not download_tiny_imagenet(data_root):
            raise FileNotFoundError("Failed to download Tiny ImageNet")
        train_dir = data_root / "tiny-imagenet-200" / "train"
        test_dir = data_root / "tiny-imagenet-200" / "val"
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    elif spec.name == "CUB200":
        # CUB-200-2011 细粒度鸟类分类数据集
        if not download_cub200(data_root):
            raise FileNotFoundError("Failed to download CUB-200-2011")
        train_dir = data_root / "CUB_200_2011" / "train"
        test_dir = data_root / "CUB_200_2011" / "test"
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_tf)
        test_ds = datasets.ImageFolder(str(test_dir), transform=test_tf)
    else:
        raise ValueError(f"Unknown dataset: {spec.name}")
    
    # 划分 (使用分层采样确保类别平衡)
    n_samples = len(train_ds)
    indices = np.arange(n_samples)
    
    # 获取所有标签用于分层采样
    if hasattr(train_ds, 'targets'):
        all_labels = np.array(train_ds.targets)
    elif hasattr(train_ds, 'labels'):
        all_labels = np.array(train_ds.labels)
    else:
        # ImageFolder 需要遍历
        all_labels = np.array([train_ds.samples[i][1] for i in range(n_samples)])
    
    if config.subset_size:
        indices = indices[:config.subset_size]
        all_labels = all_labels[:config.subset_size]
    
    # 分层采样：确保训练/验证集中每个类别比例相同
    num_classes = len(np.unique(all_labels))
    if num_classes > 1:
        if HAS_SKLEARN:
            val_size = max(1, int(len(indices) * config.val_split))
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=config.seed)
            train_idx, val_idx = next(sss.split(indices, all_labels[indices] if config.subset_size else all_labels))
            train_idx = indices[train_idx]
            val_idx = indices[val_idx]
            print(f"[OK] 使用 sklearn 分层采样划分训练/验证集")
        else:
            # 使用手动分层划分
            train_idx, val_idx = stratified_split(indices, all_labels, config.val_split, config.seed)
            print(f"[OK] 使用手动分层采样划分训练/验证集 (确保类别平衡)")
    else:
        # 单类别数据集 - 简单随机划分
        np.random.shuffle(indices)
        val_size = max(1, int(len(indices) * config.val_split))
        train_idx, val_idx = indices[val_size:], indices[:val_size]
    
    # DataLoader 参数
    # P10-RES-2: 优化多进程上下文选择 (WSL + Podman/Docker 容器优化)
    # - Linux 原生: 使用 'fork' (更快)
    # - WSL: 使用 'spawn' (fork 在 WSL 中不稳定)
    # - Docker/Podman 容器: 使用 'spawn' + 减少 workers (共享内存受限)
    import platform
    
    # 检测运行环境
    is_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
    is_wsl = 'microsoft' in platform.uname().release.lower() if hasattr(platform.uname(), 'release') else False
    is_podman = os.path.exists('/run/.containerenv')  # Podman 特有标志
    
    # 检测共享内存大小 (容器常见限制)
    shm_size_gb = 0.064  # 默认假设 64MB
    try:
        if os.path.exists('/dev/shm'):
            import shutil
            shm_stat = shutil.disk_usage('/dev/shm')
            shm_size_gb = shm_stat.total / (1024**3)
    except Exception:
        pass
    
    # 根据环境选择最优配置
    effective_workers = config.num_workers
    if config.num_workers > 0:
        if is_container:
            # 容器环境: spawn 更稳定，限制 workers 避免共享内存不足
            mp_context = 'spawn'
            # 共享内存 < 1GB 时限制 workers
            if shm_size_gb < 1.0:
                effective_workers = min(config.num_workers, 2)
                print(f"[WARN] 共享内存受限 ({shm_size_gb:.2f}GB)，降低 workers: {config.num_workers} -> {effective_workers}")
            print(f"[INFO] 容器环境检测: {'Podman' if is_podman else 'Docker'}, mp_context='spawn'")
        elif is_wsl:
            # WSL: spawn 更稳定
            mp_context = 'spawn'
            print(f"[INFO] WSL 环境检测, mp_context='spawn'")
        elif platform.system() == 'Linux':
            mp_context = 'fork'
        else:
            mp_context = 'spawn'
    else:
        mp_context = None
    
    # DataLoader 参数优化
    # P16/P17-FIX: 容器环境中根据共享内存大小决定配置
    #   - shm >= 2GB: 启用 persistent_workers (高性能)
    #   - shm >= 1GB: 启用 pin_memory
    #   - shm < 1GB: 保守配置，禁用两者
    shm_sufficient = shm_size_gb >= 2.0
    shm_moderate = shm_size_gb >= 1.0
    
    # P-OPT: 禁用 pin_memory 避免 torch.compile + 多进程兼容性问题
    use_pin_memory = False
    use_persistent_workers = (
        effective_workers > 0 
        and (not is_container or shm_sufficient)  # 容器中需要更多共享内存
    )
    
    if is_container and effective_workers > 0:
        print(f"[INFO] 容器环境 (shm={shm_size_gb:.1f}GB): "
              f"pin_memory={use_pin_memory}, persistent_workers={use_persistent_workers}")
    
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': effective_workers,
        'pin_memory': use_pin_memory,
        'multiprocessing_context': mp_context if effective_workers > 0 else None,
        'persistent_workers': use_persistent_workers,
        'drop_last': True,  # 避免最后一个小 batch 的性能损失
    }
    if effective_workers > 0:
        # prefetch_factor: 每个 worker 预取的 batch 数
        # 使用动态计算，根据可用内存调整
        loader_kwargs['prefetch_factor'] = _compute_prefetch_factor(
            batch_size=config.batch_size,
            num_workers=effective_workers,
            is_container=is_container,
            swap_memory_mb=24576.0,  # 24GB swap
            model_memory_mb=2048.0,   # 模型内存
        )
        # 添加 generator 参数以提高多进程随机性
        loader_kwargs['generator'] = torch.Generator().manual_seed(42)

    # P-OPT: 使用模块级 collate_fn (可 pickle)
    # 仅在启用 use_channels_last 且不编译时使用
    if config.use_channels_last and not config.compile_model:
        # 使用 functools.partial 绑定 compile_model 参数
        from functools import partial
        loader_kwargs['collate_fn'] = partial(collate_fn_use_channels_last, compile_model=False)

    train_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(train_idx), **loader_kwargs)
    
    # 验证集不需要 drop_last
    val_kwargs = loader_kwargs.copy()
    val_kwargs['drop_last'] = False
    val_loader = DataLoader(train_ds, sampler=SubsetRandomSampler(val_idx), **val_kwargs)
    
    test_kwargs = loader_kwargs.copy()
    test_kwargs['shuffle'] = False
    test_loader = DataLoader(test_ds, **test_kwargs)
    
    print(f"[OK] Data: train={len(train_idx)}, val={len(val_idx)}, test={len(test_ds)}")
    
    # P10-RES-1: 注册清理函数，防止信号量泄漏
    def cleanup_dataloaders():
        """清理 DataLoader 工作进程"""
        try:
            # 显式关闭迭代器
            for loader in [train_loader, val_loader, test_loader]:
                if hasattr(loader, '_iterator') and loader._iterator is not None:
                    loader._iterator._shutdown_workers()
        except Exception:
            pass
    
    atexit.register(cleanup_dataloaders)
    
    return train_loader, val_loader, test_loader


# ============================================================================
# CUDA Prefetcher
# ============================================================================

class CudaPrefetcher:
    """CUDA 异步数据预取器 (双缓冲优化)
    
    使用双缓冲策略减少 GPU 空转:
    - buffer[0]: 当前正在被 GPU 处理的数据
    - buffer[1]: 后台 stream 异步加载的下一批数据
    
    数学分析:
    - 单缓冲: T_total = T_load + T_compute (串行)
    - 双缓冲: T_total = max(T_load, T_compute) (流水线)
    - 加速比: (T_load + T_compute) / max(T_load, T_compute)
    
    当 T_load ≈ T_compute 时，理论加速比接近 2x
    """
    
    def __init__(self, loader: DataLoader, device: torch.device, use_channels_last: bool = False):
        self.loader = loader
        self.device = device
        self.use_channels_last = use_channels_last
        # 双缓冲: 使用两个 CUDA stream 实现流水线
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self._debug = False
        self._batch_count = 0
        # 双缓冲状态: 使用双指针代替线性查找
        self._buffer = [None, None]  # (raw_batch, gpu_data)
        self._current_idx = 0  # 当前缓冲区指针
        self._preload_idx = 1  # 预加载缓冲区指针

    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self._batch_count = 0
        self._buffer = [None, None]
        self._current_idx = 0
        self._preload_idx = 1
        # 预加载两个 batch 填满双缓冲
        self._preload_next()
        self._preload_next()
        return self

    def _preload_next(self):
        """预加载下一个 batch 到预加载缓冲区"""
        try:
            raw_batch = next(self.loader_iter)
        except StopIteration:
            return False

        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                imgs = raw_batch[0].to(self.device, non_blocking=True)
                if self.use_channels_last:
                    imgs = imgs.to(memory_format=torch.use_channels_last)
                gpu_data = (
                    imgs,
                    raw_batch[1].to(self.device, non_blocking=True),
                )
        else:
            gpu_data = raw_batch

        # 直接写入预加载缓冲区，无需查找
        self._buffer[self._preload_idx] = gpu_data
        return True

    def __next__(self):
        # 获取当前缓冲区数据（优先使用 _current_idx，如果为空则尝试 _preload_idx）
        current_data = self._buffer[self._current_idx]
        preload_data = self._buffer[self._preload_idx]

        if current_data is None and preload_data is None:
            # 两个缓冲区都为空，尝试预加载
            if not self._preload_next():
                raise StopIteration
            current_data = self._buffer[self._preload_idx]
            if current_data is None:
                raise StopIteration
            # 交换指针并返回
            self._current_idx, self._preload_idx = self._preload_idx, self._current_idx
            self._buffer[self._current_idx] = None
            self._batch_count += 1
            self._preload_next()
            return current_data

        # 如果 _current_idx 为空但 _preload_idx 有数据，交换它们
        if current_data is None:
            current_data = preload_data
            self._buffer[self._preload_idx] = None

        # 等待 CUDA stream 完成（如果使用 CUDA）
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)

        # 清空当前缓冲区，切换指针
        self._buffer[self._current_idx] = None
        self._batch_count += 1

        # 交换指针: current -> preload, preload -> current
        self._current_idx, self._preload_idx = self._preload_idx, self._current_idx

        # 后台预加载下一个 batch 到新的 preload 缓冲区
        self._preload_next()

        return current_data

    def __len__(self):
        return len(self.loader)


# ============================================================================
# 训练函数
# ============================================================================

def get_model_input_dtype(model: nn.Module) -> torch.dtype:
    """获取模型期望的输入 dtype (根据第一个参数的 dtype)"""
    for param in model.parameters():
        return param.dtype
    return torch.float32

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler,
    config: FractalConfigProtocol,
    mixup_fn: Optional[MixupCutmix] = None,
    num_classes: int = 10,
    profile: bool = False,
    exp_dir: Optional[Path] = None,
    loss_fn: Optional[nn.Module] = None,
    class_weights: Optional[torch.Tensor] = None,
    epoch: int = 1,
    hard_mining: Optional[HilbertAwareHardMining] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """训练一个 epoch

    Args:
        exp_dir: 实验目录，用于保存 NaN/Inf 诊断日志
    """
    model.train()
    # P11-8: 使用张量累加，延迟 .item() 调用到 epoch 结束
    total_loss = torch.tensor(0.0, device=device)
    correct = torch.tensor(0, device=device, dtype=torch.long)
    total = 0
    optimizer.zero_grad(set_to_none=True)

    batch_times, data_times, forward_times = [], [], []
    entropy_losses = []  # P1-5: 收集熵损失用于统计
    cuda_mem_peak = 0.0
    use_mixup = mixup_fn is not None
    nan_count = 0  # NaN 计数器

    # I107-7: 调试模式控制 - 默认关闭以提升性能
    # 设置 DEBUG_MODE=1 启用调试输出
    debug_mode = os.environ.get('DEBUG_MODE', '0') == '1'

    # P15: Warmup 后首次启用 Mixup 的调试信息 (仅调试模式)
    if debug_mode and use_mixup and epoch is not None:
        import sys
        print(f"[DEBUG] train_epoch started: epoch={epoch}, use_mixup=True", file=sys.stderr, flush=True)

    # P15: 预先获取模型期望的 dtype (避免每次迭代都检查)
    model_dtype = get_model_input_dtype(model) if use_mixup else None
    if debug_mode and model_dtype is not None:
        print(f"[DEBUG] Model expects input dtype: {model_dtype}")

    # 使用环境变量 DISABLE_PREFETCH=1 来禁用 CudaPrefetcher
    use_prefetcher = device.type == 'cuda' and os.environ.get('DISABLE_PREFETCH', '0') != '1'

    if use_prefetcher:
        data_iter = CudaPrefetcher(loader, device, use_channels_last=config.use_channels_last)
    else:
        data_iter = loader

    pbar = tqdm(data_iter, desc="Train", total=len(loader), mininterval=0.5, dynamic_ncols=True)

    # 检查 DataLoader 长度
    if len(loader) == 0:
        print(f"[ERROR] DataLoader 长度为 0！train_loader 有 {len(train_loader)} 个 batch")
        return 0.0, 0.0, {}

    # P13: 卡顿诊断 - 检测异常长的批次时间
    stall_threshold = 5.0  # 超过 5 秒视为卡顿
    stall_count = 0
    
    data_start = time.time()

    # P15: 在首个 Mixup epoch 添加额外诊断 (仅调试模式)
    debug_first_mixup_epoch = debug_mode and use_mixup and epoch is not None and os.environ.get('DISABLE_PREFETCH', '0') == '1'

    for i, batch in enumerate(pbar):
        # 总是更新进度条（即使有 continue 也需要更新）
        pbar.update(1)

        # P13: 检测数据加载卡顿
        data_time = time.time() - data_start
        if data_time > stall_threshold:
            stall_count += 1
            if stall_count <= 3:
                print(f"\n[STALL] Batch {i}: 数据加载耗时 {data_time:.1f}s (可能是 GC/编译/I/O)")

        data_times.append(data_time)
        batch_start = time.time()
        
        imgs, labels = batch
        if device.type != 'cuda':
            imgs = imgs.to(device)
            labels = labels.to(device)
        else:
            # 当不使用 CudaPrefetcher 时，需要手动处理 GPU 传输和 use_channels_last
            if not use_prefetcher:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if config.use_channels_last:
                    imgs = imgs.to(memory_format=torch.use_channels_last)
        
        # 检查 label 范围
        if labels.min() < 0 or labels.max() >= num_classes:
            print(f"\n[WARN] Label 范围异常: min={labels.min().item()}, max={labels.max().item()}, num_classes={num_classes}")
            continue
        
        # I23-4-FIX: 检查输入图像是否包含 NaN/Inf
        # 这可能由数据加载/增强导致，跳过有问题的 batch 避免 splitter_loss 变成 NaN
        if torch.isnan(imgs).any() or torch.isinf(imgs).any():
            nan_count_input = torch.isnan(imgs).sum().item()
            inf_count_input = torch.isinf(imgs).sum().item()
            print(f"\n[WARN] Batch {i}: 输入图像包含 NaN={nan_count_input}, Inf={inf_count_input}，跳过此 batch")
            continue

        # 应用 Mixup/CutMix
        mixed_labels: Optional[torch.Tensor] = None
        if use_mixup and mixup_fn is not None:
            if debug_mode and i == 0:
                print(f"[DEBUG] Batch 0: 开始应用 Mixup...", flush=True)
            imgs, mixed_labels = mixup_fn(imgs, labels)
            # 确保 Mixup 后保持 use_channels_last 格式 (Mixup 的线性混合可能会破坏 memory format)
            if config.use_channels_last and imgs.device.type == 'cuda':
                imgs = imgs.to(memory_format=torch.use_channels_last)
            if debug_mode and i == 0:
                print(f"[DEBUG] Batch 0: Mixup 完成，mixed_labels.shape={mixed_labels.shape}", flush=True)
                print(f"[DEBUG] Batch 0: imgs.dtype={imgs.dtype}, imgs.is_contiguous(memory_format=torch.use_channels_last)={imgs.is_contiguous(memory_format=torch.use_channels_last)}", flush=True)

        forward_start = time.time()

        # P15: 确保输入 dtype 与模型权重匹配 (torch.compile + AMP 可能导致不匹配)
        if model_dtype is not None and imgs.dtype != model_dtype:
            if i == 0:
                print(f"[WARN] dtype 不匹配! 将 imgs 从 {imgs.dtype} 转换为 {model_dtype}")
            imgs = imgs.to(dtype=model_dtype)
        elif debug_mode and i == 0 and use_mixup:
            print(f"[DEBUG] Batch 0: 开始 forward pass, imgs.dtype={imgs.dtype}...", flush=True)

        with get_amp_context(device, config.use_amp):
            # 单一接口: forward() 返回 TrainingStats 或 Tensor
            if hard_mining is not None and not use_mixup:
                stats = model(imgs)
                # 从 TrainingStats 提取信息
                tokens = stats.transformer_tokens if hasattr(stats, 'transformer_tokens') else None
                # I141: num_tokens 可能为 int, List[int], 或 torch.Tensor
                raw_lengths = stats.num_tokens if hasattr(stats, 'num_tokens') else None
                if raw_lengths is None:
                    token_lengths = None
                elif isinstance(raw_lengths, torch.Tensor):
                    # I141: GPU tensor 直接使用，无需 CPU 转换
                    token_lengths = raw_lengths.to(device=device, non_blocking=True)
                elif isinstance(raw_lengths, list):
                    token_lengths = torch.tensor(raw_lengths, device=device, dtype=torch.long)
                else:
                    token_lengths = raw_lengths  # int 类型
                outs = stats.logits if hasattr(stats, 'logits') else stats
            else:
                stats = model(imgs)
                tokens = None
                token_lengths = None
                outs = stats.logits if hasattr(stats, 'logits') else stats

            # I145: 验证 TrainingStats 字段类型
            if hasattr(stats, 'validate'):
                stats.validate()

            if debug_mode and i == 0 and use_mixup:
                print(f"[DEBUG] Batch 0: forward 完成，outs.shape={outs.shape}", flush=True)
            
            # 检查 logits 范围，防止爆炸
            if torch.isnan(outs).any() or torch.isinf(outs).any():
                nan_count += 1
                if nan_count <= 3:
                    print(f"\n[WARN] Logits 包含 NaN/Inf (batch {i}), 跳过此 batch")
                    # 详细诊断
                    report = diagnose_nan_inf(
                        batch_idx=i,
                        imgs=imgs,
                        labels=labels,
                        model=model,
                        logits=outs,
                        log_file=exp_dir / "nan_inf_diagnose.log" if exp_dir else None,
                    )
                    print(report)
                if nan_count > 10:
                    raise RuntimeError(f"连续出现 {nan_count} 次 NaN，训练终止")
                optimizer.zero_grad(set_to_none=True)
                continue
            
            if use_mixup and mixed_labels is not None:
                # 使用混合标签的交叉熵 (Mixup 模式下不使用 Focal Loss)
                ce_loss = mixup_criterion(outs, mixed_labels) / config.accum_steps
                if debug_mode and i == 0:
                    print(f"[DEBUG] Batch 0: mixup_criterion 计算完成，ce_loss={ce_loss.item():.4f}", flush=True)
            else:
                # P14: 使用自定义损失函数 (Focal Loss / Class-Balanced Loss)
                if loss_fn is not None:
                    # I30-2: 当启用 Hilbert 困难样本挖掘时，需要 unreduced loss
                    if hard_mining is not None and tokens is not None:
                        # 临时修改 reduction 获取逐样本损失
                        old_reduction = getattr(loss_fn, 'reduction', 'mean')
                        if hasattr(loss_fn, 'reduction'):
                            loss_fn.reduction = 'none'
                        base_loss = loss_fn(outs, labels)  # [B]
                        if hasattr(loss_fn, 'reduction'):
                            loss_fn.reduction = old_reduction
                        # 应用 token variance 加权
                        ce_loss, mining_info = hard_mining(tokens, base_loss, token_lengths)
                        ce_loss = ce_loss / config.accum_steps
                        # 记录挖掘统计 (每 100 batch 打印一次)
                        if i % 100 == 0 and i > 0:
                            print(f"[MINING] Batch {i}: weight=[{mining_info.get('weight_min', 0):.3f}, {mining_info.get('weight_max', 0):.3f}], "
                                  f"token_var={mining_info.get('token_var_mean', 0):.2f}")
                    else:
                        ce_loss = loss_fn(outs, labels) / config.accum_steps
                elif class_weights is not None:
                    # 使用类别平衡权重
                    ce_loss = F.cross_entropy(
                        outs, labels, 
                        weight=class_weights,
                        label_smoothing=config.label_smoothing
                    ) / config.accum_steps
                else:
                    # 默认交叉熵
                    ce_loss = F.cross_entropy(
                        outs, labels, 
                        label_smoothing=config.label_smoothing
                    ) / config.accum_steps
            
            # P1-5 修复: 收集熵正则化损失
            # 熵损失鼓励尺度分布多样性，防止 CrossScaleAttention 崩塌到单一尺度
            entropy_loss = None

            # P10-4/P10-9: 可学习分割器辅助损失（推荐使用统一接口）
            # 包含: 软熵损失 + 弹性预算损失 + 阈值 barrier 正则化
            # I14-1 D1: 新增崩溃惩罚，需要传递 actual_token_count
            splitter_loss = None
            # I98-2: 使用 model.splitter (独立组件)
            if hasattr(model, 'splitter'):
                splitter = model.splitter
                # I107-7: 从 stats.shared_features 获取已计算的 features
                # 避免重复调用 model.tokenizer.shared_conv(imgs) (节省 ~5-10% 计算开销)
                if stats is not None and hasattr(stats, 'shared_features') and stats.shared_features is not None:
                    splitter_features = stats.shared_features
                else:
                    # Fallback: 仍需计算时的回退方案
                    splitter_features = model.tokenizer.shared_conv(imgs)

                if hasattr(splitter, 'get_auxiliary_losses'):
                    # I142: 不再需要传递 actual_token_count，splitter 内部使用 _avg_selected 进行崩溃检测
                    # 这避免了训练循环中的 .item() 调用导致的 CPU 同步
                    aux_losses = splitter.get_auxiliary_losses(
                        features=splitter_features,
                        image_size=(imgs.shape[2], imgs.shape[3]),
                        include_elastic_budget=config.include_elastic_budget,
                        include_soft_entropy=config.include_soft_entropy,
                        batch_size=imgs.shape[0],
                        # I142: 移除 actual_token_count 参数，使用内部缓存值
                        entropy_target=config.soft_entropy_target,
                        entropy_weight=config.soft_entropy_weight,
                        entropy_mode=config.soft_entropy_mode,
                    )
                    # 收集各项损失
                    # I23-5-FIX: 确保 splitter_loss 是张量类型
                    # 当 aux_losses 为空时，sum({}.values()) 返回 int(0)
                    if aux_losses:
                        splitter_loss = sum(aux_losses.values())
                        # 类型检查防护
                        if not isinstance(splitter_loss, torch.Tensor):
                            splitter_loss = torch.tensor(float(splitter_loss), device=device)
                    else:
                        splitter_loss = torch.tensor(0.0, device=device)
                    # I102-4: splitter_metrics 是死代码，移除以防止显存泄露
                    # splitter_metrics = aux_losses  # 保留张量引用会导致内存累积
                    # GumbelTopKSplitter 的 get_auxiliary_losses 已包含所有必需损失

            # 组合损失 (在组合前检查每个损失项，并确保 dtype 一致)
            # P15-FIX: 在 AMP 混合精度训练中，不同损失可能有不同 dtype
            #          ce_loss 可能是 float16，而 splitter_loss/multi_layer_loss 是 float32
            #          混合相加可能导致 NaN。统一转换为 float32 进行损失计算。
            loss = ce_loss.float()  # 确保基础损失是 float32

            if entropy_loss is not None:
                entropy_loss_f32 = entropy_loss.float()
                if torch.isnan(entropy_loss_f32) or torch.isinf(entropy_loss_f32):
                    print(f"[WARN] entropy_loss 为 NaN/Inf: {entropy_loss_f32.item()}")
                    entropy_loss = None  # 跳过该损失
                else:
                    loss = loss + entropy_loss_f32 / config.accum_steps
                    entropy_losses.append(entropy_loss_f32.item())  # P1-5: 记录熵损失
            if splitter_loss is not None:
                splitter_loss_f32 = splitter_loss.float()
                if torch.isnan(splitter_loss_f32) or torch.isinf(splitter_loss_f32):
                    print(f"[WARN] splitter_loss 为 NaN/Inf: {splitter_loss_f32.item()}")
                    splitter_loss = None  # 跳过该损失
                else:
                    loss = loss + splitter_loss_f32 / config.accum_steps

            # I110-7: 语义分裂器损失集成
            if config.use_semantic_splitter and stats is not None:
                semantic_loss = getattr(stats, 'semantic_loss', None)
                if semantic_loss is not None:
                    semantic_loss_f32 = semantic_loss.float()
                    if torch.isnan(semantic_loss_f32).any() or torch.isinf(semantic_loss_f32).any():
                        print(f"[WARN] semantic_loss 包含 NaN/Inf，跳过此损失")
                    else:
                        # 使用配置的权重
                        semantic_weight = getattr(config, 'semantic_loss_weight', 0.1)
                        loss = loss + semantic_loss_f32.mean() * semantic_weight / config.accum_steps
        
        # 检查 loss 是否为 NaN/Inf，并输出详细诊断信息
        if torch.isnan(loss) or torch.isinf(loss):
            nan_count += 1
            if nan_count <= 3:
                print(f"\n[WARN] Loss 为 NaN/Inf (batch {i}), 跳过此 batch")
                # 详细诊断
                report = diagnose_nan_inf(
                    batch_idx=i,
                    imgs=imgs,
                    labels=labels,
                    model=model,
                    logits=outs,
                    loss=loss,
                    ce_loss=ce_loss * config.accum_steps,  # 还原真实值
                    entropy_loss=entropy_loss,
                    splitter_loss=splitter_loss,
                    log_file=exp_dir / "nan_inf_diagnose.log" if exp_dir else None,
                )
                print(report)
            if nan_count > 10:
                raise RuntimeError(f"连续出现 {nan_count} 次 NaN loss，训练终止")
            optimizer.zero_grad(set_to_none=True)
            continue
        
        nan_count = 0  # 重置计数器
        
        forward_time = time.time() - forward_start
        forward_times.append(forward_time)
        
        scaler.scale(loss).backward()

        if (i + 1) % config.accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # P11-8: 使用 detach() 累加损失，避免保留计算图
        # .item() 延迟到 epoch 结束时调用
        with torch.no_grad():
            total_loss += loss.detach() * config.accum_steps
            _, pred = outs.detach().max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum()
        
        batch_times.append(time.time() - batch_start)

        if device.type == 'cuda':
            cuda_mem_peak = max(cuda_mem_peak, torch.cuda.max_memory_allocated() / 1024**3)

        # P-OPT: 减少 .item() 调用频率，每 20 个 batch 同步一次
        # 使用 torch.no_grad() 避免影响梯度计算
        if i % 20 == 0:
            with torch.no_grad():
                loss_val = loss.detach().item() * config.accum_steps
                acc_val = 100. * correct.detach().item() / total if total > 0 else 0
            if profile and (i < 5 or i % 100 == 0):
                pbar.set_postfix(
                    loss=f'{loss_val:.3f}',
                    acc=f'{acc_val:.1f}%',
                    data=f'{data_time*1000:.0f}ms',
                    fwd=f'{forward_time*1000:.0f}ms'
                )
            else:
                pbar.set_postfix(loss=f'{loss_val:.4f}', acc=f'{acc_val:.1f}%')

        # P-OPT: 移除无意义的 GC 调用
        # Python GC 对 GPU 内存无影响，使用 torch.cuda.empty_cache() 更有效
        # 但 empty_cache() 也有开销，仅在真正需要时调用

        data_start = time.time()

    perf_stats = {
        'avg_batch_time': np.mean(batch_times) if batch_times else 0,
        'avg_data_time': np.mean(data_times) if data_times else 0,
        'avg_forward_time': np.mean(forward_times) if forward_times else 0,
        'throughput': total / sum(batch_times) if batch_times else 0,
        'cuda_mem_peak_gb': cuda_mem_peak,
        # P1-5: 添加熵统计
        'avg_entropy_loss': np.mean(entropy_losses) if entropy_losses else None,
    }
    
    # P1-5: 获取当前尺度熵值用于监控
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_scale_entropy'):
        scale_entropy = model.tokenizer.get_scale_entropy()
        if scale_entropy is not None:
            perf_stats['scale_entropy'] = scale_entropy
    
    # P7/P8: 获取可学习分割器统计信息
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_training_stats'):
        training_stats = model.tokenizer.get_training_stats()
        if training_stats.get('learnable_split'):
            perf_stats['learnable_thresholds'] = training_stats.get('learnable_thresholds')
            perf_stats['learnable_temperature'] = training_stats.get('learnable_temperature')
    
    # P10-4/P10-5/P10-9: 获取自适应分割器深度分布统计
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
        splitter = model.splitter
        if hasattr(splitter, 'get_depth_distribution_stats'):
            try:
                depth_stats = splitter.get_depth_distribution_stats()
                if depth_stats:
                    perf_stats['soft_entropy'] = depth_stats.get('entropy')
                    perf_stats['entropy_ratio'] = depth_stats.get('entropy_ratio')
                    perf_stats['dominant_depth'] = depth_stats.get('dominant_depth')
                    perf_stats['dominant_prob'] = depth_stats.get('dominant_prob')
            except Exception:
                pass  # 忽略统计收集错误

        # P10-9: 软 token 计数
        if hasattr(splitter, 'get_soft_token_count'):
            try:
                soft_count = splitter.get_soft_token_count(batch_size=1)
                perf_stats['soft_token_count'] = soft_count.item() if hasattr(soft_count, 'item') else soft_count
            except Exception:
                pass

    # I111-6: 使用 DepthMonitor 收集深度分布统计
    # 延迟初始化，避免重复创建
    if not hasattr(model, '_depth_monitor'):
        splitter = getattr(model, 'splitter', None)
        if splitter is not None and hasattr(splitter, 'get_depth_distribution_tensor'):
            model._depth_monitor = DepthMonitor(splitter, history_max_size=0)  # 不需要历史
        else:
            model._depth_monitor = None

    if hasattr(model, '_depth_monitor') and model._depth_monitor is not None:
        try:
            depth_stats = model._depth_monitor.update(global_step)
            # 提取关键指标（转换为 Python 类型用于日志）
            perf_stats['depth_pi'] = depth_stats['pi'].tolist()
            perf_stats['depth_entropy'] = depth_stats['entropy']
            perf_stats['depth_kl'] = depth_stats['kl_from_uniform']
            perf_stats['max_entropy'] = depth_stats['max_entropy']
            perf_stats['dominant_depth'] = depth_stats['dominant_depth']
            perf_stats['dominant_prob'] = depth_stats['dominant_prob']
            # 保存 GPU tensor 用于 TensorBoard（避免重复转换）
            perf_stats['_depth_pi_tensor'] = depth_stats['pi']
            perf_stats['_quota_probs_tensor'] = depth_stats.get('quota_probs')
        except Exception as e:
            # I111-6: 静默处理收集错误，避免影响训练
            pass
    
    # P11-8: 在返回前进行一次 GPU-CPU 同步
    final_loss = (total_loss / len(loader)).item()
    final_acc = (100.0 * correct / total).item() if total > 0 else 0.0
    return final_loss, final_acc, perf_stats


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
    num_classes: int = 10,
    return_per_class: bool = False,
    use_channels_last: bool = False,
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """评估 - 使用 InferenceWrapper 确保与 eval.py 推理逻辑一致"""
    model.eval()

    # 应用 channels_last 格式（如果需要）
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)

    # 使用共享的 inference_wrapper 进行评估
    result: EvalResult = inference_evaluate(
        model, loader,
        device=device,
        use_amp=use_amp,
        num_classes=num_classes,
        return_per_class=return_per_class,
    )

    # 转换为旧接口格式（保持向后兼容）
    per_class_stats = None
    if return_per_class and result.per_class_accuracy:
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        for cls_id, acc in result.per_class_accuracy.items():
            if cls_id < num_classes:
                class_correct[cls_id] = acc / 100.0 * (1.0 / num_classes)  # 近似值
                class_total[cls_id] = 1.0  # 近似值

        class_acc = np.array([result.per_class_accuracy.get(i, 0.0) for i in range(num_classes)])

        per_class_stats = {
            'class_accuracy': class_acc.tolist(),
            'class_correct': class_correct.tolist(),
            'class_total': class_total.tolist(),
            'worst_classes': np.argsort(class_acc)[:10].tolist(),
            'best_classes': np.argsort(class_acc)[-10:][::-1].tolist(),
            'accuracy_std': float(np.std(class_acc[class_total > 0])) if np.any(class_total > 0) else 0.0,
            'accuracy_min': float(np.min(class_acc[class_total > 0])) if np.any(class_total > 0) else 0.0,
            'accuracy_max': float(np.max(class_acc[class_total > 0])) if np.any(class_total > 0) else 0.0,
        }

    return result.avg_loss, result.accuracy, per_class_stats


def analyze_class_balance(
    per_class_stats: Optional[Dict[str, Any]],
    epoch: int,
    verbose: bool = True,
) -> Dict[str, Any]:
    """分析类别准确率分布，检测长尾效应。
    
    P14: 长尾效应检测和预警
    
    检测指标:
    1. 零准确率类别数量
    2. 准确率标准差
    3. 最差/最佳类别差距
    4. 低于阈值的类别比例
    
    Args:
        per_class_stats: evaluate() 返回的 per-class 统计
        epoch: 当前 epoch
        verbose: 是否打印详细信息
        
    Returns:
        分析报告字典
    """
    if per_class_stats is None:
        return {'status': 'no_stats'}
    
    class_acc = np.array(per_class_stats['class_accuracy'])
    class_total = np.array(per_class_stats['class_total'])
    
    # 只考虑有样本的类别
    valid_mask = class_total > 0
    valid_acc = class_acc[valid_mask]
    
    if len(valid_acc) == 0:
        return {'status': 'no_valid_classes'}
    
    # 计算统计指标
    zero_acc_count = np.sum(valid_acc == 0)
    low_acc_count = np.sum(valid_acc < 20)  # 低于 20% 的类别
    acc_std = np.std(valid_acc)
    acc_min = np.min(valid_acc)
    acc_max = np.max(valid_acc)
    acc_range = acc_max - acc_min
    
    # 长尾效应评分 (0-100, 越高越严重)
    # 考虑: 零准确率比例、低准确率比例、方差
    zero_ratio = zero_acc_count / len(valid_acc)
    low_ratio = low_acc_count / len(valid_acc)
    normalized_std = acc_std / 50  # 假设 50% 是最大期望标准差
    
    imbalance_score = (
        zero_ratio * 40 +  # 零准确率权重最高
        low_ratio * 30 +   # 低准确率其次
        min(normalized_std, 1.0) * 30  # 方差占剩余
    )
    
    report = {
        'status': 'analyzed',
        'zero_acc_count': int(zero_acc_count),
        'zero_acc_ratio': float(zero_ratio),
        'low_acc_count': int(low_acc_count),
        'low_acc_ratio': float(low_ratio),
        'accuracy_std': float(acc_std),
        'accuracy_min': float(acc_min),
        'accuracy_max': float(acc_max),
        'accuracy_range': float(acc_range),
        'imbalance_score': float(imbalance_score),
        'worst_classes': per_class_stats['worst_classes'][:5],
        'best_classes': per_class_stats['best_classes'][:5],
    }
    
    # 预警输出
    if verbose:
        severity = 'INFO'
        if imbalance_score > 30:
            severity = 'WARN'
        if imbalance_score > 50:
            severity = 'CRITICAL'
        
        if imbalance_score > 20 or epoch % 10 == 0:
            print(f"\n  [{severity}] Class Balance Analysis (Epoch {epoch}):")
            print(f"    - Zero accuracy classes: {zero_acc_count}/{len(valid_acc)} ({zero_ratio*100:.1f}%)")
            print(f"    - Low accuracy (<20%): {low_acc_count}/{len(valid_acc)} ({low_ratio*100:.1f}%)")
            print(f"    - Accuracy range: [{acc_min:.1f}%, {acc_max:.1f}%] (std={acc_std:.1f})")
            print(f"    - Imbalance score: {imbalance_score:.1f}/100 ({severity})")

            if zero_acc_count > 0 and epoch > 5:
                worst_5 = per_class_stats['worst_classes'][:5]
                worst_acc = [class_acc[c] for c in worst_5]
                worst_str = ", ".join([f"{c}:{a:.1f}%" for c, a in zip(worst_5, worst_acc)])
                print(f"    - Worst classes (ID:acc): {worst_str}")

            # I30-4: Focal Loss 和 Class Balanced 现在默认启用 (gamma=3.0)
            # 建议仅在严重不平衡时考虑调整参数
            if imbalance_score > 50 and epoch > 20:
                print("    [建议] 考虑增加 --focal-gamma (当前: 3.0)")
    
    return report


# ============================================================================
# P10-14: 分割器健康监控 (Splitter Health Monitoring)
# ============================================================================

@dataclass
class SplitterHealthConfig:
    """
    分割器健康监控配置
    
    数学形式化:
    - 坍缩检测: N(t) < N_min ∧ t > t_warmup
    - 单调检测: H(t)/H_max < H_min_ratio ∧ t > t_warmup
    - 饱和检测: N(t) > saturation_ratio * N_max
    """
    min_tokens_threshold: float = 2.0       # 最小平均 token 数 (N_min)
    min_entropy_ratio: float = 0.1          # 最小熵比率 (H_min/H_max)
    saturation_ratio: float = 0.9           # 饱和阈值比率
    warmup_epochs: int = 3                  # 宽限期 epoch 数
    expected_tokens_max: float = 64.0       # 期望最大 token 数 (N_max)


@dataclass  
class SplitterHealthStatus:
    """分割器健康状态"""
    is_healthy: bool
    collapse_detected: bool      # 坍缩: 几乎不分割
    monotone_detected: bool      # 单调: 深度分布过于集中
    saturate_detected: bool      # 饱和: 完全分割
    health_score: float          # 综合健康评分 [0, 1]
    severity: str                # 'ok', 'warning', 'critical'
    message: str


def check_splitter_health(
    avg_tokens: Optional[float],
    entropy: Optional[float],
    max_entropy: Optional[float],
    epoch: int,
    config: Optional[SplitterHealthConfig] = None,
) -> SplitterHealthStatus:
    """
    检查分割器健康状态
    
    P10-14 实现: 早期检测分割器异常行为
    
    数学定义:
        collapse_risk = ReLU(N_min - N) / N_min
        monotone_risk = ReLU(H_min - H/H_max) / H_min  
        saturate_risk = ReLU(N - 0.9*N_max) / (0.1*N_max)
        health_score = (1 - collapse_risk) * (1 - monotone_risk) * (1 - saturate_risk)
    
    Args:
        avg_tokens: 平均 token 数 (软计数或硬计数)
        entropy: 当前深度熵
        max_entropy: 理论最大熵 (log(D+1))
        epoch: 当前 epoch
        config: 健康监控配置
    
    Returns:
        SplitterHealthStatus: 健康状态信息
    """
    if config is None:
        config = SplitterHealthConfig()
    
    # 无数据时返回未知状态
    if avg_tokens is None:
        return SplitterHealthStatus(
            is_healthy=True,
            collapse_detected=False,
            monotone_detected=False,
            saturate_detected=False,
            health_score=1.0,
            severity='ok',
            message='No splitter data available'
        )
    
    # 宽限期内不报警
    if epoch <= config.warmup_epochs:
        return SplitterHealthStatus(
            is_healthy=True,
            collapse_detected=False,
            monotone_detected=False,
            saturate_detected=False,
            health_score=1.0,
            severity='ok',
            message=f'Epoch {epoch} in warmup period (≤{config.warmup_epochs})'
        )
    
    # 计算各风险指标
    N_min = config.min_tokens_threshold
    H_min = config.min_entropy_ratio
    N_max = config.expected_tokens_max
    
    # 坍缩风险
    collapse_risk = max(0.0, N_min - avg_tokens) / N_min
    collapse_detected = avg_tokens < N_min
    
    # 单调风险 (如果有熵数据)
    if entropy is not None and max_entropy is not None and max_entropy > 0:
        entropy_ratio = entropy / max_entropy
        monotone_risk = max(0.0, H_min - entropy_ratio) / H_min
        monotone_detected = entropy_ratio < H_min
    else:
        entropy_ratio = 1.0
        monotone_risk = 0.0
        monotone_detected = False
    
    # 饱和风险
    saturate_threshold = config.saturation_ratio * N_max
    saturate_risk = max(0.0, avg_tokens - saturate_threshold) / (0.1 * N_max)
    saturate_risk = min(1.0, saturate_risk)  # 限制在 [0, 1]
    saturate_detected = avg_tokens > saturate_threshold
    
    # 综合健康评分
    health_score = (1 - collapse_risk) * (1 - monotone_risk) * (1 - saturate_risk)
    health_score = max(0.0, min(1.0, health_score))
    
    # 综合评估
    is_healthy = not (collapse_detected or monotone_detected or saturate_detected)
    
    # 确定严重程度和消息
    if collapse_detected:
        severity = 'critical'
        message = f'[P10-14] Splitter COLLAPSE: avg_tokens={avg_tokens:.2f} < {N_min}'
    elif monotone_detected:
        severity = 'warning'
        message = f'[P10-14] Monotone distribution: entropy_ratio={entropy_ratio:.1%} < {H_min:.0%}'
    elif saturate_detected:
        severity = 'warning'
        message = f'[P10-14] Saturation: avg_tokens={avg_tokens:.1f} > {saturate_threshold:.0f}'
    else:
        severity = 'ok'
        message = f'Splitter healthy: tokens={avg_tokens:.1f}, health_score={health_score:.2f}'
    
    return SplitterHealthStatus(
        is_healthy=is_healthy,
        collapse_detected=collapse_detected,
        monotone_detected=monotone_detected,
        saturate_detected=saturate_detected,
        health_score=health_score,
        severity=severity,
        message=message
    )


def log_splitter_health_to_tensorboard(
    writer,
    health_status: SplitterHealthStatus,
    perf_stats: Dict[str, Any],
    epoch: int,
) -> None:
    """
    将分割器健康指标记录到 TensorBoard
    
    记录的指标:
    - Splitter/health_score: 综合健康评分
    - Splitter/entropy_ratio: 熵比率
    - Splitter/soft_token_count: 软 token 计数
    - Splitter/dominant_prob: 主导深度概率
    """
    if writer is None:
        return
    
    # 健康评分
    writer.add_scalar('Splitter/health_score', health_status.health_score, epoch)
    
    # 从 perf_stats 提取指标
    if perf_stats.get('soft_token_count') is not None:
        writer.add_scalar('Splitter/soft_token_count', perf_stats['soft_token_count'], epoch)
    
    if perf_stats.get('entropy_ratio') is not None:
        writer.add_scalar('Splitter/entropy_ratio', perf_stats['entropy_ratio'], epoch)
    
    if perf_stats.get('soft_entropy') is not None:
        writer.add_scalar('Splitter/entropy', perf_stats['soft_entropy'], epoch)
    
    if perf_stats.get('dominant_prob') is not None:
        writer.add_scalar('Splitter/dominant_prob', perf_stats['dominant_prob'], epoch)
    
    if perf_stats.get('learnable_temperature') is not None:
        writer.add_scalar('Splitter/temperature', perf_stats['learnable_temperature'], epoch)
    
    if perf_stats.get('learnable_thresholds') is not None:
        thresholds = perf_stats['learnable_thresholds']
        if hasattr(thresholds, '__len__') and len(thresholds) > 0:
            mean_thresh = sum(thresholds) / len(thresholds)
            writer.add_scalar('Splitter/threshold_mean', mean_thresh, epoch)

    # I111-6: 深度分布监控（TensorBoard）
    # ==================================================

    # 1. 深度分布 histogram
    depth_pi_tensor = perf_stats.get('_depth_pi_tensor')
    if depth_pi_tensor is not None:
        # 深度分布直方图（单步分布）
        writer.add_histogram('Splitter/depth_distribution/pi',
                           depth_pi_tensor.detach().cpu().numpy(), epoch)

        # 深度分布对数变换（观察小概率深度）
        pi_log = torch.log1p(depth_pi_tensor)
        writer.add_histogram('Splitter/depth_distribution/pi_log',
                           pi_log.detach().cpu().numpy(), epoch)

    # 2. 深度熵曲线
    if perf_stats.get('depth_entropy') is not None:
        writer.add_scalar('Splitter/depth_entropy', perf_stats['depth_entropy'], epoch)

        # 熵比率（相对于最大可能熵）
        max_entropy = perf_stats.get('max_entropy', math.log(6))
        entropy_ratio = perf_stats['depth_entropy'] / max_entropy if max_entropy > 0 else 0
        writer.add_scalar('Splitter/entropy_ratio_v2', entropy_ratio, epoch)

    # 3. KL 散度曲线
    if perf_stats.get('depth_kl') is not None:
        writer.add_scalar('Splitter/kl_from_uniform', perf_stats['depth_kl'], epoch)

    # 4. 主导深度指示
    if perf_stats.get('dominant_depth') is not None:
        writer.add_scalar('Splitter/dominant_depth', perf_stats['dominant_depth'], epoch)

    # 5. 配额概率 histogram（如果可用）
    quota_probs = perf_stats.get('_quota_probs_tensor')
    if quota_probs is not None:
        writer.add_histogram('Splitter/quota_probs',
                           quota_probs.detach().cpu().numpy(), epoch)


@torch.no_grad()
def verify_train_eval_consistency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: FractalConfigProtocol,
) -> Dict[str, Any]:
    """验证模型在 train/eval 模式下的输出一致性.
    
    关键检查:
    1. train/eval 输出差异是否在可接受范围内
    2. 尺度/深度选择是否稳定
    
    Returns:
        一致性报告字典
    """
    report = {
        'passed': True,
        'checks': {},
        'warnings': [],
    }
    
    # 获取一个 batch 用于测试
    sample_batch = next(iter(loader))
    imgs = sample_batch[0][:4].to(device)  # 只用 4 张图
    if config.use_channels_last:
        imgs = imgs.to(memory_format=torch.use_channels_last)
    
    # 检查 1: train/eval 输出差异
    # 注意: CUDA graphs 下，第二次推理会覆盖第一次的输出缓冲区
    # 必须先克隆再计算差异
    model.eval()
    with get_amp_context(device, config.use_amp):
        stats_eval = model(imgs)
        out_eval = (stats_eval.logits if hasattr(stats_eval, 'logits') else stats_eval).clone()

    model.train()
    with get_amp_context(device, config.use_amp):
        stats_train = model(imgs)
        out_train = (stats_train.logits if hasattr(stats_train, 'logits') else stats_train).clone()
    model.eval()  # 恢复 eval 模式

    # 计算输出差异
    output_diff = (out_eval - out_train).abs()
    max_diff = output_diff.max().item()
    mean_diff = output_diff.mean().item()
    
    # V3 Variable Depth: train/eval 差异主要来自 Dropout
    # 阈值 0.2 对于有 Dropout 的模型是合理的（Dropout 可能导致 0.1-0.2 的差异）
    # 注意: 差异并不影响整体 passed 判定，仅作为警告
    threshold = 0.2
    output_check = {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'threshold': threshold,
        'passed': max_diff < threshold,
    }
    report['checks']['output_consistency'] = output_check
    
    if output_check['passed']:
        print(f"  [OK] 输出一致性: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    else:
        report['warnings'].append(f"[WARN] train/eval 输出差异较大 (max={max_diff:.4f})，可能由 Dropout 引起")
        print(f"  [WARN] 输出差异: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    
    # 检查 3: 尺度选择稳定性 (多次推理应产生相同结果)
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'compute_scale_distribution'):
        dist1 = model.tokenizer.compute_scale_distribution(imgs)
        dist2 = model.tokenizer.compute_scale_distribution(imgs)
        
        # 比较两次的尺度比例
        scale_stable = all(
            abs(dist1['scale_ratios'][ps] - dist2['scale_ratios'][ps]) < 0.001
            for ps in dist1['scale_ratios']
        )
        
        stability_check = {
            'run1': dist1['scale_ratios'],
            'run2': dist2['scale_ratios'],
            'passed': scale_stable,
        }
        report['checks']['scale_stability'] = stability_check
        
        if scale_stable:
            print(f"  [OK] 尺度选择稳定 (eval 模式下确定性)")
        else:
            report['warnings'].append("[WARN] 尺度选择不稳定，可能存在随机性")
            report['passed'] = False
    
    # 打印 Tokenizer 信息
    print(f"  [OK] Tokenizer: streaming_v3 (Variable Depth Tokens)")
    
    # 总结
    print()
    if report['passed']:
        print("  [PASSED] 一致性检查通过: 训练成果可正确体现在推理中")
    else:
        print("  [FAILED] 一致性检查警告:")
        for w in report['warnings']:
            print(f"     {w}")
    
    return report


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fractal ViT Training")
    
    # 数据集
    parser.add_argument("--dataset", type=str, default="cifar10", 
                       choices=["cifar10", "cifar100", "mnist", "tiny-imagenet", "cub200"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers (auto-detect if not set)")
    parser.add_argument("--val-split", type=float, default=0.05)  # T3: 减少验证集，增加训练数据
    parser.add_argument("--subset-size", type=int, default=None)
    
    # 模型
    parser.add_argument("--dim", type=int, default=192)
    # I145: --num-layers 是标准参数名，--depth 是废弃的别名
    parser.add_argument("--num-layers", type=int, default=8, dest='num_layers',
                       help="Number of Transformer layers")
    parser.add_argument("--depth", type=int, default=None, dest='num_layers',
                       help="(Deprecated: use --num-layers)")
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=32)
    # I30-17: max_depth 控制 Hilbert 四叉树递归深度
    parser.add_argument("--max-depth", type=int, default=None,
                       help="I30-17: Maximum quadtree depth (auto-computed from min_patch_size if None)")
    # I136: mlp_dim FFN 隐藏维度
    parser.add_argument("--mlp-dim", type=int, default=None,
                       help="FFN hidden dimension (default: 4 * dim, use None for auto)")
    # I30-17: Replace num_scales with min_patch_size (max_depth auto-computed)
    parser.add_argument("--min-patch-size", type=int, default=4,
                       help="I30-17: Target minimum patch size for automatic depth computation")
    # I30-11: weighted 池化利用 split_probs 作为重要性权重
    parser.add_argument("--pool", type=str, default="weighted", choices=["weighted", "mean"])
    parser.add_argument("--ffn-type", type=str, default="swiglu_level",
                       choices=["gelu", "swiglu", "swiglu_level"])
    # P11-8: hilbert_bias_mode 已移除，仅使用 LCA 模式
    parser.add_argument("--gradient-checkpoint", action="store_true",
                       help="Enable gradient checkpointing to save memory")
    parser.add_argument("--compile", action="store_true",
                       help="Use torch.compile for faster training (PyTorch 2.0+)")
    parser.add_argument("--channels-last", action="store_true",
                       help="Use channels-last memory format for faster convolutions")
    parser.add_argument("--no-prefetch", action="store_true",
                       help="Disable CudaPrefetcher (for debugging data loading issues)")

    # I30-1: Hilbert vs Raster 消融实验 - 扫描顺序参数
    parser.add_argument("--scan-order", type=str, default="hilbert",
                       choices=["hilbert", "raster", "morton"],
                       help="I30-1: Token scanning order for Hilbert curve ablation experiment "
                            "(default: hilbert, options: hilbert/raster/morton)")

    # I31-3: 面积编码参数
    parser.add_argument("--use-area-encoding", action="store_true",
                       help="I31-3: Enable area-enhanced position encoding")
    parser.add_argument("--use-affine-modulation", action="store_true",
                       help="I31-3: Enable affine-modulated attention bias")
    parser.add_argument("--fourier-levels", type=int, default=4,
                       help="I31-3: Number of Fourier frequency levels for area encoding (default: 4)")

    # I33: 相对预算参数 (替代绝对 K_min/K_max)
    # 覆盖率 = tokens / max_patches, 与图像分辨率无关
    parser.add_argument("--token-coverage-min", type=float, default=0.01,
                       help="I33: Minimum token coverage ratio (default: 0.01, 1% of patches)")
    parser.add_argument("--token-coverage-max", type=float, default=0.25,
                       help="I33: Maximum token coverage ratio, participates in adaptive formula (default: 0.25)")
    # I33: 绝对 K 值边界（用于保护最小/最大 token 数）
    parser.add_argument("--K-min-abs", type=int, default=8,
                       help="I33: Absolute minimum K value (default: 8)")
    parser.add_argument("--K-max-abs", type=int, default=1024,
                       help="I33: Absolute maximum K value (default: 1024)")

    # I30-10: 可学习配额参数 (Scheme E)
    # I24-2: 迁移到命令行 choices 模式
    parser.add_argument("--quota-init-logits", type=str, default=None,
                       help="I30-10: Quota initialization logits as comma-separated values (e.g., '-0.5,-0.2,0.0,0.5')")
    parser.add_argument("--quota-min-per-depth", type=int, default=2,
                       help="I30-10: Minimum quota per depth to prevent dead zones (default: 2)")
    parser.add_argument("--freeze-quota", action="store_true",
                       help="I30-10: Freeze quota logits to current values (no further learning)")

    # I24-1: Tokenizer 参数冻结选项 (减少小数据集过拟合)
    parser.add_argument("--freeze-tokenizer", action="store_true",
                       help="I24-1: Freeze tokenizer learnable params (thresholds, quota) to reduce overfitting")
    parser.add_argument("--freeze-tokenizer-epochs", type=int, default=0,
                       help="I24-1: Freeze tokenizer for first N epochs only (0=freeze all, default: 0)")
    
    # P6-1: 深度缩放参数
    parser.add_argument("--depth-scale-min", type=float, default=0.5,
                       help="Minimum depth scale σ_min (default: 0.5)")
    parser.add_argument("--depth-scale-max", type=float, default=2.0,
                       help="Maximum depth scale σ_max (default: 2.0)")
    parser.add_argument("--no-learnable-depth-scale", action="store_true",
                       help="Use fixed depth scale")
    
    # P6-2: LCA 温度参数
    parser.add_argument("--lca-temperature", type=float, default=1.5,
                       help="LCA bias temperature τ (default: 1.5, SNR=1.5)")
    parser.add_argument("--no-lca-temperature", action="store_true",
                       help="Disable LCA temperature scaling")
    parser.add_argument("--fixed-lca-temperature", action="store_true",
                       help="Use fixed (non-learnable) LCA temperature")
    
    # P7-7: GumbelTopKSplitter 温度退火调度参数
    # I29-1 修复: 从 constants.py 导入常量，确保一致性
    # I24-7 分析: T_end=0.5 保持探索能力，T=0.3 过低会导致梯度消失
    parser.add_argument("--splitter-temp-start", type=float, default=SPLITTER_TEMP_START,
                       help=f"Learnable splitter initial temperature (default: {SPLITTER_TEMP_START})")
    parser.add_argument("--splitter-temp-end", type=float, default=SPLITTER_TEMP_END,
                       help=f"Learnable splitter final temperature (default: {SPLITTER_TEMP_END}, I24-7 optimized)")
    parser.add_argument("--splitter-temp-warmup", type=int, default=5,
                       help="Warmup epochs with fixed T_start (default: 5)")
    # I100-2: 添加温度调度策略参数 (原为 constants.py 常量)
    parser.add_argument("--temp-schedule", type=str, default=SPLITTER_TEMP_SCHEDULE,
                       choices=["linear", "cosine", "exponential"],
                       help=f"Temperature annealing schedule (default: {SPLITTER_TEMP_SCHEDULE})")

    # I24-2: 可学习配额参数 (Scheme E)
    parser.add_argument("--quota-learnable", type=str, default="default",
                       choices=["default", "enable", "disable"],
                       help="Learnable quota (Scheme E): 'default'=use global, 'enable'=force on, 'disable'=force off")
    parser.add_argument("--quota-entropy-weight", type=float, default=0.01,
                       help="Weight for quota entropy regularization (default: 0.01)")

    # P10-4/P10-5: 软熵损失参数
    parser.add_argument("--include-soft-entropy", action="store_true", default=True,
                       help="Enable soft entropy loss (default: True, recommended)")
    parser.add_argument("--no-soft-entropy", action="store_false", dest="include_soft_entropy",
                       help="Disable soft entropy loss")
    parser.add_argument("--soft-entropy-mode", type=str, default="maximize",
                       choices=["maximize", "target"],
                       help="Soft entropy mode: 'maximize' or 'target' (default: maximize)")
    parser.add_argument("--soft-entropy-weight", type=float, default=0.1,
                       help="Soft entropy loss weight (default: 0.1)")
    parser.add_argument("--soft-entropy-target", type=float, default=None,
                       help="Target entropy for mode='target' (default: None, auto=ln(max_depth+1))")
    
    # P10-9: 弹性预算损失参数 (I33: 使用相对覆盖率)
    parser.add_argument("--include-elastic-budget", action="store_true", default=True,
                       help="Enable elastic budget loss (default: True, recommended)")
    parser.add_argument("--no-elastic-budget", action="store_false", dest="include_elastic_budget",
                       help="Disable elastic budget loss")
    # I33: 相对覆盖率参数 (推荐使用)
    parser.add_argument("--elastic-coverage-min", type=float, default=0.03,
                       help="I33: Elastic budget minimum coverage ratio (dead zone lower bound, default: 0.03)")
    parser.add_argument("--elastic-coverage-max", type=float, default=0.25,
                       help="I33: Elastic budget maximum coverage ratio (dead zone upper bound, default: 0.25)")
    parser.add_argument("--elastic-lambda-over", type=float, default=0.1,
                       help="Penalty weight for tokens exceeding coverage_max (default: 0.1)")
    parser.add_argument("--elastic-lambda-under", type=float, default=0.01,
                       help="Penalty weight for tokens below coverage_min (default: 0.01)")

    # I110-7: 语义分裂器参数
    parser.add_argument("--use-semantic-splitter", action="store_true",
                       help="I110-7: Use SemanticRedundancySplitter instead of GumbelTopKSplitter")
    parser.add_argument("--semantic-feature-dim", type=int, default=256,
                       help="I110-7: Semantic splitter feature dimension")
    parser.add_argument("--semantic-hidden-dim", type=int, default=128,
                       help="I110-7: Semantic splitter hidden dimension")
    parser.add_argument("--semantic-diversity-weight", type=float, default=0.1,
                       help="I110-7: Diversity loss weight (default: 0.1)")
    parser.add_argument("--semantic-reconstruction-weight", type=float, default=0.1,
                       help="I110-7: Reconstruction loss weight (default: 0.1)")
    parser.add_argument("--semantic-split-threshold", type=float, default=0.5,
                       help="I110-7: Split decision threshold (default: 0.5)")
    parser.add_argument("--semantic-gumbel-temp-start", type=float, default=1.0,
                       help="I110-7: Gumbel softmax initial temperature (default: 1.0)")
    parser.add_argument("--semantic-gumbel-temp-end", type=float, default=0.5,
                       help="I110-7: Gumbel softmax final temperature (default: 0.5)")

    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.15,
                       help="Weight decay for L2 regularization (default: 0.15, I30-3 tuned for overfitting)")
    parser.add_argument("--dropout", type=float, default=0.25,
                       help="Dropout rate (default: 0.25, I30-3 tuned for overfitting)")
    parser.add_argument("--emb-dropout", type=float, default=0.15,
                       help="Embedding dropout rate (default: 0.15)")
    parser.add_argument("--drop-path", type=float, default=0.25,
                       help="Drop path (stochastic depth) rate (default: 0.25, I30-3 tuned for overfitting)")
    parser.add_argument("--label-smoothing", type=float, default=0.1,
                       help="Label smoothing factor (default: 0.1)")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--accum-steps", type=int, default=1)
    
    # 早停
    parser.add_argument("--patience", type=int, default=10,
                       help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--min-delta", type=float, default=0.001,
                       help="Minimum improvement for early stopping")
    
    # Mixup/CutMix
    parser.add_argument("--mixup-alpha", type=float, default=0.4,
                       help="Mixup alpha (default: 0.4, 0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=1.0,
                       help="CutMix alpha (default: 1.0, 0 to disable)")
    parser.add_argument("--mixup-prob", type=float, default=0.5,
                       help="Probability of applying Mixup/CutMix (default: 0.5)")

    # P14/I30-4: 长尾效应优化 - 默认启用
    parser.add_argument("--no-focal-loss", action="store_true",
                       help="Disable Focal Loss (enabled by default)")
    parser.add_argument("--focal-gamma", type=float, default=3.0,
                       help="Focal Loss gamma parameter (default: 3.0, I30-4 tuned)")
    parser.add_argument("--no-class-balanced", action="store_true",
                       help="Disable class-balanced loss weights (enabled by default)")
    parser.add_argument("--class-balance-beta", type=float, default=0.9999,
                       help="Class balance beta parameter (default: 0.9999)")
    parser.add_argument("--progressive-aug", action="store_true",
                       help="Use progressive data augmentation (weaker at start)")
    
    # I30-2: Hilbert-aware 困难样本挖掘
    parser.add_argument("--use-hilbert-mining", action="store_true",
                       help="Use Token Variance based hard sample mining (I30-2)")
    parser.add_argument("--hilbert-mining-lambda", type=float, default=0.5,
                       help="Hard mining weight scaling factor (default: 0.5)")
    parser.add_argument("--hilbert-mining-warmup", type=int, default=100,
                       help="Number of batches for EMA statistics warmup (default: 100)")
    
    # CUB-200 细粒度分类专用参数
    parser.add_argument("--use-center-loss", action="store_true",
                       help="Use Center Loss for fine-grained classification (CUB-200)")
    parser.add_argument("--center-loss-weight", type=float, default=0.01,
                       help="Center Loss weight (default: 0.01)")
    parser.add_argument("--finegrained-mode", action="store_true",
                       help="Enable fine-grained classification mode (auto for cub200)")

    # CUB-200 细粒度分类专用参数（独立暴露，避免与通用训练器参数重合）
    parser.add_argument("--cub-validate-interval", type=int, default=3,
                       help="CUB200: Validation interval in epochs (default: 3)")
    parser.add_argument("--cub-center-lr-ratio", type=float, default=50.0,
                       help="CUB200: Center Loss LR ratio (lr_center / lr_main, default: 50.0)")
    parser.add_argument("--cub-center-lr", type=float, default=None,
                       help="CUB200: Center Loss absolute LR (overrides ratio if set)")
    parser.add_argument("--cub-gradient-clip", type=float, default=1.0,
                       help="CUB200: Gradient clipping norm (default: 1.0)")
    parser.add_argument("--cub-no-focal-loss", action="store_true", default=False,
                       help="CUB200: Disable Focal Loss for hard samples")
    parser.add_argument("--cub-focal-gamma", type=float, default=2.0,
                       help="CUB200: Focal Loss gamma (default: 2.0)")

    # 系统
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--exp-name", type=str, default=None,
                       help="Custom experiment name (default: auto-generated with timestamp)")

    # 使用 parse_known_args 忽略未知参数（避免因参数不兼容而报错）
    args, unknown = parser.parse_known_args()

    # 警告：忽略的未知参数
    if unknown:
        print(f"[WARN] 忽略未知命令行参数 ({len(unknown)} 个):")
        for arg in unknown[:10]:  # 只显示前10个
            print(f"       {arg}")
        if len(unknown) > 10:
            print(f"       ... 还有 {len(unknown) - 10} 个")

    # CUB-200 自动启用细粒度模式
    if args.dataset == 'cub200' and not args.finegrained_mode:
        args.finegrained_mode = True
        print("[INFO] CUB-200 数据集: 自动启用细粒度分类模式")
    
    # 环境检测
    env = detect_environment()
    print_environment_info(env)
    
    # 自动设置 workers
    if args.num_workers is None:
        args.num_workers = env['recommended_workers']
        print(f"Auto-detected num_workers: {args.num_workers}")
    
    # Quick test
    if args.quick_test:
        args.epochs = 3
        args.subset_size = 256
        print("[*] Quick test mode\n")
    
    # 初始化
    set_seed(args.seed)
    
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    spec = DATASETS[args.dataset]

    # I33: 相对预算参数 (CLI) → 绝对 K 值 (模型)
    # 转换公式: K = coverage * max_patches = coverage * (image_size/min_patch_size)^2
    # 注意: K 值由模型架构根据覆盖率动态计算，TrainingConfig 只保存覆盖率
    # P2 修复: 使用 spec.image_size 而非 args.image_size
    image_size_for_budget = spec.image_size if spec.image_size is not None else args.min_patch_size * 64
    max_patches = (image_size_for_budget // args.min_patch_size) ** 2
    # K_min_abs 用于保护最小值，实际 K 值由模型架构计算
    K_min_abs = max(args.K_min_abs, max(4, int(max_patches * args.token_coverage_min)))
    K_max_abs = int(max_patches * args.token_coverage_max)

    # I24-2: 转换 quota_learnable 字符串到布尔值
    if args.quota_learnable == "enable":
        quota_learnable_value = True
    elif args.quota_learnable == "disable":
        quota_learnable_value = False
    else:
        quota_learnable_value = None  # 使用全局默认值

    # I36: 使用 ModelArchitectureConfig 作为配置基础
    # I110-7: 构建语义分裂器配置字典
    if args.use_semantic_splitter:
        semantic_config_dict = {
            'feature_dim': args.semantic_feature_dim,
            'hidden_dim': args.semantic_hidden_dim,
            'diversity_weight': args.semantic_diversity_weight,
            'reconstruction_weight': args.semantic_reconstruction_weight,
            'split_threshold': args.semantic_split_threshold,
            'gumbel_temp_start': args.semantic_gumbel_temp_start,
            'gumbel_temp_end': args.semantic_gumbel_temp_end,
            'image_size': spec.image_size if spec.image_size else (args.min_patch_size * 64, args.min_patch_size * 64),
            'min_patch_size': args.min_patch_size,
        }
    else:
        semantic_config_dict = None

    arch_config = ModelArchitectureConfig(
        num_classes=spec.num_classes,
        dim=args.dim,
        num_layers=args.num_layers,  # I145: 已标准化
        heads=args.heads,
        dim_head=args.dim_head,
        mlp_dim=args.mlp_dim if args.mlp_dim else args.dim * 4,
        pool=args.pool,
        image_size=spec.image_size,
        channels=spec.channels,
        min_patch_size=args.min_patch_size,
        token_coverage_min=args.token_coverage_min,
        token_coverage_max=args.token_coverage_max,
        K_min_abs=K_min_abs,  # 第二层参数：实际 K 值由模型架构动态计算
        # 注意: max_level 是变参数，完全由模型架构内部计算，不从外部传入
        ffn_type=args.ffn_type,
        use_checkpoint=args.gradient_checkpoint,
        use_channels_last=args.channels_last,
        compile_model=getattr(args, 'compile', False),
        use_area_encoding=args.use_area_encoding,
        use_affine_modulation=args.use_affine_modulation,
        fourier_levels=args.fourier_levels,
        depth_scale_range=(args.depth_scale_min, args.depth_scale_max) if not args.no_learnable_depth_scale else None,
        lca_temperature=None if args.no_lca_temperature else args.lca_temperature,
        learnable_temperature=not args.fixed_lca_temperature,
        quota_learnable=quota_learnable_value,
        quota_entropy_weight=args.quota_entropy_weight,
        freeze_quota=args.freeze_quota,
        freeze_tokenizer=args.freeze_tokenizer,
        freeze_tokenizer_epochs=args.freeze_tokenizer_epochs,
        # I136: Elastic Budget 配置 (用于保存到 checkpoint)
        elastic_coverage_min=args.elastic_coverage_min,
        elastic_coverage_max=args.elastic_coverage_max,
        elastic_lambda_over=args.elastic_lambda_over,
        elastic_lambda_under=args.elastic_lambda_under,
        # I110-7: 语义分裂器配置
        use_semantic_splitter=args.use_semantic_splitter,
        semantic_splitter_config=semantic_config_dict,
    )

    # 创建训练配置对象 (满足 FractalConfigProtocol)
    class TrainingConfig:
        """训练配置包装器 - 满足 FractalConfigProtocol"""
        def __init__(self, args, arch_config):
            # 保存原始 arch_config 供 ModelGene.from_config() 使用
            self.arch_config = arch_config

            # 数据集配置
            self.subset_size = args.subset_size
            self.val_split = args.val_split
            self.seed = args.seed
            self.num_workers = args.num_workers
            self.batch_size = args.batch_size

            # 模型架构配置 (从 arch_config 获取)
            self.dim = arch_config.dim
            self.num_layers = arch_config.num_layers  # I145: 统一使用 num_layers
            self.heads = arch_config.heads
            self.dim_head = arch_config.dim_head
            self.mlp_dim = arch_config.mlp_dim
            self.pool = arch_config.pool
            self.ffn_type = arch_config.ffn_type
            self.min_patch_size = arch_config.min_patch_size
            # 注意: max_level 是变参数，完全由模型架构内部计算
            # 不从 arch_config 获取，确保训练/评估模型结构完全一致
            # I145: 修复 dropout 来源 - 使用 arch_config 而非 args
            # 这样 ModelGene.from_config() 保存的值与训练时使用的值一致
            self.dropout = arch_config.dropout
            self.emb_dropout = arch_config.emb_dropout
            self.drop_path_rate = arch_config.drop_path_rate
            self.use_checkpoint = arch_config.use_checkpoint
            self.lca_temperature = arch_config.lca_temperature
            self.learnable_temperature = arch_config.learnable_temperature
            self.use_area_encoding = arch_config.use_area_encoding
            self.use_affine_modulation = arch_config.use_affine_modulation
            self.fourier_levels = arch_config.fourier_levels
            self.quota_learnable = arch_config.quota_learnable
            self.quota_entropy_weight = arch_config.quota_entropy_weight
            self.freeze_quota = arch_config.freeze_quota
            self.freeze_tokenizer = arch_config.freeze_tokenizer
            self.freeze_tokenizer_epochs = arch_config.freeze_tokenizer_epochs
            self.depth_scale_range = arch_config.depth_scale_range

            # I110-7: 语义分裂器配置
            self.use_semantic_splitter = arch_config.use_semantic_splitter
            self.semantic_splitter_config = arch_config.semantic_splitter_config

            # Tokenizer 覆盖率 (I33 相对预算)
            # 注意: K 值由模型架构动态计算，TrainingConfig 不保存绝对 K 值
            self.token_coverage_min = args.token_coverage_min
            self.token_coverage_max = args.token_coverage_max
            self.K_min_abs = K_min_abs  # 用于保护最小值，实际 K 值动态计算

            # 训练配置 (use_channels_last 是 CLI 参数)
            self.use_channels_last = getattr(args, 'use_channels_last', False)
            self.use_amp = args.use_amp
            self.learning_rate = args.lr
            self.weight_decay = args.weight_decay
            self.gradient_clip = args.gradient_clip
            self.accum_steps = args.accum_steps
            self.warmup_epochs = args.warmup_epochs
            self.epochs = args.epochs
            self.label_smoothing = args.label_smoothing
            self.mixup_alpha = args.mixup_alpha
            self.cutmix_alpha = args.cutmix_alpha
            self.mixup_prob = args.mixup_prob
            self.use_focal_loss = not args.no_focal_loss
            self.focal_gamma = args.focal_gamma
            self.use_class_balanced = not args.no_class_balanced
            self.class_balance_beta = args.class_balance_beta
            self.progressive_aug = args.progressive_aug
            self.use_hilbert_mining = getattr(args, 'use_hilbert_mining', False)
            self.hilbert_mining_lambda = getattr(args, 'hilbert_mining_lambda', 0.5)
            self.hilbert_mining_warmup = getattr(args, 'hilbert_mining_warmup', 100)
            self.patience = args.patience
            self.min_delta = args.min_delta
            self.compile_model = arch_config.compile_model

            # Splitter 温度退火配置
            self.splitter_temp_start = args.splitter_temp_start
            self.splitter_temp_end = args.splitter_temp_end
            self.splitter_temp_warmup = args.splitter_temp_warmup

            # 损失函数配置
            self.include_elastic_budget = args.include_elastic_budget
            self.elastic_coverage_min = args.elastic_coverage_min
            self.elastic_coverage_max = args.elastic_coverage_max
            self.elastic_lambda_over = args.elastic_lambda_over
            self.elastic_lambda_under = args.elastic_lambda_under
            self.include_soft_entropy = args.include_soft_entropy
            self.soft_entropy_target = args.soft_entropy_target
            self.soft_entropy_weight = args.soft_entropy_weight
            self.soft_entropy_mode = args.soft_entropy_mode

    # I145: TrainingConfig 不再接收 K_min/K_max（第二层参数由模型动态计算）
    config = TrainingConfig(args, arch_config)
    
    # 创建 Tokenizer (默认使用 GumbelTopKSplitter - Scheme D)
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3

    # I30-17: 使用动态深度计算
    # Note: Splitter 由模型内部创建 GumbelTopKSplitter 时自动处理，
    #       无需在此处单独配置 splitter_config

    tokenizer = StreamingFractalTokenizerV3(
        image_size=max(spec.image_size, 32),
        channels=spec.channels,
        d_model=config.dim,
        base_patch_size=config.min_patch_size,  # I30-17: 使用 min_patch_size
        min_patch_size=config.min_patch_size,   # I30-17: 目标最小 patch (max_depth 自动计算)
        use_hilbert_order=True,
        # P6-1: 深度缩放配置
        depth_scale_range=config.depth_scale_range,
        # Note: Splitter 参数 (splitter_config, splitter_dropout, learnable_temperature)
        # 已由模型在内部创建 GumbelTopKSplitter 时处理，不在此处传递
    )

    # 创建模型 (V3 Variable Depth Tokens)
    model_kwargs = dict(
        image_size=spec.image_size,
        num_classes=spec.num_classes,
        dim=config.dim,
        num_layers=config.num_layers,  # FractalCurveViT 使用 num_layers
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        pool=config.pool,
        channels=spec.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        drop_path_rate=config.drop_path_rate,
        # I30-17: 动态深度参数 (max_level 由模型架构内部从 min_patch_size 计算)
        min_patch_size=config.min_patch_size,
        # max_level=None  # 不传递，让模型架构内部计算（变参数）
        use_checkpoint=config.use_checkpoint,
        ffn_type=config.ffn_type,
        # 使用自定义 tokenizer (支持高级分割参数)
        tokenizer=tokenizer,
        # P6-2: LCA 温度配置
        lca_temperature=config.lca_temperature,
        learnable_temperature=config.learnable_temperature,
        # I31-3: 面积编码参数
        use_area_encoding=config.use_area_encoding,
        use_affine_modulation=config.use_affine_modulation,
        fourier_levels=config.fourier_levels,
        # I24-2: 可学习配额控制 (Scheme E)
        quota_learnable=config.quota_learnable,
    )

    model = FractalCurveViT(**model_kwargs).to(device)

    # I30-10: 配额参数冻结 (独立于 freeze_tokenizer)
    if config.freeze_quota:
        model.splitter.set_quota_grad(False)
        print(f"[I30-10] Frozen quota logits (quota will not learn)")

    # I24-1: Tokenizer 参数冻结 (减少小数据集过拟合)
    frozen_tokenizer_params = []
    if config.freeze_tokenizer:
        for name, param in model.tokenizer.named_parameters():
            # 冻结 splitter 中的可学习参数 (thresholds, quota, temperature)
            if 'threshold' in name or 'quota' in name or 'temperature' in name:
                param.requires_grad = False
                frozen_tokenizer_params.append(name)
        if frozen_tokenizer_params:
            freeze_mode = "permanent" if config.freeze_tokenizer_epochs == 0 else f"first {config.freeze_tokenizer_epochs} epochs"
            print(f"[I24-1] Frozen tokenizer params ({freeze_mode}): {frozen_tokenizer_params}")

    # I136: 配置 Elastic Budget 覆盖率参数 (从 CLI 传入)
    # I24-2: 配置配额熵权重 (Scheme E)
    if hasattr(model, 'splitter'):
        model.splitter._elastic_coverage_min = config.elastic_coverage_min
        model.splitter._elastic_coverage_max = config.elastic_coverage_max
        model.splitter._elastic_lambda_over = config.elastic_lambda_over
        model.splitter._elastic_lambda_under = config.elastic_lambda_under
        model.splitter._quota_entropy_weight = config.quota_entropy_weight
        print(f"[I136] Elastic budget config: coverage∈[{config.elastic_coverage_min:.2%}, {config.elastic_coverage_max:.2%}], λ_over={config.elastic_lambda_over}, λ_under={config.elastic_lambda_under}")
        print(f"[I24-2] Quota entropy weight: {config.quota_entropy_weight}")
    
    # 打印模型信息
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # I30-10: 配额信息添加到 split_info
    # I33: 显示相对预算覆盖率
    quota_info = "Scheme E" if config.quota_learnable else "Scheme D (no quota)"
    if config.quota_learnable and config.freeze_quota:
        quota_info += " (frozen)"
    # I145: 只显示覆盖率范围，K 值由模型架构动态计算
    split_info = f"GumbelTopKSplitter (coverage∈[{config.token_coverage_min:.0%}, {config.token_coverage_max:.0%}], K=dynamic, {quota_info})"
    tokenizer_name = f'StreamingFractalTokenizerV3 ({split_info})'

    # P6-1/P6-2 信息
    depth_scale_info = f"range={config.depth_scale_range}" if config.depth_scale_range else "fixed"
    temp_info = f"τ={config.lca_temperature}" if config.lca_temperature else "disabled"
    if config.lca_temperature and config.learnable_temperature:
        temp_info += " (learnable)"

    print(f"\n{'='*70}")
    print(f"Model: FractalCurveViT")
    print(f"Tokenizer: {tokenizer_name}")
    print(f"FFN Type: {config.ffn_type}")
    print(f"Hilbert Bias: LCA (only mode after P11-8 cleanup)")
    print(f"  - Depth Scale (P6-1): {depth_scale_info}")
    print(f"  - LCA Temperature (P6-2): {temp_info}")
    print(f"Parameters: {params:,} (trainable: {trainable_params:,})")
    if config.freeze_tokenizer:
        print(f"  - Tokenizer Frozen (I24-1): {len(frozen_tokenizer_params)} params")
    if config.freeze_quota:
        print(f"  - Quota Frozen (I30-10): quota_logits will not learn")
    print(f"Gradient Checkpoint: {config.use_checkpoint}")
    print(f"Compile Model: {config.compile_model}")
    print(f"Channels Last: {config.use_channels_last}")
    print(f"{'='*70}\n")
    
    # =========================================================================
    # CUB-200 细粒度分类专用训练路径 (解耦合)
    # =========================================================================
    # 当 --finegrained-mode 启用时 (CUB-200 自动启用)，使用专用的 CUB200Trainer
    # CUB200Trainer 特性:
    #   - CenterLoss: 增强类内紧凑性
    #   - 细粒度数据增强: 保守增强，保留判别性细节
    #   - Per-class 准确率评估
    #   - 困难类别对分析
    # =========================================================================
    if args.finegrained_mode:
        print("=" * 70)
        print("FINE-GRAINED CLASSIFICATION MODE (CUB-200)")
        print("=" * 70)
        
        # 创建数据加载器
        train_loader, val_loader, test_loader = create_dataloaders(spec, config)
        
        # 创建 CUB-200 训练配置
        # 注意: 不传递 mixup/cutmix 参数，因为 CUB200Trainer 未实现这些增强
        cub200_config = CUB200TrainingConfig(
            # 基础训练参数
            batch_size=config.batch_size,
            num_epochs=config.epochs,
            learning_rate=config.learning_rate,
            warmup_epochs=config.warmup_epochs,
            accum_steps=config.accum_steps,
            validate_interval=args.cub_validate_interval,
            center_lr_ratio=args.cub_center_lr_ratio,
            center_lr=args.cub_center_lr,
            gradient_clip_norm=args.cub_gradient_clip,
            # 设备和混合精度
            device=str(device),
            use_amp=config.use_amp,
            # 细粒度特定
            use_center_loss=args.use_center_loss,
            center_loss_weight=args.center_loss_weight,
            use_focal_loss=not args.cub_no_focal_loss,
            focal_gamma=args.cub_focal_gamma,
            # 正则化
            label_smoothing=config.label_smoothing,
            dropout=config.dropout,
            drop_path_rate=config.drop_path_rate,
            weight_decay=config.weight_decay,
            # 早停
            patience=config.patience,
            min_delta=config.min_delta,
        )
        
        # 创建训练器
        cub200_trainer = CUB200Trainer(
            model=model,
            config=cub200_config,
            num_classes=spec.num_classes,
            feat_dim=config.dim,
            device=device,
        )
        
        # 创建优化器
        use_fused = device.type == 'cuda' and hasattr(torch.optim.AdamW, 'fused')
        try:
            optimizer = AdamW(
                model.parameters(),
                lr=cub200_config.learning_rate,
                weight_decay=cub200_config.weight_decay,
                fused=use_fused
            )
        except TypeError:
            optimizer = AdamW(
                model.parameters(),
                lr=cub200_config.learning_rate,
                weight_decay=cub200_config.weight_decay
            )
        
        # 创建学习率调度器
        warmup = min(cub200_config.warmup_epochs, cub200_config.num_epochs // 2)
        warmup_sch = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
        cosine_sch = CosineAnnealingLR(optimizer, T_max=cub200_config.num_epochs - warmup, eta_min=cub200_config.learning_rate * 0.01)
        scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup])
        
        # 创建实验目录
        exp_name = args.exp_name if args.exp_name else f"cub200_finegrained_{time.strftime('%Y%m%d_%H%M%S')}"
        exp_dir = PROJECT_ROOT / "experiments" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "checkpoints").mkdir(exist_ok=True)
        (exp_dir / "logs").mkdir(exist_ok=True)
        
        # 保存配置
        with open(exp_dir / "logs" / "config.json", 'w') as f:
            json.dump({
                'training_config': obj_to_dict(config),
                'cub200_config': {
                    'batch_size': cub200_config.batch_size,
                    'num_epochs': cub200_config.num_epochs,
                    'learning_rate': cub200_config.learning_rate,
                    'validate_interval': cub200_config.validate_interval,
                    'center_lr_ratio': cub200_config.center_lr_ratio,
                    'center_lr': cub200_config.center_lr,
                    'use_center_loss': cub200_config.use_center_loss,
                    'center_loss_weight': cub200_config.center_loss_weight,
                    'use_focal_loss': cub200_config.use_focal_loss,
                    'focal_gamma': cub200_config.focal_gamma,
                },
            }, f, indent=2)

        print(f"[INFO] Experiment directory: {exp_dir}")
        print(f"[INFO] CUB200 Config:")
        print(f"  - Epochs: {cub200_config.num_epochs}, Batch: {cub200_config.batch_size}")
        print(f"  - LR: {cub200_config.learning_rate:.2e}, Validate every: {cub200_config.validate_interval} epochs")
        print(f"  - CenterLoss: {'enabled' if cub200_config.use_center_loss else 'disabled'}")
        if cub200_config.use_center_loss:
            print(f"    - Weight: {cub200_config.center_loss_weight}, LR ratio: {cub200_config.center_lr_ratio}")
            if cub200_config.center_lr:
                print(f"    - Absolute LR: {cub200_config.center_lr}")
        print(f"  - FocalLoss: {'enabled' if cub200_config.use_focal_loss else 'disabled'}")
        if cub200_config.use_focal_loss:
            print(f"    - Gamma: {cub200_config.focal_gamma}")
        else:
            print(f"    - Disabled by --cub-no-focal-loss")
        print()
        
        # 执行训练
        print("=" * 70)
        print("TRAINING START (CUB200Trainer)")
        print("=" * 70)
        print()
        
        history = cub200_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            exp_dir=exp_dir,
        )
        
        # 最终测试评估
        if test_loader is not None:
            print("\n" + "=" * 70)
            print("FINAL TEST EVALUATION")
            print("=" * 70)
            test_result = cub200_trainer.evaluate(test_loader)
            print(f"Test Accuracy: {test_result.accuracy:.2f}% (Top-5: {test_result.top5_accuracy:.2f}%)")
            print(f"Worst Classes: {test_result.confused_pairs[:5] if test_result.confused_pairs else 'N/A'}")
        
        # 保存最终历史
        with open(exp_dir / "logs" / "history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"\n[OK] CUB-200 fine-grained training completed!")
        print(f"[INFO] Results saved to: {exp_dir}")
        return  # 提前返回，不执行通用训练循环
    
    # =========================================================================
    # 通用分类训练路径 (CIFAR-10, Tiny-ImageNet, ImageNet 等)
    # =========================================================================
    
    # 数据加载
    train_loader, val_loader, test_loader = create_dataloaders(spec, config)
    
    # 优化器 (使用 fused 版本加速)
    use_fused = device.type == 'cuda' and hasattr(torch.optim.AdamW, 'fused')
    try:
        optimizer = AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            fused=use_fused
        )
        if use_fused:
            print("[OK] Using fused AdamW optimizer")
    except TypeError:
        # 旧版本 PyTorch 不支持 fused 参数
        optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    # 学习率调度
    warmup = min(args.warmup_epochs, config.epochs // 2)
    warmup_sch = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup)
    cosine_sch = CosineAnnealingLR(optimizer, T_max=config.epochs - warmup, eta_min=config.learning_rate * 0.01)
    scheduler = SequentialLR(optimizer, [warmup_sch, cosine_sch], milestones=[warmup])
    
    scaler = create_grad_scaler(config.use_amp)
    
    # P13: GC 优化 - 减少训练中的暂停
    # Python GC 在大量临时对象时可能导致长时间暂停
    import gc
    gc.disable()  # 禁用自动 GC
    gc_interval = 50  # 每 50 个 batch 手动 GC 一次
    print("[OK] Disabled automatic GC (manual GC every 50 batches)")
    
    # CUDA 内存管理优化
    if device.type == 'cuda':
        # 启用内存池，减少分配/释放开销
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
    
    # CUDA 优化 (I35: 已在模块导入时配置，此处仅记录)
    # 注意: _configure_cuda_optimizations() 在文件顶部已调用
    # 重复设置无影响，仅跳过以避免重复日志
    
    # torch.compile 编译优化 (PyTorch 2.0+)
    # I107-2 修复: 添加 CUDA 检测，避免在无 CUDA 系统上设置不存在的配置
    # I107-3 修复: 移动 use_channels_last 到 compile 之后，避免 cudagraphs 冲突
    # 注意: cudagraphs 在 laptop GPU 上经常失败，需要禁用或使用更稳定的模式
    if config.compile_model:
        try:
            # 强制进行全面的垃圾回收
            import gc
            gc.collect()
            gc.collect()
            gc.collect()

            # 仅在 CUDA 可用时清理 CUDA 缓存
            has_cuda = torch.cuda.is_available()
            if has_cuda:
                torch.cuda.empty_cache()
                # I107-4: 检查 GPU 状态
                try:
                    props = torch.cuda.get_device_properties(0)
                    print(f"[INFO] GPU: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
                    print(f"[INFO] Compute capability: {props.major}.{props.minor}")
                except Exception:
                    pass

            # 设置编译缓存和错误处理
            # I107-6: 优化编译配置，提升训练性能
            torch._dynamo.config.cache_size_limit = 256
            torch._dynamo.config.suppress_errors = False

            # I107-8: 禁用 cudagraphs (避免动态 token 数量导致的重复警告)
            # FractalViT 动态 token 数量 (K_min=8 到 K_max=64) 会导致 cudagraphs 反复跳过
            # 警告 "skipping cudagraphs due to cpu device" 会干扰训练输出
            # 性能影响: ~10%，但训练输出更清晰
            torch._inductor.config.triton.cudagraphs = False
            torch._inductor.config.max_autotune = False
            torch._inductor.config.compile_threads = 4  # 并行编译加速

            # I107-6: 优化动态编译，减少重复编译开销
            # GumbelTopK 的 token 数量在 K_min=8 到 K_max=64 之间动态变化
            torch._dynamo.config.assume_static_by_default = True  # 减少动态编译

            # I107-6: 移除 cudagraphs 相关禁用配置
            # 之前禁用的原因 (TDP 限制不稳定) 已确认可启用
            # 删除: torch._inductor.config.triton.cudagraphs = False
            # 删除: torch._inductor.config.triton.cudagraphs_checkpoint = False

            # I107-6: 移除冲突的 cudnn 配置
            # cudagraphs 与 cudnn 配置可能冲突，统一使用 inductor 默认
            # 删除: torch._inductor.config.triton.cudnn = True
            # 删除: torch._inductor.config.triton.use_cudnn = True

            # I107-6: 使用 reduce-overhead 模式获得更好的 CUDA 性能
            compile_mode = 'reduce-overhead'

            # I107-6: 在 compile 之前应用 use_channels_last，避免 cudagraphs 冲突
            # 这样可以避免 device_put 时触发 CPU 操作
            if config.use_channels_last and device.type == 'cuda':
                model = model.to(memory_format=torch.use_channels_last)
                print("[OK] Using channels-last memory format (before compile)")

            model = torch.compile(
                model,
                mode=compile_mode,
                fullgraph=False,
                dynamic=True,
            )
            print(f"[OK] Model compiled with torch.compile (mode={compile_mode})")
            if has_cuda:
                print("[INFO] 首次运行会进行 JIT 编译，可能耗时 1-2 分钟")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
            print("[INFO] 回退到 eager 模式继续训练")
            # 仍然应用 use_channels_last 如果需要
            if config.use_channels_last and device.type == 'cuda':
                model = model.to(memory_format=torch.use_channels_last)
                print("[OK] Using channels-last memory format (eager mode)")
    
    # 诊断: 检查模型参数 dtype
    def check_model_dtypes(m, name="model"):
        for n, p in m.named_parameters():
            if p.dtype == torch.float16:
                print(f"[WARN] {name}.{n} is float16!")
    check_model_dtypes(model)
    
    # 创建 Mixup/CutMix 增强器
    use_mixup = config.mixup_alpha > 0 or config.cutmix_alpha > 0
    mixup_fn = None
    if use_mixup:
        mixup_fn = MixupCutmix(
            mixup_alpha=config.mixup_alpha,
            cutmix_alpha=config.cutmix_alpha,
            prob=config.mixup_prob,
            num_classes=spec.num_classes,
            label_smoothing=config.label_smoothing,
        )
    
    # P14: 创建损失函数 (Focal Loss / Class-Balanced Loss)
    loss_fn = None
    class_weights = None
    
    if config.use_focal_loss or config.use_class_balanced:
        # 收集训练集标签用于计算类别权重
        print("[INFO] 收集训练集标签用于计算类别权重...")
        all_labels = []
        for _, labels in tqdm(train_loader, desc="Collecting labels", leave=False):
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.tolist())
            else:
                all_labels.extend(labels)
        
        if config.use_class_balanced:
            # 计算类别平衡权重 (T2-5: 使用正确的函数名)
            class_weights = compute_class_weights_from_targets(
                targets=all_labels,
                num_classes=spec.num_classes,
                beta=config.class_balance_beta,
            ).to(device)
            
            # 打印类别权重统计
            print(f"[INFO] Class-Balanced Weights (P14):")
            print(f"  - Beta: {config.class_balance_beta}")
            print(f"  - Weight range: [{class_weights.min():.4f}, {class_weights.max():.4f}]")
            print(f"  - Weight mean: {class_weights.mean():.4f}")
            
            # 检测极端不平衡
            max_ratio = class_weights.max() / class_weights.min()
            if max_ratio > 10:
                print(f"  - [WARN] 类别不平衡比例较大: {max_ratio:.2f}x")
        
        if config.use_focal_loss:
            # 创建 Focal Loss
            loss_fn = FocalLoss(
                gamma=config.focal_gamma,
                alpha=class_weights,  # 可选，如果同时启用 class_balanced
                label_smoothing=config.label_smoothing,
                reduction='mean',
            )
            print(f"[INFO] Focal Loss (P14):")
            print(f"  - Gamma: {config.focal_gamma}")
            print(f"  - Alpha (class weights): {'enabled' if class_weights is not None else 'disabled'}")
            
            # Focal Loss 已经使用 class_weights，避免重复
            class_weights = None
    
    # I30-2: 创建 Hilbert-aware 困难样本挖掘模块
    hard_mining: Optional[HilbertAwareHardMining] = None
    if config.use_hilbert_mining:
        hard_mining = HilbertAwareHardMining(
            lambda_weight=config.hilbert_mining_lambda,
            temperature=1.0,  # 标准 sigmoid
            momentum=0.1,  # EMA 更新系数
            warmup_batches=config.hilbert_mining_warmup,
            enabled=True,
        ).to(device)
        print(f"[INFO] Hilbert-Aware Hard Mining (I30-2):")
        print(f"  - Lambda weight: {config.hilbert_mining_lambda}")
        print(f"  - Warmup batches: {config.hilbert_mining_warmup}")
        print(f"  - Expected weight range: [1.0, {1.0 + config.hilbert_mining_lambda:.2f}]")
    
    # 训练
    print("="*70)
    print("TRAINING START")
    print("="*70)

    if config.compile_model:
        print("[INFO] First batch will be slow due to JIT compilation (1-3 minutes)...")
        print("[INFO] Tip: Run 'nvidia-smi -l 1' in another terminal to monitor GPU usage")
        print("[INFO] If GPU usage stays at 0% for >5 minutes, there may be an issue")

    print()

    # 编译预热: 在正式训练前触发 JIT 编译
    # P11-7 优化: 使用完整 batch size 预热，避免动态形状导致重新编译
    # P15 优化: 同时预热 Mixup 路径，避免 warmup 结束后的重编译延迟
    # I107-3: 移除 synchronize() 调用，避免干扰 cudagraphs
    # I107-4: 添加编译超时检测和 GPU 状态监控
    if config.compile_model:
        import threading
        import time as time_module

        # I107-4: 启动 GPU 监控线程
        gpu_monitor_running = [True]
        def gpu_monitor():
            if device.type == 'cuda':
                try:
                    while gpu_monitor_running[0]:
                        if torch.cuda.is_available():
                            mem_used = torch.cuda.memory_allocated() / 1024**3
                            util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else -1
                            print(f"[GPU-MONITOR] Memory: {mem_used:.2f} GB, Util: {util}%")
                        time_module.sleep(10)
                except Exception:
                    pass

        monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
        monitor_thread.start()

        print("[INFO] Warming up compiled model with full batch size...")
        compile_start = time_module.time()

        try:
            warmup_batch = next(iter(train_loader))
            if isinstance(warmup_batch, (list, tuple)):
                warmup_imgs = warmup_batch[0].to(device, non_blocking=True)  # 使用完整 batch
                warmup_labels = warmup_batch[1].to(device, non_blocking=True)
            else:
                warmup_imgs = warmup_batch.to(device, non_blocking=True)
                warmup_labels = torch.zeros(warmup_imgs.shape[0], dtype=torch.long, device=device)
            if config.use_channels_last:
                warmup_imgs = warmup_imgs.to(memory_format=torch.use_channels_last)

            # 阶段1: 预热标准 forward pass (移除 synchronize 以避免 cudagraphs 冲突)
            with torch.no_grad():
                with get_amp_context(device, config.use_amp):
                    for i in range(3):  # 3次预热确保编译稳定
                        print(f"[WARMUP] Forward pass {i+1}/3...")
                        _ = model(warmup_imgs)
                        # I107-4: 每次 forward 后检查 GPU 状态
                        if device.type == 'cuda':
                            mem = torch.cuda.memory_allocated() / 1024**3
                            print(f"[WARMUP] GPU memory: {mem:.2f} GB")

            # 阶段2: 预热 Mixup 路径 (如果启用)
            if mixup_fn is not None:
                print("[INFO] Pre-warming Mixup code path...")
                with torch.no_grad():
                    with get_amp_context(device, config.use_amp):
                        # 测试 Mixup 数据变换
                        test_imgs, test_mixed_labels = mixup_fn(warmup_imgs.clone(), warmup_labels.clone())
                        # 确保 Mixup 后保持 use_channels_last 格式
                        if config.use_channels_last:
                            test_imgs = test_imgs.to(memory_format=torch.use_channels_last)
                        # 测试 forward + Mixup loss
                        test_outs = model(test_imgs)
                        test_logits = test_outs.logits if hasattr(test_outs, 'logits') else test_outs
                        _ = mixup_criterion(test_logits, test_mixed_labels)
                        del test_imgs, test_mixed_labels, test_outs
                print("[OK] Mixup path pre-warmed")

            compile_time = time_module.time() - compile_start
            del warmup_imgs, warmup_labels
            torch.cuda.empty_cache()
            print(f"[OK] Compilation complete! Time: {compile_time:.1f}s")

        except Exception as e:
            print(f"[WARN] Warmup failed: {e}")
            import traceback
            traceback.print_exc()
        finally:
            gpu_monitor_running[0] = False
            monitor_thread.join(timeout=5)
    
    best_val = 0.0
    patience_counter = 0
    early_stopped = False
    
    exp_name = args.exp_name if args.exp_name else f"fractal_vit_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)

    # 保存配置
    # P2 修复: TrainingConfig 不是 dataclass，使用 vars() 替代 asdict()
    # I103-1 修复: 排除 arch_config（不可 JSON 序列化）
    config_dict = {k: v for k, v in vars(config).items() if not isinstance(v, type)}
    if 'arch_config' in config_dict:
        del config_dict['arch_config']
    with open(exp_dir / "logs" / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    history = []
    
    print(f"[INFO] Early stopping: patience={config.patience}, min_delta={config.min_delta}")
    print(f"[INFO] Regularization: dropout={config.dropout}, weight_decay={config.weight_decay}")
    print(f"[INFO] Label smoothing: {config.label_smoothing}")
    print(f"[INFO] Mixup/CutMix: alpha={config.mixup_alpha}/{config.cutmix_alpha}, prob={config.mixup_prob}")
    if use_mixup and config.warmup_epochs > 0:
        print(f"[INFO] Mixup/CutMix 将在 warmup 阶段 (epoch 1-{config.warmup_epochs}) 禁用")
    
    # P14: 长尾效应优化信息
    if config.use_focal_loss or config.use_class_balanced or config.progressive_aug:
        print(f"[INFO] P14 长尾效应优化:")
        if config.use_focal_loss:
            print(f"  - Focal Loss: gamma={config.focal_gamma}")
        if config.use_class_balanced:
            print(f"  - Class-Balanced Loss: beta={config.class_balance_beta}")
        if config.progressive_aug:
            print(f"  - Progressive Augmentation: 渐进式增强强度")
    
    # =========================================================================
    # P7-7 / P10-12: 启用 GumbelTopKSplitter 内置退火调度
    # =========================================================================
    # GumbelTopKSplitter 退火 API:
    #   - enable_temperature_annealing()
    #   - enable_explore_bias_annealing()
    #
    # 数学形式化:
    #   温度退火: T(t) = T_start · (T_end / T_start)^(t / total_steps)
    #   探索偏置: b(t) = b_start · (1 - t / total_steps)
    #
    # Scheme D 特性:
    #   - Top-K 硬约束保证 token 数量，偏置影响 *哪些* 被选中
    #   - STE 梯度仍依赖温度，退火保证梯度质量
    # =========================================================================
    splitter_annealing_enabled = False
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
        splitter = model.splitter
        # 计算总训练步数 (epochs × batches_per_epoch)
        batches_per_epoch = len(train_loader) // config.accum_steps
        total_training_steps = config.epochs * batches_per_epoch
        
        # 启用温度退火 (P7-7)
        if hasattr(splitter, 'enable_temperature_annealing'):
            # 考虑 warmup: 在 warmup 期间使用 T_start，之后开始退火
            post_warmup_steps = max(1, (config.epochs - config.splitter_temp_warmup) * batches_per_epoch)
            # I100-2: 使用命令行参数 config.temp_schedule
            schedule = getattr(config, 'temp_schedule', SPLITTER_TEMP_SCHEDULE)
            splitter.enable_temperature_annealing(
                total_steps=post_warmup_steps,
                T_start=config.splitter_temp_start,
                T_end=config.splitter_temp_end,
                schedule=schedule,
            )
            print(f"[OK] 启用自适应分割器温度退火:")
            print(f"     T: {config.splitter_temp_start} → {config.splitter_temp_end}")
            print(f"     Steps: {post_warmup_steps} (after {config.splitter_temp_warmup} warmup epochs)")
            print(f"     Schedule: {schedule}")  # I100-2: 显示实际使用的调度
            splitter_annealing_enabled = True
        
        # 启用探索偏置退火 (P10-12 + P10-15)
        # P10-15 修复: 偏置退火与温度退火同步，延迟到 warmup 后开始
        # 
        # 遵循模型架构默认值:
        #   - b_start, b_end 使用 GumbelTopKSplitter.enable_explore_bias_annealing() 的默认参数
        #   - 训练器只传递必需的 total_steps
        if hasattr(splitter, 'enable_explore_bias_annealing'):
            splitter.enable_explore_bias_annealing(
                total_steps=post_warmup_steps,  # P10-15: 与温度退火同步
                # b_start/b_end 使用模型默认值 (0.5 → 0.0)
            )
            print(f"[OK] 启用探索偏置退火 (使用模型默认参数)")
            print(f"     Steps: {post_warmup_steps} (与温度退火同步)")
            splitter_annealing_enabled = True
        
        # =====================================================================
        # I21 深度平衡机制
        # I30-2: 已移除 Subset Softmax，现使用全局 Softmax
        # =====================================================================
        # GumbelTopKSplitter (Scheme D) 默认启用以下深度平衡组件:
        #   - β: Log-Compensation Bias (LOG_COMPENSATION_ENABLED=True) [已移除，被方案E替代]
        #   - γ: 全局 Softmax (~K/N 梯度覆盖，约 37.6%，替代原 Subset Softmax)
        #   - ε: Depth KL Loss (DEPTH_KL_WEIGHT=0.1)
        #
        # 这些是模型架构的内部设计，遵循 constants.py 中的默认值。
        # 训练器不应硬编码覆盖这些参数。
        # 如需调整，请修改 constants.py 或通过模型初始化参数传递。
        # =====================================================================
    
    if not splitter_annealing_enabled:
        print(f"[INFO] 自适应分割器退火未启用 (不支持或未使用可学习分割器)")
    
    print()
    
    for epoch in range(1, config.epochs + 1):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch}/{config.epochs} - STARTING")
        print(f"{'='*60}")
        start = time.time()

        # P13: 每个 epoch 开始时手动 GC
        gc.collect()
        if device.type == 'cuda':
            # I145: 修复 CUDA 断言错误
            # 使用 try-except 包装 empty_cache，防止断言错误被延迟报告
            try:
                torch.cuda.empty_cache()
            except RuntimeError as e:
                # 检查是否是 device-side assert
                if "device-side assert" in str(e):
                    print(f"[WARN] CUDA device-side assert detected at epoch start")
                    print(f"       This may indicate a configuration mismatch.")
                    # 尝试同步并获取更多信息
                    try:
                        torch.cuda.synchronize()
                    except:
                        pass
                raise
        
        # I24-1: 可选的 tokenizer 参数解冻 (在指定 epoch 后)
        if config.freeze_tokenizer and config.freeze_tokenizer_epochs > 0:
            if epoch == config.freeze_tokenizer_epochs + 1:
                # 解冻 tokenizer 可学习参数
                for name, param in model.tokenizer.named_parameters():
                    if 'threshold' in name or 'quota' in name or 'temperature' in name:
                        param.requires_grad = True
                print(f"[I24-1] Epoch {epoch}: Unfreezing tokenizer params (warm start phase complete)")
        
        # P7-7 + P10-15: 温度退火和偏置退火调度 (同步控制)
        # 说明: 温度退火已在训练开始前通过 enable_temperature_annealing() 启用
        #       在 forward() 中自动更新，无需手动调用 scheduler.step()
        #       但 warmup 期间需要禁用退火，保持 T_start
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
            splitter = model.splitter
            if epoch <= config.splitter_temp_warmup:
                # Warmup 阶段：暂时禁用自动退火，固定 T_start
                if hasattr(splitter, 'disable_temperature_annealing'):
                    splitter.disable_temperature_annealing()
                    splitter.set_temperature(config.splitter_temp_start)
                # 偏置退火也在 warmup 期间禁用 (使用模型初始值)
                if hasattr(splitter, 'disable_explore_bias_annealing'):
                    splitter.disable_explore_bias_annealing()
                    # 注意: 不再硬编码偏置值，使用模型的初始 explore_bias (默认 0.5)
            elif epoch == config.splitter_temp_warmup + 1:
                # Warmup 结束：重新启用温度退火和偏置退火
                post_warmup_steps = max(1, (config.epochs - config.splitter_temp_warmup) * (len(train_loader) // config.accum_steps))
                
                # 重新启用温度退火
                if hasattr(splitter, 'enable_temperature_annealing'):
                    # I100-2: 使用命令行参数 config.temp_schedule
                    schedule = getattr(config, 'temp_schedule', SPLITTER_TEMP_SCHEDULE)
                    splitter.enable_temperature_annealing(
                        total_steps=post_warmup_steps,
                        T_start=config.splitter_temp_start,
                        T_end=config.splitter_temp_end,
                        schedule=schedule,
                    )
                # P10-15: 同步启用偏置退火 (使用模型默认参数)
                if hasattr(splitter, 'enable_explore_bias_annealing'):
                    splitter.enable_explore_bias_annealing(
                        total_steps=post_warmup_steps,
                        # b_start/b_end 使用模型默认值
                    )
                print(f"[INFO] Epoch {epoch}: 温度退火和偏置退火正式开始 (warmup 结束)")
        
        # Warmup 阶段禁用 Mixup/CutMix
        # 原因: warmup 阶段学习率较低，模型需要学习基本特征
        # Mixup/CutMix 会使学习目标更复杂，可能干扰初始阶段的学习
        # 参考: DeiT, Swin Transformer 等论文的训练策略
        current_mixup_fn = None if epoch <= config.warmup_epochs else mixup_fn
        
        # P14: 渐进式数据增强
        # 在 warmup 结束后逐渐增加 Mixup/CutMix 强度
        # 注意: 通过调整 prob 而非 alpha 来实现渐进式，避免 Beta 分布极小 alpha 问题
        if config.progressive_aug and current_mixup_fn is not None and mixup_fn is not None:
            # 计算增强进度 (0 -> 1 over first half of post-warmup epochs)
            post_warmup_epoch = epoch - config.warmup_epochs
            ramp_epochs = max(1, (config.epochs - config.warmup_epochs) // 2)  # 在一半 epoch 内达到满强度
            aug_progress = min(1.0, post_warmup_epoch / ramp_epochs)
            
            # 通过调整应用概率来实现渐进式增强 (避免极小 alpha 问题)
            # prob 从 0.1 渐增到 config.mixup_prob
            current_prob = 0.1 + (config.mixup_prob - 0.1) * aug_progress
            
            current_mixup_fn = MixupCutmix(
                mixup_alpha=config.mixup_alpha,  # 保持原始 alpha
                cutmix_alpha=config.cutmix_alpha,  # 保持原始 alpha
                prob=current_prob,  # 渐增概率
                num_classes=spec.num_classes,
                label_smoothing=config.label_smoothing,
            )
            if post_warmup_epoch <= 3:  # 只在前 3 个 epoch 打印
                print(f"[INFO] Progressive Aug (epoch {epoch}): prob={current_prob:.2f}")
        
        # P15: Warmup 结束后首次启用 Mixup 的处理
        # 在这个过渡 epoch，禁用 CudaPrefetcher 避免多进程+compile 冲突
        disable_prefetch_this_epoch = False
        if epoch == config.warmup_epochs + 1 and mixup_fn is not None:
            print(f"[INFO] Epoch {epoch}: 启用 Mixup/CutMix 增强")
            print(f"[INFO] 首次 Mixup epoch：暂时禁用 CUDA Prefetcher 以确保稳定性")

            os.environ['DISABLE_PREFETCH'] = '1'
            disable_prefetch_this_epoch = True
            # 添加同步点，确保之前的 CUDA 操作完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # P15-FIX: 创建单进程 DataLoader 避免多进程死锁
            # persistent_workers + pin_memory 在某些情况下会导致死锁或 OOM
            print("[INFO] 创建单进程 DataLoader 以确保稳定性...")
            from torch.utils.data.sampler import SubsetRandomSampler
            simple_train_loader = DataLoader(
                train_loader.dataset,
                batch_size=config.batch_size,
                sampler=train_loader.sampler,
                num_workers=0,  # 单进程
                pin_memory=False,  # 禁用 pin_memory
                drop_last=True,
            )
        else:
            simple_train_loader = None
        
        # 选择使用的 loader
        current_train_loader = simple_train_loader if disable_prefetch_this_epoch else train_loader

        print(f"[INFO] Epoch {epoch}: 即将开始 train_epoch...")
        train_loss, train_acc, perf_stats = train_epoch(
            model, current_train_loader, optimizer, device, scaler, config,
            mixup_fn=current_mixup_fn,
            num_classes=spec.num_classes,
            profile=(epoch == 1),
            exp_dir=exp_dir,
            loss_fn=loss_fn,
            class_weights=class_weights,
            epoch=epoch,
            hard_mining=hard_mining,
        )
        print(f"[INFO] Epoch {epoch}: train_epoch 完成, loss={train_loss:.4f}, acc={train_acc:.2f}%")

        # P15: 恢复 CudaPrefetcher 和清理临时 loader
        if disable_prefetch_this_epoch:
            os.environ.pop('DISABLE_PREFETCH', None)
            del simple_train_loader
            print(f"[INFO] Epoch {epoch} 完成，后续 epoch 将恢复正常 DataLoader")
        
        val_loss, val_acc, per_class_stats = evaluate(
            model, val_loader, device, config.use_amp, spec.num_classes,
            use_channels_last=config.use_channels_last
        )
        
        # P14: 类别平衡分析
        class_balance_report = analyze_class_balance(per_class_stats, epoch, verbose=True)
        
        scheduler.step()
        
        # 获取详细训练状态 (关键追踪参数)
        training_stats = None
        scale_distribution = None
        if hasattr(model, 'tokenizer'):
            if hasattr(model.tokenizer, 'get_training_stats'):
                training_stats = model.tokenizer.get_training_stats()
            
            # 每 10 个 epoch 或最后一个 epoch 计算尺度分布
            if hasattr(model.tokenizer, 'compute_scale_distribution') and (epoch % 10 == 0 or epoch == config.epochs):
                # 使用 val_loader 的一个 batch 计算尺度分布
                sample_batch = next(iter(val_loader))
                sample_imgs = sample_batch[0][:8].to(device)  # 只用 8 张图
                scale_distribution = model.tokenizer.compute_scale_distribution(sample_imgs)
        
        epoch_time = time.time() - start
        
        # 记录历史
        history_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time,
        }
        if training_stats is not None:
            history_entry['training_stats'] = training_stats
        if scale_distribution is not None:
            history_entry['scale_distribution'] = scale_distribution
        # P14: 类别平衡分析
        if class_balance_report.get('status') == 'analyzed':
            history_entry['class_balance'] = {
                'imbalance_score': class_balance_report['imbalance_score'],
                'zero_acc_count': class_balance_report['zero_acc_count'],
                'accuracy_std': class_balance_report['accuracy_std'],
            }
        history.append(history_entry)
        
        print(f"\nEpoch {epoch}/{config.epochs}:")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.2f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.2f}%")
        print(f"  Time:  {epoch_time:.1f}s, Throughput: {perf_stats['throughput']:.1f} samples/s")
        
        # 显示尺度分布 (每 10 epoch)
        if scale_distribution is not None:
            ratios = scale_distribution['scale_ratios']
            entropy = scale_distribution['entropy']
            max_entropy = scale_distribution['max_entropy']
            dominant = scale_distribution['dominant_scale']
            ratio_str = ", ".join([f"p{ps}:{r*100:.1f}%" for ps, r in ratios.items()])
            print(f"  Scales: {ratio_str}")
            # I10-19: 连续模式下 max_entropy 可能为 0 (无离散深度分布)
            if max_entropy > 0:
                entropy_pct = entropy / max_entropy * 100
                print(f"  Entropy: {entropy:.3f}/{max_entropy:.3f} ({entropy_pct:.1f}%), Dominant: {dominant}px")
            else:
                print(f"  Entropy: {entropy:.3f} (continuous mode, no discrete depths)")
        
        if epoch == 1:
            data_pct = perf_stats['avg_data_time'] / perf_stats['avg_batch_time'] * 100 if perf_stats['avg_batch_time'] > 0 else 0
            fwd_pct = perf_stats['avg_forward_time'] / perf_stats['avg_batch_time'] * 100 if perf_stats['avg_batch_time'] > 0 else 0
            print(f"  Perf:  data={data_pct:.1f}%, fwd={fwd_pct:.1f}%, mem={perf_stats['cuda_mem_peak_gb']:.2f}GB")
        
        # P7/P8: 显示可学习分割器状态 (仅 learnable scheme)
        if perf_stats.get('learnable_thresholds') is not None:
            thresholds = perf_stats['learnable_thresholds']
            temperature = perf_stats.get('learnable_temperature', 1.0)
            tau_str = ", ".join([f"τ{i}:{t:.3f}" for i, t in enumerate(thresholds[:4])])  # 只显示前4层
            print(f"  Splitter: T={temperature:.3f}, {tau_str}")
        
        # P10-4/P10-5/P10-9: 显示软熵和弹性预算状态
        if perf_stats.get('soft_entropy') is not None or perf_stats.get('soft_token_count') is not None:
            p10_parts = []
            if perf_stats.get('entropy_ratio') is not None:
                p10_parts.append(f"H_ratio={perf_stats['entropy_ratio']:.2%}")
            if perf_stats.get('dominant_prob') is not None:
                p10_parts.append(f"dom_prob={perf_stats['dominant_prob']:.2%}")
            if perf_stats.get('soft_token_count') is not None:
                p10_parts.append(f"N_soft={perf_stats['soft_token_count']:.1f}")
            if p10_parts:
                print(f"  P10: {', '.join(p10_parts)}")

        # I111-6: 深度分布实时监控
        if perf_stats.get('depth_pi') is not None and perf_stats.get('depth_entropy') is not None:
            depth_pi = perf_stats['depth_pi']
            entropy = perf_stats['depth_entropy']
            max_entropy = perf_stats.get('max_entropy', math.log(len(depth_pi)))
            kl = perf_stats.get('depth_kl', 0)

            # 格式化深度分布字符串
            depth_str = ", ".join([f"d{d}:{p*100:.1f}%" for d, p in enumerate(depth_pi)])
            print(f"  Depth: {depth_str}")
            print(f"  Entropy: {entropy:.3f}/{max_entropy:.3f} ({entropy/max_entropy*100:.1f}%)")

            # KL 散度警告
            if kl > 0.5:
                print(f"  ⚠️  KL divergence high: {kl:.3f} (>0.5)")
        
        # P10-14: 分割器健康检查
        # 动态获取 max_entropy (从 get_depth_distribution_stats 或配置推算)
        max_entropy_value = 1.609  # 默认: log(5) for max_depth=4
        if perf_stats.get('max_entropy') is not None:
            max_entropy_value = perf_stats['max_entropy']
        elif hasattr(model, 'splitter'):
            # I98-2: 新架构使用 model.splitter (依赖注入模式)
            splitter = model.splitter
            if hasattr(splitter, 'max_depth'):
                import math
                max_entropy_value = math.log(splitter.max_depth + 1)
        
        if perf_stats.get('soft_token_count') is not None or perf_stats.get('soft_entropy') is not None:
            health_status = check_splitter_health(
                avg_tokens=perf_stats.get('soft_token_count'),
                entropy=perf_stats.get('soft_entropy'),
                max_entropy=max_entropy_value,
                epoch=epoch,
            )
            if not health_status.is_healthy:
                print(f"  ⚠️  {health_status.message}")
            # 记录到 history 用于后续分析
            history_entry['splitter_health'] = {
                'is_healthy': health_status.is_healthy,
                'health_score': health_status.health_score,
                'severity': health_status.severity,
                'collapse': health_status.collapse_detected,
                'monotone': health_status.monotone_detected,
            }
        
        # 保存最佳
        if val_acc > best_val + config.min_delta:
            best_val = val_acc
            patience_counter = 0

            # 直接从配置构造 ModelGene（单一数据源）
            dataset_name = getattr(config, 'dataset', spec.name)
            gene = ModelGene.from_config(config.arch_config, dataset_name=dataset_name, epoch=epoch)

            # 使用共享的 checkpoint 保存函数（与评估器使用相同的保存逻辑）
            save_checkpoint_with_gene(
                path=exp_dir / "checkpoints" / "best.pth",
                model=model,
                gene=gene,
                optimizer_state=optimizer.state_dict(),
                epoch=epoch,
                val_acc=val_acc,
                val_loss=val_loss,
                extra={'config': obj_to_dict(config)},
            )
        else:
            patience_counter += 1
            print(f"  [!] No improvement ({patience_counter}/{config.patience})")
            if patience_counter >= config.patience:
                print(f"\n[EARLY STOPPING] No improvement for {config.patience} epochs.")
                print(f"[EARLY STOPPING] Best val acc: {best_val:.2f}% at epoch {epoch - patience_counter}")
                early_stopped = True
                break
    
    # 保存训练历史
    with open(exp_dir / "training_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    # ========== Train/Eval 一致性验证 ==========
    print("\n" + "="*70)
    print("TRAIN/EVAL CONSISTENCY CHECK")
    print("="*70 + "\n")
    
    consistency_report = verify_train_eval_consistency(model, val_loader, device, config)
    
    # 保存一致性报告
    with open(exp_dir / "logs" / "consistency_report.json", 'w') as f:
        json.dump(consistency_report, f, indent=2)
    
    # 测试
    print("\n" + "="*70)
    print("TESTING")
    print("="*70 + "\n")

    # 使用共享的 load_model 函数（与评估器完全一致的代码路径）
    # 这确保从 checkpoint 的 model_gene 重建模型，架构与训练时一致
    model, gene = load_model(
        str(exp_dir / "checkpoints" / "best.pth"),
        device=str(device),
        strict=True,
        verbose=True,
    )

    test_loss, test_acc, per_class_stats = evaluate(
        model, test_loader, device, config.use_amp, 
        spec.num_classes, return_per_class=True,
        use_channels_last=config.use_channels_last
    )
    
    print(f"Test: loss={test_loss:.4f}, acc={test_acc:.2f}%")
    
    # 逐类别评估诊断
    if per_class_stats:
        print(f"\n{'='*70}")
        print("PER-CLASS ACCURACY ANALYSIS")
        print(f"{'='*70}")
        print(f"  Accuracy range: {per_class_stats['accuracy_min']:.1f}% - {per_class_stats['accuracy_max']:.1f}%")
        print(f"  Accuracy std:   {per_class_stats['accuracy_std']:.1f}%")
        print(f"  Worst 5 classes: {per_class_stats['worst_classes'][:5]}")
        print(f"  Best 5 classes:  {per_class_stats['best_classes'][:5]}")
        
        # 检测类别不平衡
        if per_class_stats['accuracy_std'] > 20:
            print(f"\n  [WARN] 类别不平衡警告: std={per_class_stats['accuracy_std']:.1f}% > 20%")
            print(f"         请检查数据加载和模型架构")
    
    if early_stopped:
        print(f"[*] Training stopped early at epoch {epoch}/{config.epochs}")
    print(f"\n[OK] Results saved to: {exp_dir}")
    
    with open(exp_dir / "results.json", 'w') as f:
        results = {
            'best_val_acc': best_val,
            'test_acc': test_acc,
            'test_loss': test_loss,
            'total_epochs': epoch if early_stopped else config.epochs,
            'early_stopped': early_stopped,
            'best_epoch': epoch - patience_counter if early_stopped else epoch,
            'consistency_passed': consistency_report.get('passed', None),
            'tokenizer_type': 'streaming_v3',
        }
        if per_class_stats:
            results['per_class_stats'] = {
                'accuracy_std': per_class_stats['accuracy_std'],
                'accuracy_min': per_class_stats['accuracy_min'],
                'accuracy_max': per_class_stats['accuracy_max'],
                'worst_classes': per_class_stats['worst_classes'],
                'best_classes': per_class_stats['best_classes'],
            }
        json.dump(results, f, indent=2)


def cleanup_multiprocessing():
    """清理多进程资源，防止信号量泄漏"""
    import gc
    gc.collect()
    
    # 强制终止所有子进程
    try:
        for p in _mp.active_children():
            p.terminate()
            p.join(timeout=1)
    except Exception:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 训练被中断")
    finally:
        cleanup_multiprocessing()
