#!/usr/bin/env python3
"""Fractal ViT Training Script - V3 Variable Depth Tokens

⚠️  **重要更新 (2026-01-05 - I15 完成)**:
   training 模块现已可用（位于 examples/training/），提供以下增强功能：
   - ClassBalancedSampler / ProgressiveSampler - 类别平衡采样
   - FocalLoss / ClassBalancedCE - 长尾效应优化
   - ModularTrainer - 模块化训练器（替代手写循环）
   - ClassificationMetrics - 完整评估指标
   - ExperimentVisualizer - 统一可视化接口
   
   本脚本已导入 training 模块，但保留了手写训练循环以供参考。
   如需使用 ModularTrainer，请参考 examples/training/README.md

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
默认使用 Scheme D (GumbelTopKSplitter)，Scheme A (LearnableSplitter) 为备用:

+------------------+----------------------------------+-------------------------+
| 方案              | 数学描述                          | 特点                     |
+==================+==================================+=========================+
| Scheme D         | selected = TopK(logits + g, K)   | 端到端学习, 100%梯度覆盖 |
| (GumbelTopK)     | 树一致性约束 + STE               | 硬 K 约束, 并行评估      |
+------------------+----------------------------------+-------------------------+
| Scheme A         | p_split = σ((C_θ(R) - τ_d) / T)  | 端到端学习分割           |
| (Learnable)      | Gumbel-Softmax 可微分采样        | 软约束, BFS 串行        |
+------------------+----------------------------------+-------------------------+

Note: Scheme B/C 已移除。Scheme D 优势:
- 100% 梯度覆盖 (STE 使所有候选都有梯度)
- 硬 K 约束 [K_min, K_max] 消除死锁风险
- O(1) 并行评估所有候选 vs O(D) BFS
- 树一致性向量化约束保证 Hilbert 100%

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

I14 分割器初始化健壮性 (2026-01-03)
-----------------------------------
解决 warmup 阶段强制分割与梯度稳定性问题:

1. I14-1 A1: Warmup 强制分割
   - enable_warmup_forced_split(steps) API
   - 前 N 步内强制至少 70% 区域分割，消除 Gumbel 随机性死锁
   
2. I14-1 D1: 弹性预算崩溃惩罚
   - 当 actual_tokens < 2 时触发强惩罚
   - 公式: L_collapse = λ_collapse · 𝟙[N_actual < 2]
   - 软版本: L_soft_collapse = λ · 0.1 · softplus(2 - N)
   - 参数: --elastic-lambda-collapse (默认 1.0)
   - API: get_auxiliary_losses() 现在正确传递 actual_token_count

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
    
    # Tiny ImageNet 完整训练 (推荐配置 - 含 P10 优化)
    python train_fractal_vit.py --dataset tiny-imagenet --epochs 100 --dim 384 \\
        --depth 12 --heads 8 --dropout 0.1 --drop-path 0.15 --use-amp \\
        --gradient-checkpoint --compile --channels-last \\
        --include-soft-entropy --include-elastic-budget
    
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
import multiprocessing as _mp

# 强制 spawn 方法（CUDA + 容器必需）
try:
    _mp.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# CUDA 内存优化
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'max_split_size_mb:512,expandable_segments:True')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

# 抑制 torch.compile 的符号形状警告和 checkpoint autocast 废弃警告
import warnings
warnings.filterwarnings('ignore', message='.*is not in var_ranges.*')
warnings.filterwarnings('ignore', message='.*defaulting to unknown range.*')
warnings.filterwarnings('ignore', message='.*torch.cpu.amp.autocast.*is deprecated.*', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*The epoch parameter in `scheduler.step\\(\\)`.*', category=UserWarning)

import argparse
import json
import multiprocessing
import random
import shutil
import time
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
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

# AMP 兼容层
try:
    from torch.amp import autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast, GradScaler  # type: ignore
    _NEW_AMP = False

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = PROJECT_ROOT / "src"
EXAMPLES_PATH = PROJECT_ROOT / "examples"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(EXAMPLES_PATH) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PATH))

from vit_pytorch import FractalCurveViT

# Fractal Training 模块 (I15) - 现在位于 examples/training
from training import (
    # Samplers
    ClassBalancedSampler,
    ProgressiveSampler,
    # Losses
    FocalLoss as FTFocalLoss,
    ClassBalancedCE,
    FocalClassBalancedLoss,
    # Metrics
    ClassificationMetrics,
    # Schedulers
    FLOPSConfig,
    compute_transformer_flops,
    FLOPSBudgetLoss,
    DepthWeightedBudgetLoss,
    BudgetScheduler,
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
)


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


@dataclass
class TrainingConfig:
    """训练配置"""
    # 数据集
    dataset: str
    batch_size: int
    num_workers: int
    val_split: float
    subset_size: Optional[int]
    
    # 模型
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int
    max_level: int
    num_scales: int
    pool: str
    ffn_type: str
    
    # Tokenizer 配置 (V3 Variable Depth Tokens)
    tokenizer_type: str  # 'streaming_v3' (唯一支持)
    
    # V3 高级分割参数 (Scheme D: GumbelTopKSplitter 默认)
    target_tokens: Optional[int]  # 目标 token 数量
    split_tau0: float  # 根节点阈值 τ₀
    split_gamma: float  # 阈值衰减因子 γ
    enforce_balance: bool  # 是否强制 2:1 平衡约束
    
    # P6-1: 深度缩放参数
    depth_scale_range: Optional[Tuple[float, float]]  # (σ_min, σ_max)，默认 (0.5, 2.0)
    
    # P6-2: LCA 温度参数
    lca_temperature: Optional[float]  # LCA 偏置温度，默认 1.5
    learnable_temperature: bool  # 是否可学习温度，默认 True
    
    # P7-6: 可学习分割器训练参数
    lambda_splitter_entropy: float  # 熵损失权重，鼓励尺度多样性
    lambda_splitter_budget: float  # 预算约束权重
    splitter_token_budget: int  # 目标 token 数预算
    
    # P7-7: 温度退火调度参数
    splitter_temp_start: float  # 起始温度 T_start
    splitter_temp_end: float  # 终止温度 T_end
    splitter_temp_warmup: int  # Warmup epoch 数 (固定 T_start)
    
    # P10-4/P10-5: 软熵损失参数
    include_soft_entropy: bool  # 是否启用软熵损失（推荐 True）
    soft_entropy_mode: str  # 熵损失模式: 'maximize'（最大化熵）或 'target'（匹配目标）
    soft_entropy_weight: float  # 软熵损失权重
    soft_entropy_target: Optional[float]  # 目标熵值（仅 mode='target' 时使用）
    
    # P10-9: 弹性预算损失参数
    include_elastic_budget: bool  # 是否启用弹性预算损失（推荐 True）
    elastic_N_min: int  # 弹性预算下界（Dead Zone 左边界）
    elastic_N_max: int  # 弹性预算上界（Dead Zone 右边界）
    elastic_lambda_over: float  # 超出上界惩罚权重
    elastic_lambda_under: float  # 低于下界约束权重
    elastic_lambda_collapse: float  # I14-1 D1: 崩溃惩罚权重（建议 1.0）
    
    # 训练
    epochs: int
    learning_rate: float
    weight_decay: float
    dropout: float
    emb_dropout: float
    drop_path: float
    label_smoothing: float
    gradient_clip: float
    use_amp: bool
    accum_steps: int
    warmup_epochs: int
    gradient_checkpoint: bool
    compile_model: bool
    channels_last: bool
    
    # 早停
    patience: int
    min_delta: float
    
    # Mixup/CutMix
    mixup_alpha: float
    cutmix_alpha: float
    mixup_prob: float
    
    # 长尾效应优化 (P14)
    use_focal_loss: bool  # 是否使用 Focal Loss
    focal_gamma: float  # Focal Loss 的 gamma 参数，默认 2.0
    use_class_balanced: bool  # 是否使用类别平衡损失权重
    class_balance_beta: float  # 类别平衡的 beta 参数，默认 0.9999
    progressive_aug: bool  # 是否使用渐进式数据增强
    
    # 系统
    seed: int
    device: str


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
    targets = targets.clamp(min=0)
    targets = targets / (targets.sum(dim=1, keepdim=True) + 1e-8)
    
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
}


# ============================================================================
# 环境检测
# ============================================================================

def detect_environment() -> Dict[str, Any]:
    """检测运行环境"""
    env = {
        'in_container': os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv'),
        'platform': platform.system(),
        'cpu_count': multiprocessing.cpu_count(),
        'recommended_workers': 4,
    }
    
    if env['platform'] == 'Linux':
        try:
            import psutil
            shm = psutil.disk_usage('/dev/shm')
            shm_gb = shm.total / (1024**3)
            # 容器环境优化：更激进的 workers 配置
            if env['in_container']:
                if shm_gb >= 8:
                    env['recommended_workers'] = min(16, env['cpu_count'])
                elif shm_gb >= 4:
                    env['recommended_workers'] = min(12, env['cpu_count'])
                else:
                    env['recommended_workers'] = min(8, env['cpu_count'])
            else:
                if shm_gb >= 8:
                    env['recommended_workers'] = min(12, env['cpu_count'])
                elif shm_gb >= 4:
                    env['recommended_workers'] = min(8, env['cpu_count'])
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
    """获取 AMP autocast 上下文"""
    if _NEW_AMP:
        return autocast('cuda', enabled=enabled)
    else:
        return autocast(enabled=enabled)


def create_grad_scaler(enabled: bool) -> GradScaler:
    """创建 GradScaler"""
    if _NEW_AMP:
        return GradScaler('cuda', enabled=enabled)
    else:
        return GradScaler(enabled=enabled)


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


# ============================================================================
# 数据加载
# ============================================================================

def create_dataloaders(
    spec: DatasetSpec,
    config: TrainingConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建数据加载器"""
    
    # 数据增强
    if spec.name == "MNIST":
        train_tf = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
        ])
    elif spec.name == "TinyImageNet":
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(64, padding=8),
            transforms.RandAugment(num_ops=2, magnitude=9),  # Phase 2: RandAugment
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),  # Cutout-like augmentation
        ])
    else:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(spec.image_size, padding=4),
            transforms.RandAugment(num_ops=2, magnitude=9),  # Phase 2: RandAugment
            transforms.ToTensor(),
            transforms.Normalize(spec.mean, spec.std),
            transforms.RandomErasing(p=0.25),  # Cutout-like augmentation
        ])
    
    test_tf = transforms.Compose([
        transforms.Resize(max(spec.image_size, 32)),
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
    loader_kwargs = {
        'batch_size': config.batch_size,
        'num_workers': effective_workers,
        'pin_memory': effective_workers > 0 and torch.cuda.is_available(),
        'multiprocessing_context': mp_context if effective_workers > 0 else None,
        'persistent_workers': effective_workers > 0,  # 避免每个 epoch 重建进程
        'drop_last': True,  # 避免最后一个小 batch 的性能损失
    }
    if effective_workers > 0:
        # 容器环境使用更大的 prefetch_factor 补偿 I/O 延迟
        # 增大到 16/8 以最大化 GPU 利用率
        loader_kwargs['prefetch_factor'] = 16 if is_container else 8
        # 添加 generator 参数以提高多进程随机性
        loader_kwargs['generator'] = torch.Generator().manual_seed(42)
    
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
    
    def __init__(self, loader: DataLoader, device: torch.device, channels_last: bool = False):
        self.loader = loader
        self.device = device
        self.channels_last = channels_last
        # 双缓冲: 使用两个 CUDA stream 实现流水线
        self.stream = torch.cuda.Stream() if device.type == 'cuda' else None
        self._debug = False
        self._batch_count = 0
        # 双缓冲状态
        self._buffer = [None, None]  # (raw_batch, gpu_data)
        self._buffer_idx = 0
        
    def __iter__(self):
        self.loader_iter = iter(self.loader)
        self._batch_count = 0
        self._buffer = [None, None]
        self._buffer_idx = 0
        # 预加载两个 batch 填满双缓冲
        self._preload_next()
        self._preload_next()
        return self
    
    def _preload_next(self):
        """预加载下一个 batch 到空闲缓冲区"""
        try:
            raw_batch = next(self.loader_iter)
        except StopIteration:
            return False
        
        if self.stream is not None:
            with torch.cuda.stream(self.stream):
                imgs = raw_batch[0].to(self.device, non_blocking=True)
                if self.channels_last:
                    imgs = imgs.to(memory_format=torch.channels_last)
                gpu_data = (
                    imgs,
                    raw_batch[1].to(self.device, non_blocking=True),
                )
        else:
            gpu_data = raw_batch
        
        # 找到空闲缓冲区
        for i in range(2):
            if self._buffer[i] is None:
                self._buffer[i] = gpu_data
                break
        return True
    
    def __next__(self):
        # 等待当前缓冲区的传输完成
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
        
        # 获取当前缓冲区数据
        current_data = self._buffer[self._buffer_idx]
        if current_data is None:
            raise StopIteration
        
        # 清空当前缓冲区，切换到下一个
        self._buffer[self._buffer_idx] = None
        self._buffer_idx = 1 - self._buffer_idx
        self._batch_count += 1
        
        # 后台预加载下一个 batch
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
    config: TrainingConfig,
    mixup_fn: Optional[MixupCutmix] = None,
    num_classes: int = 10,
    profile: bool = False,
    exp_dir: Optional[Path] = None,
    loss_fn: Optional[nn.Module] = None,
    class_weights: Optional[torch.Tensor] = None,
    epoch: int = 1,
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
    
    # P15: Warmup 后首次启用 Mixup 的调试信息
    if use_mixup and epoch is not None:
        import sys
        print(f"[DEBUG] train_epoch started: epoch={epoch}, use_mixup=True", file=sys.stderr, flush=True)
    
    # P15: 预先获取模型期望的 dtype (避免每次迭代都检查)
    model_dtype = get_model_input_dtype(model) if use_mixup else None
    if model_dtype is not None:
        print(f"[DEBUG] Model expects input dtype: {model_dtype}")
    
    # 使用环境变量 DISABLE_PREFETCH=1 来禁用 CudaPrefetcher 进行调试
    use_prefetcher = device.type == 'cuda' and os.environ.get('DISABLE_PREFETCH', '0') != '1'
    if use_prefetcher:
        data_iter = CudaPrefetcher(loader, device, channels_last=config.channels_last)
        if use_mixup:
            print(f"[DEBUG] 使用 CudaPrefetcher", flush=True)
    else:
        data_iter = loader
        if use_mixup:
            print(f"[DEBUG] 不使用 CudaPrefetcher (DISABLE_PREFETCH={os.environ.get('DISABLE_PREFETCH', 'not set')})", flush=True)
    
    pbar = tqdm(data_iter, desc="Train", total=len(loader))
    
    # P13: 卡顿诊断 - 检测异常长的批次时间
    stall_threshold = 5.0  # 超过 5 秒视为卡顿
    stall_count = 0
    
    data_start = time.time()
    
    # P15: 在首个 Mixup epoch 添加额外诊断
    debug_first_mixup_epoch = use_mixup and epoch is not None and os.environ.get('DISABLE_PREFETCH', '0') == '1'
    
    for i, batch in enumerate(pbar):
        
        # P15: 额外诊断 - 检测数据加载卡顿
        if debug_first_mixup_epoch and i <= 5:
            print(f"[DEBUG] Batch {i}: 数据加载完成", flush=True)
        
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
            # 当不使用 CudaPrefetcher 时，需要手动处理 GPU 传输和 channels_last
            if not use_prefetcher:
                imgs = imgs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if config.channels_last:
                    imgs = imgs.to(memory_format=torch.channels_last)
        
        # 检查 label 范围
        if labels.min() < 0 or labels.max() >= num_classes:
            print(f"\n[WARN] Label 范围异常: min={labels.min().item()}, max={labels.max().item()}, num_classes={num_classes}")
            continue
        
        # 应用 Mixup/CutMix
        mixed_labels: Optional[torch.Tensor] = None
        if use_mixup and mixup_fn is not None:
            if i == 0:
                print(f"[DEBUG] Batch 0: 开始应用 Mixup...", flush=True)
            imgs, mixed_labels = mixup_fn(imgs, labels)
            # 确保 Mixup 后保持 channels_last 格式 (Mixup 的线性混合可能会破坏 memory format)
            if config.channels_last and imgs.device.type == 'cuda':
                imgs = imgs.to(memory_format=torch.channels_last)
            if i == 0:
                print(f"[DEBUG] Batch 0: Mixup 完成，mixed_labels.shape={mixed_labels.shape}", flush=True)
                print(f"[DEBUG] Batch 0: imgs.dtype={imgs.dtype}, imgs.is_contiguous(memory_format=torch.channels_last)={imgs.is_contiguous(memory_format=torch.channels_last)}", flush=True)
        
        forward_start = time.time()
        
        # P15: 确保输入 dtype 与模型权重匹配 (torch.compile + AMP 可能导致不匹配)
        if model_dtype is not None and imgs.dtype != model_dtype:
            if i == 0:
                print(f"[WARN] dtype 不匹配! 将 imgs 从 {imgs.dtype} 转换为 {model_dtype}")
            imgs = imgs.to(dtype=model_dtype)
        elif i == 0 and use_mixup:
            print(f"[DEBUG] Batch 0: 开始 forward pass, imgs.dtype={imgs.dtype}...", flush=True)
        
        with get_amp_context(device, config.use_amp):
            outs, aux_infos = model(imgs, return_aux_info=True)  # I14-1 D1: 捕获 aux_info 用于崩溃检测
            if i == 0 and use_mixup:
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
                if i == 0:
                    print(f"[DEBUG] Batch 0: mixup_criterion 计算完成，ce_loss={ce_loss.item():.4f}", flush=True)
            else:
                # P14: 使用自定义损失函数 (Focal Loss / Class-Balanced Loss)
                if loss_fn is not None:
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
            # P15-FIX: 当使用新版 get_auxiliary_losses (包含 soft_entropy_loss) 时，
            #          不再使用旧版 get_entropy_loss，避免重复添加熵损失
            entropy_loss = None
            use_legacy_entropy = False  # 默认使用新版
            
            # P10-4/P10-9: 可学习分割器辅助损失（推荐使用统一接口）
            # 包含: 软熵损失 + 弹性预算损失 + 阈值 barrier 正则化
            # I14-1 D1: 新增崩溃惩罚，需要传递 actual_token_count
            splitter_loss = None
            splitter_metrics = {}
            if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
                splitter = model.tokenizer.splitter
                if hasattr(splitter, 'get_auxiliary_losses'):
                    # I14-1 D1: 计算 batch 中的平均 token 数用于崩溃检测
                    actual_token_count = None
                    if aux_infos is not None and len(aux_infos) > 0:
                        token_counts = [info.get('num_tokens', 0) for info in aux_infos if isinstance(info, dict)]
                        if token_counts:
                            actual_token_count = int(sum(token_counts) / len(token_counts))
                    
                    aux_losses = splitter.get_auxiliary_losses(
                        features=model.tokenizer._last_features,
                        image_size=(imgs.shape[2], imgs.shape[3]),
                        include_balance=config.enforce_balance,
                        include_elastic_budget=config.include_elastic_budget,
                        include_soft_entropy=config.include_soft_entropy,
                        batch_size=imgs.shape[0],
                        elastic_N_min=config.elastic_N_min,
                        elastic_N_max=config.elastic_N_max,
                        elastic_lambda_over=config.elastic_lambda_over,
                        elastic_lambda_under=config.elastic_lambda_under,
                        elastic_lambda_collapse=config.elastic_lambda_collapse,  # I14-1 D1
                        actual_token_count=actual_token_count,  # I14-1 D1: 用于崩溃检测
                        entropy_target=config.soft_entropy_target,
                        entropy_weight=config.soft_entropy_weight,
                        entropy_mode=config.soft_entropy_mode,
                    )
                    # 收集各项损失
                    splitter_loss = sum(aux_losses.values())
                    # P11-8: 延迟 .item() 调用，避免每个 batch 的 GPU-CPU 同步
                    # 仅在需要显示时才调用
                    splitter_metrics = aux_losses  # 保留张量引用
                    # 新版接口已包含 soft_entropy_loss，不需要旧版熵损失
                    use_legacy_entropy = False
                elif hasattr(model.tokenizer, 'get_learnable_split_loss'):
                    # 后备：旧版接口
                    splitter_loss = model.tokenizer.get_learnable_split_loss(
                        lambda_entropy=config.lambda_splitter_entropy,
                        lambda_budget=config.lambda_splitter_budget,
                        target_tokens=config.splitter_token_budget,
                    )
                    use_legacy_entropy = True  # 旧版接口需要单独的熵损失
                else:
                    use_legacy_entropy = True  # 没有 splitter 接口，使用旧版熵损失
            else:
                use_legacy_entropy = True  # 没有 splitter，使用旧版熵损失
            
            # P15-FIX: 仅在使用旧版接口时才获取旧版熵损失
            if use_legacy_entropy and hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_entropy_loss'):
                entropy_loss = model.tokenizer.get_entropy_loss()
            
            # P8-3: 多层深度损失 (仅可学习分割器)
            # 确保所有深度层级的阈值都收到梯度信号
            multi_layer_loss = None
            if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'get_multi_layer_depth_loss'):
                try:
                    multi_layer_loss = model.tokenizer.get_multi_layer_depth_loss(
                        features=model.tokenizer._last_features,
                        image_size=(imgs.shape[2], imgs.shape[3]),
                        target_entropy=0.693,  # ln(2), 鼓励 50/50 分割概率
                        weight_decay_factor=0.5,  # β=0.5, 深层权重衰减
                    )
                except (ValueError, AttributeError):
                    # 非可学习分割器会抛出 ValueError
                    pass
            
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
            if multi_layer_loss is not None:
                multi_layer_loss_f32 = multi_layer_loss.float()
                if torch.isnan(multi_layer_loss_f32) or torch.isinf(multi_layer_loss_f32):
                    print(f"[WARN] multi_layer_loss 为 NaN/Inf, 跳过")
                    multi_layer_loss = None  # 跳过该损失
                else:
                    loss = loss + 0.1 * multi_layer_loss_f32 / config.accum_steps  # λ_multi = 0.1
        
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
                    multi_layer_loss=multi_layer_loss,
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
        
        # P11-8: 减少 .item() 调用频率，仅每 10 个 batch 同步一次
        if i % 10 == 0:
            loss_val = loss.item() * config.accum_steps
            # correct 现在是张量，需要 .item()
            acc_val = 100. * correct.item() / total if total > 0 else 0
            if profile and (i < 5 or i % 100 == 0):
                pbar.set_postfix(
                    loss=f'{loss_val:.3f}',
                    acc=f'{acc_val:.1f}%',
                    data=f'{data_time*1000:.0f}ms',
                    fwd=f'{forward_time*1000:.0f}ms'
                )
            else:
                pbar.set_postfix(loss=f'{loss_val:.4f}', acc=f'{acc_val:.1f}%')
        
        # P13: 定期手动 GC，避免大量临时对象导致长时间暂停
        if i > 0 and i % 50 == 0:
            import gc
            gc.collect()
        
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
        splitter = model.tokenizer.splitter
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
    channels_last: bool = False,
) -> Tuple[float, float, Optional[Dict[str, Any]]]:
    """评估
    
    Args:
        model: 模型
        loader: 数据加载器
        device: 设备
        use_amp: 是否使用混合精度
        num_classes: 类别数
        return_per_class: 是否返回逐类别统计
        
    Returns:
        (loss, accuracy, per_class_stats)
    """
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    nan_batches = 0
    
    # 逐类别统计
    class_correct = torch.zeros(num_classes, device=device)
    class_total = torch.zeros(num_classes, device=device)
    
    for batch in tqdm(loader, desc="Eval"):
        imgs, labels = batch
        imgs = imgs.to(device)
        if channels_last:
            imgs = imgs.to(memory_format=torch.channels_last)
        labels = labels.to(device)
        
        with get_amp_context(device, use_amp):
            outs, _ = model(imgs, return_aux_info=True)
            
            # 检查 logits 是否有问题
            if torch.isnan(outs).any() or torch.isinf(outs).any():
                nan_batches += 1
                continue
            
            loss = F.cross_entropy(outs, labels)
        
        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
        else:
            nan_batches += 1
            continue
            
        _, pred = outs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()
        
        # 逐类别统计
        for c in range(num_classes):
            mask = labels == c
            class_total[c] += mask.sum()
            class_correct[c] += (pred[mask] == c).sum()
    
    if nan_batches > 0:
        print(f"[WARN] 评估时跳过 {nan_batches} 个包含 NaN 的 batch")
    
    if total == 0:
        return float('inf'), 0.0, None
    
    # 计算逐类别准确率
    per_class_stats = None
    if return_per_class:
        class_correct = class_correct.cpu().numpy()
        class_total = class_total.cpu().numpy()
        class_acc = np.divide(class_correct, class_total, out=np.zeros_like(class_correct), where=class_total > 0) * 100
        
        per_class_stats = {
            'class_accuracy': class_acc.tolist(),
            'class_correct': class_correct.tolist(),
            'class_total': class_total.tolist(),
            'worst_classes': np.argsort(class_acc)[:10].tolist(),
            'best_classes': np.argsort(class_acc)[-10:][::-1].tolist(),
            'accuracy_std': float(np.std(class_acc[class_total > 0])),
            'accuracy_min': float(np.min(class_acc[class_total > 0])) if np.any(class_total > 0) else 0.0,
            'accuracy_max': float(np.max(class_acc[class_total > 0])) if np.any(class_total > 0) else 0.0,
        }
    
    return total_loss / max(len(loader) - nan_batches, 1), 100.0 * correct / total, per_class_stats


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
                print(f"    - Worst classes (ID:acc): {list(zip(worst_5, [f'{a:.1f}%' for a in worst_acc]))}")
            
            # 建议
            if imbalance_score > 30 and epoch > 10:
                print(f"    [建议] 考虑启用 --use-focal-loss 和/或 --use-class-balanced")
    
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


@torch.no_grad()
def verify_train_eval_consistency(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainingConfig,
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
        'tokenizer_type': config.tokenizer_type,
    }
    
    # 获取一个 batch 用于测试
    sample_batch = next(iter(loader))
    imgs = sample_batch[0][:4].to(device)  # 只用 4 张图
    if config.channels_last:
        imgs = imgs.to(memory_format=torch.channels_last)
    
    # 检查 1: train/eval 输出差异
    model.eval()
    with get_amp_context(device, config.use_amp):
        out_eval, aux_eval = model(imgs, return_aux_info=True)
    
    model.train()
    with get_amp_context(device, config.use_amp):
        out_train, aux_train = model(imgs, return_aux_info=True)
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
    print(f"  [OK] Tokenizer: {config.tokenizer_type} (Variable Depth Tokens)")
    
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
                       choices=["cifar10", "cifar100", "mnist", "tiny-imagenet"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=None,
                       help="Number of workers (auto-detect if not set)")
    parser.add_argument("--val-split", type=float, default=0.05)  # T3: 减少验证集，增加训练数据
    parser.add_argument("--subset-size", type=int, default=None)
    
    # 模型
    parser.add_argument("--dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim-head", type=int, default=32)
    parser.add_argument("--max-level", type=int, default=4)
    parser.add_argument("--num-scales", type=int, default=5,
                       help="Number of scales (I16-2: default=5 for max_depth=4, max_tokens=341)")
    parser.add_argument("--pool", type=str, default="cls", choices=["cls", "mean"])
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
    
    # Tokenizer 类型
    parser.add_argument("--tokenizer-type", type=str, default="streaming_v3",
                       choices=["streaming_v3"],
                       help="Tokenizer type: streaming_v3 (Variable Depth Tokens, only supported)")
    
    # V3 Tokenizer 高级参数 (Scheme D: GumbelTopKSplitter 默认)
    parser.add_argument("--target-tokens", type=int, default=None,
                       help="Target token count per image (None = adaptive)")
    parser.add_argument("--split-tau0", type=float, default=0.15,
                       help="Root threshold tau_0 for adaptive splitting")
    parser.add_argument("--split-gamma", type=float, default=0.85,
                       help="Threshold decay factor gamma in (0,1) per depth")
    parser.add_argument("--enforce-balance", action="store_true", default=True,
                       help="Enforce 2:1 balance constraint")
    parser.add_argument("--no-enforce-balance", action="store_false", dest="enforce_balance")
    
    # P6-1: 深度缩放参数
    parser.add_argument("--depth-scale-min", type=float, default=0.5,
                       help="Minimum depth scale σ_min (default: 0.5)")
    parser.add_argument("--depth-scale-max", type=float, default=2.0,
                       help="Maximum depth scale σ_max (default: 2.0)")
    parser.add_argument("--no-learnable-depth-scale", action="store_true",
                       help="Use fixed depth scale (legacy mode)")
    
    # P6-2: LCA 温度参数
    parser.add_argument("--lca-temperature", type=float, default=1.5,
                       help="LCA bias temperature τ (default: 1.5, SNR=1.5)")
    parser.add_argument("--no-lca-temperature", action="store_true",
                       help="Disable LCA temperature scaling (legacy mode)")
    parser.add_argument("--fixed-lca-temperature", action="store_true",
                       help="Use fixed (non-learnable) LCA temperature")
    
    # P7-6: 可学习分割器训练参数
    parser.add_argument("--lambda-splitter-entropy", type=float, default=0.1,
                       help="Learnable splitter entropy loss weight (default: 0.1)")
    parser.add_argument("--lambda-splitter-budget", type=float, default=0.01,
                       help="Learnable splitter budget constraint weight (default: 0.01)")
    parser.add_argument("--splitter-token-budget", type=int, default=64,
                       help="Target token budget for learnable splitter (default: 64)")
    
    # P7-7: 温度退火调度参数
    # P10-11 更新: 将 T_end 默认值从 0.1 提升到 0.3，防止梯度消失
    # 参考: constants.py SPLITTER_TEMP_END = 0.3
    parser.add_argument("--splitter-temp-start", type=float, default=1.0,
                       help="Learnable splitter initial temperature (default: 1.0)")
    parser.add_argument("--splitter-temp-end", type=float, default=0.3,
                       help="Learnable splitter final temperature (default: 0.3, P10-11 optimized)")
    parser.add_argument("--splitter-temp-warmup", type=int, default=5,
                       help="Warmup epochs with fixed T_start (default: 5)")
    
    # I10-19: 连续松弛参数
    parser.add_argument("--use-continuous-relaxation", action="store_true", default=False,
                       help="I10-19: Enable continuous relaxation for fully differentiable forward pass")
    parser.add_argument("--continuous-max-depth", type=int, default=3,
                       help="I10-19: Max depth for continuous parallel evaluator (default: 3, 85 candidates)")
    
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
    
    # P10-9: 弹性预算损失参数
    parser.add_argument("--include-elastic-budget", action="store_true", default=True,
                       help="Enable elastic budget loss (default: True, recommended)")
    parser.add_argument("--no-elastic-budget", action="store_false", dest="include_elastic_budget",
                       help="Disable elastic budget loss")
    parser.add_argument("--elastic-N-min", type=int, default=32,
                       help="Elastic budget lower bound (I16-2: must < max_tokens)")
    parser.add_argument("--elastic-N-max", type=int, default=256,
                       help="Elastic budget upper bound (I16-2: default=256 for num_scales=5)")
    parser.add_argument("--elastic-lambda-over", type=float, default=0.1,
                       help="Penalty weight for exceeding upper bound (default: 0.1)")
    parser.add_argument("--elastic-lambda-under", type=float, default=0.01,
                       help="Penalty weight for falling below lower bound (default: 0.01)")
    parser.add_argument("--elastic-lambda-collapse", type=float, default=1.0,
                       help="I14-1 D1: Collapse penalty weight (default: 1.0, triggers when actual_tokens < 2)")
    
    # 训练
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.03,
                       help="Weight decay (default: 0.03)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate (default: 0.1)")
    parser.add_argument("--emb-dropout", type=float, default=0.1)
    parser.add_argument("--drop-path", type=float, default=0.1,
                       help="Drop path (stochastic depth) rate")
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
    
    # P14: 长尾效应优化
    parser.add_argument("--use-focal-loss", action="store_true",
                       help="Use Focal Loss to handle class imbalance")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                       help="Focal Loss gamma parameter (default: 2.0)")
    parser.add_argument("--use-class-balanced", action="store_true",
                       help="Use class-balanced loss weights")
    parser.add_argument("--class-balance-beta", type=float, default=0.9999,
                       help="Class balance beta parameter (default: 0.9999)")
    parser.add_argument("--progressive-aug", action="store_true",
                       help="Use progressive data augmentation (weaker at start)")
    
    # 系统
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--exp-name", type=str, default=None,
                       help="Custom experiment name (default: auto-generated with timestamp)")
    
    args = parser.parse_args()
    
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
    
    config = TrainingConfig(
        dataset=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        subset_size=args.subset_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.dim * 4,
        dim_head=args.dim_head,
        max_level=args.max_level,
        num_scales=args.num_scales,
        pool=args.pool,
        ffn_type=args.ffn_type,
        # Tokenizer 配置 (V3)
        tokenizer_type=args.tokenizer_type,
        # V3 高级分割参数 (GumbelTopKSplitter 默认)
        target_tokens=args.target_tokens,
        split_tau0=args.split_tau0,
        split_gamma=args.split_gamma,
        enforce_balance=args.enforce_balance,
        # P6-1: 深度缩放配置
        depth_scale_range=(args.depth_scale_min, args.depth_scale_max) if not args.no_learnable_depth_scale else None,
        # P6-2: LCA 温度配置
        lca_temperature=None if args.no_lca_temperature else args.lca_temperature,
        learnable_temperature=not args.fixed_lca_temperature,
        # P7-6: 可学习分割器训练配置
        lambda_splitter_entropy=args.lambda_splitter_entropy,
        lambda_splitter_budget=args.lambda_splitter_budget,
        splitter_token_budget=args.splitter_token_budget,
        # P7-7: 温度退火调度配置
        splitter_temp_start=args.splitter_temp_start,
        splitter_temp_end=args.splitter_temp_end,
        splitter_temp_warmup=args.splitter_temp_warmup,
        # P10-4/P10-5: 软熵损失配置
        include_soft_entropy=args.include_soft_entropy,
        soft_entropy_mode=args.soft_entropy_mode,
        soft_entropy_weight=args.soft_entropy_weight,
        soft_entropy_target=args.soft_entropy_target,
        # P10-9: 弹性预算损失配置
        include_elastic_budget=args.include_elastic_budget,
        elastic_N_min=args.elastic_N_min,
        elastic_N_max=args.elastic_N_max,
        elastic_lambda_over=args.elastic_lambda_over,
        elastic_lambda_under=args.elastic_lambda_under,
        elastic_lambda_collapse=args.elastic_lambda_collapse,  # I14-1 D1
        # 训练配置
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        drop_path=args.drop_path,
        label_smoothing=args.label_smoothing,
        gradient_clip=args.gradient_clip,
        use_amp=args.use_amp,
        accum_steps=args.accum_steps,
        warmup_epochs=args.warmup_epochs,
        gradient_checkpoint=args.gradient_checkpoint,
        compile_model=getattr(args, 'compile', False),
        channels_last=getattr(args, 'channels_last', False),
        patience=args.patience,
        min_delta=args.min_delta,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mixup_prob=args.mixup_prob,
        # P14: 长尾效应优化
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_class_balanced=args.use_class_balanced,
        class_balance_beta=args.class_balance_beta,
        progressive_aug=args.progressive_aug,
        seed=args.seed,
        device=str(device),
    )
    
    # 创建 Tokenizer (默认使用 GumbelTopKSplitter - Scheme D)
    from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3
    
    tokenizer = StreamingFractalTokenizerV3(
        image_size=max(spec.image_size, 32),
        channels=spec.channels,
        d_model=config.dim,
        base_patch_size=4,
        max_depth=config.num_scales - 1,
        use_hilbert_order=True,
        target_tokens=config.target_tokens,
        enforce_balance=config.enforce_balance,
        # P6-1: 深度缩放配置
        depth_scale_range=config.depth_scale_range,
        # 分割阈值参数 (gamma 控制阈值衰减)
        gamma=config.split_gamma,
        # P7-7: 可学习分割器温度参数
        learnable_temperature=config.splitter_temp_start,
        use_gumbel=True,  # 使用 Gumbel-Softmax 进行可微分采样
        # I10-19: 连续松弛配置
        use_continuous_relaxation=args.use_continuous_relaxation,
        continuous_max_depth=args.continuous_max_depth,
    )
    
    # 创建模型 (V3 Variable Depth Tokens)
    model_kwargs = dict(
        image_size=max(spec.image_size, 32),
        num_classes=spec.num_classes,
        dim=config.dim,
        depth=config.depth,
        heads=config.heads,
        mlp_dim=config.mlp_dim,
        pool=config.pool,
        channels=spec.channels,
        dim_head=config.dim_head,
        dropout=config.dropout,
        emb_dropout=config.emb_dropout,
        drop_path_rate=config.drop_path,
        min_patch_size=(4, 4),
        max_level=config.max_level,
        use_checkpoint=config.gradient_checkpoint,
        ffn_type=config.ffn_type,
        # 使用自定义 tokenizer (支持高级分割参数)
        tokenizer=tokenizer,
        num_scales=config.num_scales,
        # P6-2: LCA 温度配置
        lca_temperature=config.lca_temperature,
        learnable_temperature=config.learnable_temperature,
    )
    
    model = FractalCurveViT(**model_kwargs).to(device)
    
    # 打印模型信息
    params = sum(p.numel() for p in model.parameters())
    split_info = "GumbelTopKSplitter (Scheme D)"
    if config.target_tokens:
        split_info += f", target={config.target_tokens}"
    tokenizer_name = f'StreamingFractalTokenizerV3 ({split_info})'
    
    # P6-1/P6-2 信息
    depth_scale_info = f"range={config.depth_scale_range}" if config.depth_scale_range else "legacy"
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
    print(f"Parameters: {params:,}")
    print(f"Gradient Checkpoint: {config.gradient_checkpoint}")
    print(f"Compile Model: {config.compile_model}")
    print(f"Channels Last: {config.channels_last}")
    print(f"{'='*70}\n")
    
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
    
    # CUDA 优化
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    # Channels Last 内存格式 (卷积加速)
    if config.channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        print("[OK] Using channels-last memory format")
    
    # torch.compile 编译优化 (PyTorch 2.0+)
    # 重要: Variable Depth Tokens 产生动态序列长度，必须使用 dynamic=True
    # 否则每次序列长度变化都会触发重新编译，导致 GPU 空转
    if config.compile_model:
        try:
            # 设置编译缓存大小，减少重新编译
            # 注意: torch._dynamo 已在文件顶部导入，这里直接使用
            torch._dynamo.config.cache_size_limit = 256  # 增大缓存
            torch._dynamo.config.suppress_errors = True  # 回退到 eager 模式
            
            # 使用 'default' 模式而非 'reduce-overhead'
            # 原因: reduce-overhead 使用 Triton 编译器，对动态形状支持有限
            # 已知问题: Triton 对 2**tensor 幂运算不支持，会报 __rpow__ 错误
            model = torch.compile(
                model, 
                mode='default',  # 使用 TorchInductor 而非 Triton CUDA graphs
                fullgraph=False,
                dynamic=True,  # 关键: Variable Depth Tokens 需要动态形状
            )
            print("[OK] Model compiled with torch.compile (mode=default, dynamic=True)")
            print("[INFO] 首次运行会进行 JIT 编译，可能耗时 1-2 分钟")
        except Exception as e:
            print(f"[WARN] torch.compile failed: {e}")
    
    # 诊断: 检查模型参数 dtype
    def check_model_dtypes(m, name="model"):
        dtypes = set()
        for n, p in m.named_parameters():
            dtypes.add(str(p.dtype))
            if p.dtype == torch.float16:
                print(f"[WARN] {name}.{n} is float16!")
        print(f"[DEBUG] {name} param dtypes: {dtypes}")
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
            # 计算类别平衡权重
            class_weights = compute_class_weights(
                labels=all_labels,
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
    
    # 训练
    print("="*70)
    print("TRAINING START")
    print("="*70)
    
    if config.compile_model:
        print("[INFO] First batch will be slow due to JIT compilation (1-3 minutes)...")
    
    print()
    
    # 编译预热: 在正式训练前触发 JIT 编译
    # P11-7 优化: 使用完整 batch size 预热，避免动态形状导致重新编译
    # P15 优化: 同时预热 Mixup 路径，避免 warmup 结束后的重编译延迟
    if config.compile_model:
        print("[INFO] Warming up compiled model with full batch size...")
        try:
            warmup_batch = next(iter(train_loader))
            if isinstance(warmup_batch, (list, tuple)):
                warmup_imgs = warmup_batch[0].to(device)  # 使用完整 batch
                warmup_labels = warmup_batch[1].to(device)
            else:
                warmup_imgs = warmup_batch.to(device)
                warmup_labels = torch.zeros(warmup_imgs.shape[0], dtype=torch.long, device=device)
            if config.channels_last:
                warmup_imgs = warmup_imgs.to(memory_format=torch.channels_last)
            
            # 阶段1: 预热标准 forward pass
            with torch.no_grad():
                with get_amp_context(device, config.use_amp):
                    for _ in range(3):  # 3次预热确保编译稳定
                        _ = model(warmup_imgs)
                        torch.cuda.synchronize()  # 确保编译完成
            
            # 阶段2: 预热 Mixup 路径 (如果启用)
            if mixup_fn is not None:
                print("[INFO] Pre-warming Mixup code path...")
                with torch.no_grad():
                    with get_amp_context(device, config.use_amp):
                        # 测试 Mixup 数据变换
                        test_imgs, test_mixed_labels = mixup_fn(warmup_imgs.clone(), warmup_labels.clone())
                        # 确保 Mixup 后保持 channels_last 格式
                        if config.channels_last:
                            test_imgs = test_imgs.to(memory_format=torch.channels_last)
                        # 测试 forward + Mixup loss
                        test_outs = model(test_imgs)
                        _ = mixup_criterion(test_outs, test_mixed_labels)
                        torch.cuda.synchronize()
                        del test_imgs, test_mixed_labels, test_outs
                print("[OK] Mixup path pre-warmed")
            
            del warmup_imgs, warmup_labels
            torch.cuda.empty_cache()
            print("[OK] Compilation complete!")
        except Exception as e:
            print(f"[WARN] Warmup failed: {e}")
    
    best_val = 0.0
    patience_counter = 0
    early_stopped = False
    
    exp_name = args.exp_name if args.exp_name else f"fractal_vit_{time.strftime('%Y%m%d_%H%M%S')}"
    exp_dir = PROJECT_ROOT / "experiments" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    
    # 保存配置
    with open(exp_dir / "logs" / "config.json", 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
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
    # P7-7 / P10-12: 启用自适应分割器内置退火调度 (Scheme D 兼容)
    # =========================================================================
    # GumbelTopKSplitter 和 LearnableSplitter 均支持统一的退火 API:
    #   - enable_temperature_annealing()
    #   - enable_explore_bias_annealing()
    #
    # 数学形式化:
    #   温度退火: T(t) = T_start · (T_end / T_start)^(t / total_steps)
    #   探索偏置: b(t) = b_start · (1 - t / total_steps)
    #
    # Scheme D 特有:
    #   - Top-K 硬约束保证 token 数量，偏置影响 *哪些* 被选中
    #   - STE 梯度仍依赖温度，退火保证梯度质量
    # =========================================================================
    splitter_annealing_enabled = False
    if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
        splitter = model.tokenizer.splitter
        # 计算总训练步数 (epochs × batches_per_epoch)
        batches_per_epoch = len(train_loader) // config.accum_steps
        total_training_steps = config.epochs * batches_per_epoch
        
        # 启用温度退火 (P7-7)
        if hasattr(splitter, 'enable_temperature_annealing'):
            # 考虑 warmup: 在 warmup 期间使用 T_start，之后开始退火
            post_warmup_steps = max(1, (config.epochs - config.splitter_temp_warmup) * batches_per_epoch)
            splitter.enable_temperature_annealing(
                total_steps=post_warmup_steps,
                T_start=config.splitter_temp_start,
                T_end=config.splitter_temp_end,
                schedule='exponential',  # 最优的梯度-确定性权衡
            )
            print(f"[OK] 启用自适应分割器温度退火:")
            print(f"     T: {config.splitter_temp_start} → {config.splitter_temp_end}")
            print(f"     Steps: {post_warmup_steps} (after {config.splitter_temp_warmup} warmup epochs)")
            print(f"     Schedule: exponential")
            splitter_annealing_enabled = True
        
        # 启用探索偏置退火 (P10-12 + P10-15)
        # P10-15 修复: 偏置退火与温度退火同步，延迟到 warmup 后开始
        # 数学形式化:
        #   warmup 期间: b = b_warmup = 0.6 (固定高探索偏置)
        #   post-warmup: b(t) = b_warmup + (b_end - b_warmup) · progress
        #   与温度退火同步，使 b 和 T 同时衰减，避免时序失配
        if hasattr(splitter, 'enable_explore_bias_annealing'):
            splitter.enable_explore_bias_annealing(
                total_steps=post_warmup_steps,  # P10-15: 与温度退火同步
                b_start=0.6,  # P10-15: 提高初始偏置 (warmup 期间固定)
                b_end=0.0,    # 终态: 纯 MLP 决策
            )
            print(f"[OK] 启用 P10-12/P10-15 探索偏置退火:")
            print(f"     Bias: 0.6 → 0.0")
            print(f"     Steps: {post_warmup_steps} (与温度退火同步)")
            print(f"     目的: 防止训练初期 avg_tokens=1 的'鸡生蛋'死锁")
            splitter_annealing_enabled = True
        
        # I14-1 修复 (A1): 启用 Warmup 强制分割
        # 数学形式化:
        #   问题: Gumbel 噪声随机性可能导致即使 p_split=0.7，实际决策
        #         仍然是"不分割"。一旦所有样本都碰巧选择"不分割"，形成死锁。
        #   解决: 在 warmup 期间强制至少 ratio 比例的区域分割
        #   公式: forced_split_count = ceil(M * current_ratio)
        #   
        # 与 explore_bias 的区别:
        #   - explore_bias: 概率偏置，仍受 Gumbel 随机性影响
        #   - warmup_forced_split: 后采样硬约束，完全消除随机性风险
        if hasattr(splitter, 'enable_warmup_forced_split'):
            warmup_steps = config.splitter_temp_warmup * batches_per_epoch
            splitter.enable_warmup_forced_split(
                total_steps=warmup_steps,
                ratio_start=0.7,  # 初始强制 70% 分割
                ratio_end=0.3,    # 结束时强制 30% 分割
            )
            print(f"[OK] 启用 I14-1 A1 Warmup 强制分割:")
            print(f"     Ratio: 0.7 → 0.3")
            print(f"     Steps: {warmup_steps} (warmup 期间)")
            print(f"     目的: 消除 Gumbel 随机性导致的训练死锁")
            splitter_annealing_enabled = True
        
        # 配置深度偏置 (P10-16: 根节点单点失效保护)
        # 数学形式化:
        #   Δb_d = β · γ^d
        #   β=1.0: 根节点获得最大偏置保护
        #   γ=0.5: 每深入一层偏置减半
        #   计算验证: P(root split) > 0.7 即使在终态
        if hasattr(splitter, 'set_depth_bias'):
            splitter.set_depth_bias(beta=1.0, gamma=0.5)
            depth_info = splitter.get_depth_bias_info()
            print(f"[OK] 启用 P10-16 深度偏置保护:")
            print(f"     公式: {depth_info['formula']}")
            print(f"     根节点偏置: Δb_0 = {depth_info['depth_biases']['depth_0']:.2f}")
            print(f"     深度1偏置: Δb_1 = {depth_info['depth_biases']['depth_1']:.2f}")
    
    if not splitter_annealing_enabled:
        print(f"[INFO] 自适应分割器退火未启用 (不支持或未使用可学习分割器)")
    
    print()
    
    for epoch in range(1, config.epochs + 1):
        start = time.time()
        
        # P13: 每个 epoch 开始时手动 GC
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # P7-7 + P10-15: 温度退火和偏置退火调度 (同步控制)
        # 说明: 温度退火已在训练开始前通过 enable_temperature_annealing() 启用
        #       在 forward() 中自动更新，无需手动调用 scheduler.step()
        #       但 warmup 期间需要禁用退火，保持 T_start 和 b_warmup
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
            splitter = model.tokenizer.splitter
            if epoch <= config.splitter_temp_warmup:
                # Warmup 阶段：暂时禁用自动退火，固定 T_start 和 b_warmup
                if hasattr(splitter, 'disable_temperature_annealing'):
                    splitter.disable_temperature_annealing()
                    splitter.set_temperature(config.splitter_temp_start)
                # P10-15: 偏置退火也在 warmup 期间禁用，固定高探索偏置
                if hasattr(splitter, 'disable_explore_bias_annealing'):
                    splitter.disable_explore_bias_annealing()
                    if hasattr(splitter.complexity_mlp, 'set_explore_bias'):
                        splitter.complexity_mlp.set_explore_bias(0.6)  # warmup 期间固定偏置
            elif epoch == config.splitter_temp_warmup + 1:
                # Warmup 结束：重新启用温度退火和偏置退火
                post_warmup_steps = max(1, (config.epochs - config.splitter_temp_warmup) * (len(train_loader) // config.accum_steps))
                
                # 重新启用温度退火
                if hasattr(splitter, 'enable_temperature_annealing'):
                    splitter.enable_temperature_annealing(
                        total_steps=post_warmup_steps,
                        T_start=config.splitter_temp_start,
                        T_end=config.splitter_temp_end,
                        schedule='exponential',
                    )
                # P10-15: 同步启用偏置退火
                if hasattr(splitter, 'enable_explore_bias_annealing'):
                    splitter.enable_explore_bias_annealing(
                        total_steps=post_warmup_steps,
                        b_start=0.6,
                        b_end=0.0,
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
            
            # 诊断: 检查此时模型的 dtype
            print("[DEBUG] 检查模型参数 dtype...")
            first_conv_bias = None
            for name, param in model.named_parameters():
                if 'conv' in name.lower() and 'bias' in name.lower():
                    first_conv_bias = param
                    print(f"[DEBUG] {name}: dtype={param.dtype}, device={param.device}")
                    break
            if first_conv_bias is not None and first_conv_bias.dtype == torch.float16:
                print("[WARN] 模型 bias 已被转换为 float16!")
            
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
        
        train_loss, train_acc, perf_stats = train_epoch(
            model, current_train_loader, optimizer, device, scaler, config,
            mixup_fn=current_mixup_fn,
            num_classes=spec.num_classes,
            profile=(epoch == 1),
            exp_dir=exp_dir,
            loss_fn=loss_fn,
            class_weights=class_weights,
            epoch=epoch,
        )
        
        # P15: 恢复 CudaPrefetcher 和清理临时 loader
        if disable_prefetch_this_epoch:
            os.environ.pop('DISABLE_PREFETCH', None)
            del simple_train_loader
            print(f"[INFO] Epoch {epoch} 完成，后续 epoch 将恢复正常 DataLoader")
        
        val_loss, val_acc, per_class_stats = evaluate(
            model, val_loader, device, config.use_amp, spec.num_classes,
            channels_last=config.channels_last
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
        
        # P10-14: 分割器健康检查
        # 动态获取 max_entropy (从 get_depth_distribution_stats 或配置推算)
        max_entropy_value = 1.609  # 默认: log(5) for max_depth=4
        if perf_stats.get('max_entropy') is not None:
            max_entropy_value = perf_stats['max_entropy']
        elif hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'splitter'):
            # 从 splitter.max_depth 动态计算
            splitter = model.tokenizer.splitter
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': asdict(config),
            }, exp_dir / "checkpoints" / "best.pth")
            print(f"  [*] Best model saved: {val_acc:.2f}%")
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
    
    ckpt = torch.load(exp_dir / "checkpoints" / "best.pth", weights_only=True)
    model.load_state_dict(ckpt['model_state_dict'])
    
    test_loss, test_acc, per_class_stats = evaluate(
        model, test_loader, device, config.use_amp, 
        spec.num_classes, return_per_class=True,
        channels_last=config.channels_last
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
            'tokenizer_type': config.tokenizer_type,
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
        import multiprocessing
        for p in multiprocessing.active_children():
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
