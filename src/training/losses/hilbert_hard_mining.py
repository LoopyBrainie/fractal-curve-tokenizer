"""
Hilbert-Aware Hard Mining Loss
==============================

方案 D: Token Variance Mining - 利用 Hilbert 曲线多尺度特性识别困难样本

数学形式化:
-----------
Token 方差定义:
    TokenVar(x_i) = (1/N_i) Σ_j ||f_ij - f̄_i||²
    
其中:
    - N_i: 样本 i 的有效 token 数量 (多尺度，依赖图像复杂度)
    - f_ij ∈ R^D: 第 j 个 token 的特征向量
    - f̄_i = (1/N_i) Σ_j f_ij: 样本平均特征

样本权重公式:
    w_i = 1 + λ × σ((TokenVar(x_i) - μ) / (τ × σ))
    
其中:
    - μ, σ: EMA 统计的 TokenVar 均值和标准差
    - τ: 温度参数 (控制 sigmoid 敏感度)
    - λ: 权重缩放系数
    - σ(): sigmoid 函数

参数推导 (详见 I30-2):
--------------------
λ = 0.5:
    权重范围 [1.0, 1.5]，困难样本额外加权 50%
    设计目标: w_max / w_min = 1.5 (与 Focal Loss 互补而非替代)

τ = 1.0:
    标准化分数 z = ±2 时，权重差异显著:
    - z = +2 (困难): w ≈ 1.44
    - z = -2 (简单): w ≈ 1.06

momentum = 0.1:
    EMA 更新: μ_new = 0.9 × μ_old + 0.1 × μ_batch
    平衡统计稳定性与适应性

Hilbert 原生性:
--------------
TokenVar 自然捕获多尺度特征不一致性:
- 困难样本: 不同尺度 patch 特征差异大 → 高方差
- 简单样本: 尺度间特征一致 → 低方差
- Hilbert 曲线保持局部性: 相邻 token 来自空间相邻区域

复杂度分析:
----------
时间: O(N × D) 每样本，N ≈ 256 (平均 token 数), D = 384
空间: O(2) 存储 EMA 统计量
开销: < 1% 相对于 forward pass
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn

from vit_pytorch.constants import EPS  # I112-3: 统一数值稳定性常量


class HilbertAwareHardMining(nn.Module):
    """
    基于 Token 方差的困难样本挖掘模块
    
    与 Focal Loss 协同工作:
    - Focal Loss: 基于预测置信度的梯度重加权
    - Token Variance: 基于特征空间不一致性的样本重加权
    
    组合效果:
        总梯度增益 = Focal 增益 × Variance 增益
        困难样本: ≈ 243 × 1.44 ≈ 350 (相对简单样本)
    """
    
    def __init__(
        self,
        lambda_weight: float = 0.5,
        temperature: float = 1.0,
        momentum: float = 0.1,
        warmup_batches: int = 100,
        enabled: bool = True,
    ):
        """
        初始化困难样本挖掘模块
        
        Args:
            lambda_weight: 权重缩放系数 λ ∈ (0, 1]
                推荐 0.5，产生权重范围 [1.0, 1.5]
            temperature: sigmoid 温度 τ > 0
                τ = 1.0 对应标准 sigmoid
            momentum: EMA 更新动量 β ∈ (0, 1)
                μ_new = (1-β) × μ_old + β × μ_batch
            warmup_batches: warmup 期间禁用加权
                等待 EMA 统计收敛
            enabled: 是否启用 (用于消融实验)
        """
        super().__init__()
        
        self.lambda_weight = lambda_weight
        self.temperature = temperature
        self.momentum = momentum
        self.warmup_batches = warmup_batches
        self.enabled = enabled
        
        # EMA 统计量 (不参与反向传播)
        self.register_buffer('running_mean', torch.tensor(0.0))
        self.register_buffer('running_var', torch.tensor(1.0))
        self.register_buffer('batch_count', torch.tensor(0))
        
    def compute_token_variance(
        self,
        tokens: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        计算每个样本的 token 方差
        
        数学公式:
            TokenVar(x_i) = (1/N_i) Σ_j ||f_ij - f̄_i||²
        
        向量化实现:
            var_i = tokens[i, :N_i].var(dim=0).sum()
            
        Args:
            tokens: [B, N, D] transformer 输出 tokens
            lengths: [B] 每个样本的有效 token 数 (排除 padding)
                    如果为 None，假设所有位置有效
        
        Returns:
            variances: [B] 每个样本的 token 方差
        """
        B, N, D = tokens.shape
        device = tokens.device
        
        if lengths is None:
            # 所有 token 有效: 直接计算
            # var(dim=1) 计算每个特征维度的方差, 然后 sum
            variances = tokens.var(dim=1, unbiased=False).sum(dim=-1)
        else:
            # P0 修复: 向量化实现 (替代 Python 循环)
            # 数学形式: TokenVar(x_i) = (1/N_i) Σ_j ||f_ij - f̄_i||²
            # 向量化计算所有样本的方差

            # 创建有效区域掩码: [B, N]
            max_len = int(lengths.max())
            indices = torch.arange(N, device=device).unsqueeze(0)  # [1, N]
            lengths_expanded = lengths.unsqueeze(1)  # [B, 1]
            mask = (indices < lengths_expanded).float()  # [B, N], 有效位置为 1

            # 计算有效 token 的加权均值
            # weighted_mean = Σ(mask * f) / Σ(mask)
            token_counts = lengths.clamp(min=1).unsqueeze(1)  # [B, 1], 避免除零
            masked_tokens = tokens * mask.unsqueeze(-1)  # [B, N, D]
            means = masked_tokens.sum(dim=1, keepdim=True) / token_counts  # [B, 1, D]

            # 计算方差: Σ(||f_ij - mean_i||²) / N_i
            # 使用加权方差公式避免显式循环
            centered = masked_tokens - means  # [B, N, D]
            squared_dist = (centered.pow(2) * mask.unsqueeze(-1)).sum(dim=1)  # [B, D]
            variances = (squared_dist / token_counts.squeeze(-1)).sum(dim=-1)  # [B]

            # 对于 L <= 1 的样本，方差设为 0
            variances = variances * (lengths > 1).float()

        return variances
    
    def forward(
        self,
        tokens: torch.Tensor,
        base_loss: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        计算加权损失
        
        损失加权公式:
            L_weighted = (1/B) Σ_i w_i × L_i
            w_i = 1 + λ × σ((TokenVar_i - μ) / (τ × σ))
        
        Args:
            tokens: [B, N, D] transformer 输出 (CLS token 已移除)
            base_loss: [B] 每个样本的基础损失 (unreduced)
                       或 scalar (已 reduce，仅在 disabled 时)
            lengths: [B] 可选，每个样本的有效 token 数
            
        Returns:
            weighted_loss: scalar 加权后的总损失
            info: dict 包含诊断信息
        """
        info = {
            'enabled': self.enabled,
            'batch_count': self.batch_count.item(),
        }
        
        # 禁用或 warmup 期间: 直接返回
        if not self.enabled:
            if base_loss.dim() == 0:
                return base_loss, info
            return base_loss.mean(), info
            
        in_warmup = self.batch_count < self.warmup_batches
        
        # 计算 token 方差
        variances = self.compute_token_variance(tokens, lengths)  # [B]
        
        # 更新 EMA 统计
        with torch.no_grad():
            batch_mean = variances.mean()
            batch_var = variances.var(unbiased=False)
            
            if self.batch_count == 0:
                # 首次: 直接赋值
                self.running_mean.copy_(batch_mean)
                self.running_var.copy_(batch_var)
            else:
                # EMA 更新
                self.running_mean.mul_(1 - self.momentum).add_(
                    batch_mean * self.momentum
                )
                self.running_var.mul_(1 - self.momentum).add_(
                    batch_var * self.momentum
                )
            
            self.batch_count.add_(1)
        
        info['token_var_mean'] = batch_mean.item()
        info['token_var_std'] = batch_var.sqrt().item()
        info['running_mean'] = self.running_mean.item()
        info['running_std'] = self.running_var.sqrt().item()
        
        # Warmup 期间: 不加权，仅收集统计
        if in_warmup:
            info['in_warmup'] = True
            if base_loss.dim() == 0:
                return base_loss, info
            return base_loss.mean(), info
        
        info['in_warmup'] = False

        # 计算样本权重
        # z = (variance - μ) / (τ × σ)
        # I109-2: 添加温度下界保护，防止除零
        # I112-3: 使用 EPS 统一数值稳定性
        temp_safe = max(self.temperature, 1e-6)
        running_std = (self.running_var + EPS).sqrt()
        z = (variances - self.running_mean) / (temp_safe * running_std)

        # w = 1 + λ × σ(z)
        weights = 1.0 + self.lambda_weight * torch.sigmoid(z)
        
        info['weight_min'] = weights.min().item()
        info['weight_max'] = weights.max().item()
        info['weight_mean'] = weights.mean().item()
        
        # 加权损失
        if base_loss.dim() == 0:
            # scalar loss: 无法逐样本加权
            return base_loss, info
        
        # [B] × [B] → scalar
        weighted_loss = (weights * base_loss).mean()
        
        return weighted_loss, info
    
    def reset_statistics(self):
        """重置 EMA 统计量 (用于新实验)"""
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.batch_count.zero_()
    
    def extra_repr(self) -> str:
        return (
            f'lambda_weight={self.lambda_weight}, '
            f'temperature={self.temperature}, '
            f'momentum={self.momentum}, '
            f'warmup_batches={self.warmup_batches}'
        )


def create_hilbert_mining_loss(
    base_loss_fn: nn.Module,
    lambda_weight: float = 0.5,
    temperature: float = 1.0,
    momentum: float = 0.1,
    warmup_batches: int = 100,
) -> nn.Module:
    """
    工厂函数: 包装基础损失函数并添加困难样本挖掘
    
    用法:
        focal_loss = FocalLoss(gamma=2.5)
        mining_loss = create_hilbert_mining_loss(focal_loss)
        
        # 训练时:
        logits, tokens = model(imgs, return_tokens=True)
        loss, info = mining_loss(logits, labels, tokens=tokens)
    
    Args:
        base_loss_fn: 基础损失函数 (如 FocalLoss)
        其余参数同 HilbertAwareHardMining
        
    Returns:
        包装后的损失模块
    """
    return HilbertMiningWrapper(
        base_loss_fn=base_loss_fn,
        lambda_weight=lambda_weight,
        temperature=temperature,
        momentum=momentum,
        warmup_batches=warmup_batches,
    )


class HilbertMiningWrapper(nn.Module):
    """
    包装任意基础损失函数并添加困难样本挖掘
    """
    
    def __init__(
        self,
        base_loss_fn: nn.Module,
        lambda_weight: float = 0.5,
        temperature: float = 1.0,
        momentum: float = 0.1,
        warmup_batches: int = 100,
    ):
        super().__init__()
        self.base_loss_fn = base_loss_fn
        self.mining = HilbertAwareHardMining(
            lambda_weight=lambda_weight,
            temperature=temperature,
            momentum=momentum,
            warmup_batches=warmup_batches,
        )
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tokens: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
        return_info: bool = False,
    ) -> torch.Tensor:
        """
        计算加权损失
        
        Args:
            logits: [B, C] 模型输出
            labels: [B] 真实标签
            tokens: [B, N, D] 可选，用于计算 token 方差
            lengths: [B] 可选，有效 token 长度
            return_info: 是否返回诊断信息
            
        Returns:
            loss: scalar 损失值
            info: (可选) dict 诊断信息
        """
        # 基础损失: 需要 unreduced
        if hasattr(self.base_loss_fn, 'reduction'):
            old_reduction = self.base_loss_fn.reduction
            self.base_loss_fn.reduction = 'none'
            base_loss = self.base_loss_fn(logits, labels)  # [B]
            self.base_loss_fn.reduction = old_reduction
        else:
            base_loss = self.base_loss_fn(logits, labels)  # 可能是 scalar
        
        if tokens is None:
            # 无 token: 仅返回 mean
            loss = base_loss.mean() if base_loss.dim() > 0 else base_loss
            if return_info:
                return loss, {'enabled': False, 'reason': 'no_tokens'}
            return loss
        
        # 应用困难样本挖掘
        loss, info = self.mining(tokens, base_loss, lengths)
        
        if return_info:
            return loss, info
        return loss
