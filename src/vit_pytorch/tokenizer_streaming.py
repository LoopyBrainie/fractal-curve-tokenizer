# -*- coding: utf-8 -*-
"""
Streaming Fractal Tokenizer V3 (Unified Architecture)

数学形式化
============

统一架构 (P7 重构):
    F = SharedConv(I)              # 共享特征提取
    Regions = Splitter(F or I)     # 可学习分割
    T = Embed(F, Regions)          # 区域池化 + 深度编码

Tokenization 过程:
    T: R^{B × C × H × W} → (T ∈ R^{B × N × D}, L ∈ Z^{B × N})

Variable Depth Tokenization:
    N ∈ [N_min, N_max] 根据图像内容自适应
    Regions = LearnableSplit(F)
    Token_i = Pool(F[R_i]) * σ_d + E_d

可学习分割 (Scheme L - LearnableSplitter):
    C_θ(R) = σ(MLP(ROI-Align(F, R)))   # 可学习复杂度评估
    
    相比已移除的规则方法的优势:
    - 端到端可微分 (Gumbel-Softmax + STE)
    - 无复杂度饱和 (MLP vs 方差公式)
    - 可学习阈值: τ_d = τ_{base,d} + δ_d
    - O(D) BFS vs O(N·4^D) DP 复杂度

复杂度分析
----------
StreamingFractalTokenizerV3:
    时间: O(C · H · W) + O(N_max · D × k²)
          ├─ 特征提取 (Conv):      O(C · H · W)         — 共享卷积
          └─ 可学习分割 (MLP):     O(N_max · D × k²)    — ROI + MLP
    空间: O(C · H · W) + O(N_max · D)
          ├─ 特征图:  O(C · H · W)
          └─ Token:   O(N_max · D)

其中: C=channels, H×W=image_size, N_max=max_tokens, D=d_model

Note:
    Scheme B (BalancedGreedySplitter) 和 Scheme C (FixedBudgetDPSplitter)
    已被移除，因为 LearnableSplitter 提供了更优的功能。
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from .base_tokenizer import BaseTokenizer, TokenizerOutput, TokenSequence
from .config_fractal import FractalConfig


class StreamingFractalTokenizerV3(BaseTokenizer):
    """Variable Depth Tokenizer with Learnable Quadtree Splitting.
    
    数学形式化
    ==========
    
    统一架构 (P7 重构):
        F = SharedConv(I)              # 共享特征提取 (提升到 Tokenizer 级别)
        Regions = Splitter(F, I)       # 可学习分割
        Token_i = Pool(F[R_i]) * σ_d + E_d  # 区域池化 + 深度编码
    
    可学习分割 (LearnableSplitter):
        C_θ(R) = σ(MLP(ROI-Align(F, R)))   # 可学习复杂度
        p_split = σ((C_θ - τ_d) / T)        # 软分割决策
        z ~ Gumbel-Softmax                   # 可微分采样
    
    梯度流:
        L_cls → T → F → SharedConv (端到端)
        L_cls → T → Splitter → MLP (可学习分割)
    
    Args:
        image_size: 输入图像尺寸
        channels: 图像通道数
        d_model: 输出嵌入维度
        base_patch_size: 最细粒度 patch 大小
        max_depth: 最大四叉树深度
        use_hilbert_order: 是否使用 Hilbert 曲线排序
        target_tokens: 目标 token 数量
        enforce_balance: 是否强制 2:1 平衡约束
        depth_scale_range: (P6-1) 深度缩放范围 (σ_min, σ_max)
        gamma: 可学习分割器的阈值衰减因子 γ ∈ (0,1)
        learnable_temperature: 可学习分割的初始温度
        use_gumbel: 是否使用 Gumbel-Softmax
        
    Note:
        Scheme B (BalancedGreedySplitter) 和 Scheme C (FixedBudgetDPSplitter)
        已被移除。LearnableSplitter 提供了完全覆盖的功能并增加了:
        - 端到端可微分性
        - 无复杂度饱和
        - 可学习阈值
    """
    
    def __init__(
        self,
        image_size: Union[int, Tuple[int, int]] = 224,
        channels: int = 3,
        d_model: int = 256,
        base_patch_size: int = 4,
        max_depth: int = 4,
        use_hilbert_order: bool = True,
        target_tokens: Optional[int] = None,
        enforce_balance: bool = True,
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
        gamma: float = 0.85,
        learnable_temperature: float = 1.0,
        use_gumbel: bool = True,
    ) -> None:
        super().__init__()
        
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.image_size = image_size
        self.channels = channels
        self.d_model = d_model
        self.base_patch_size = base_patch_size
        self.max_depth = max_depth
        self.use_hilbert_order = use_hilbert_order
        self._use_learnable_split = True  # Always use LearnableSplitter
        
        # =====================================================================
        # Hilbert-Native Patch Embedding (包含 SharedConv)
        # =====================================================================
        from .embed_hilbert_patch import HilbertNativePatchEmbed
        self.patch_embed = HilbertNativePatchEmbed(
            channels=channels,
            dim=d_model,
            base_patch_size=base_patch_size,
            max_depth=max_depth,
            conv_layers=2,
            use_batch_norm=True,
            depth_scale_range=depth_scale_range,
        )
        
        # =====================================================================
        # LearnableSplitter (Scheme B/C have been removed)
        # =====================================================================
        # Mathematical justification for removal:
        # - Scheme B: C(R) = Var/(Var+σ₀²) saturates as Var → ∞
        # - Scheme C: O(N·4^D) DP complexity, not differentiable
        # - Scheme L: C_θ(R) = σ(MLP(ROI-Align(F, R))) - no saturation, O(D) BFS
        # =====================================================================
        from .split_adaptive import LearnableSplitter
        
        self.splitter = LearnableSplitter(
            feature_dim=d_model,
            max_depth=max_depth,
            hidden_dim=64,
            pool_size=4,
            temperature=learnable_temperature,
            use_gumbel=use_gumbel,
            enforce_balance=enforce_balance,
            min_region_size=base_patch_size,
            init_tau_base=0.5,
            init_tau_gamma=gamma,
        )
        
        self._last_split_stats: Optional[Dict[str, Any]] = None
        self._last_features: Optional[torch.Tensor] = None
    
    @property 
    def shared_conv(self) -> nn.Module:
        """获取共享卷积层 (用于可学习分割)."""
        return self.patch_embed.shared_conv
    
    @classmethod
    def from_config(
        cls,
        config: "FractalConfig",
        channels: int = 3,
        d_model: int = 256,
    ) -> "StreamingFractalTokenizerV3":
        """从 FractalConfig 创建 Tokenizer 实例."""
        return cls(
            image_size=config.image_size,
            channels=channels,
            d_model=d_model,
            base_patch_size=config.patch_sizes[0] if config.patch_sizes else 4,
            max_depth=config.max_depth,
            use_hilbert_order=True,
        )
    
    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """Variable Depth tokenization.
        
        数学形式化:
            1. F = SharedConv(I)           # 特征提取
            2. Regions = Splitter(F or I)  # 分割 (可学习或规则)
            3. T = Embed(F, Regions)       # 区域池化
        """
        if images.dim() != 4:
            raise ValueError(
                f"StreamingFractalTokenizerV3.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor."
            )
        
        B, C, H, W = images.shape
        
        # 1. 提取共享特征图 (可被可学习分割器复用)
        features = self.shared_conv(images)  # [B, d_model, H/p, W/p]
        self._last_features = features
        
        # 2. Adaptive/Learnable Splitting
        if self._use_learnable_split:
            # 可学习分割: 使用特征图
            from .split_adaptive import LearnableSplitter
            assert isinstance(self.splitter, LearnableSplitter)
            split_results = self.splitter(
                features, 
                image_size=(H, W),
                hard=not self.training,  # 训练时用 soft，推理时用 hard
            )
        else:
            # 规则分割: 使用原始图像
            split_results = self.splitter.split_batch(images)
        
        self._last_split_stats = {
            'num_tokens': [sr.num_tokens for sr in split_results],
            'depth_distributions': [sr.depth_distribution for sr in split_results],
        }
        
        # 3. Hilbert-Native Patch Embedding (使用预计算的特征)
        tokens, levels_info = self._embed_with_features(features, split_results)
        
        # 4. 构建输出
        sequences = []
        for b in range(B):
            num_tokens = split_results[b].num_tokens
            seq = TokenSequence(
                tokens=tokens[b, :num_tokens],
                metadata={
                    "levels": levels_info[b, :num_tokens],
                    "split_stats": {
                        "num_tokens": num_tokens,
                        "depth_distribution": split_results[b].depth_distribution,
                    },
                },
            )
            sequences.append(seq)
        
        return TokenizerOutput(sequences)
    
    def _embed_with_features(
        self,
        features: torch.Tensor,
        split_results: List,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """使用预计算的特征进行嵌入 (避免重复计算 SharedConv).
        
        这是 patch_embed.forward() 的优化版本，跳过 SharedConv。
        """
        from .split_adaptive import SplitResult
        
        B = features.shape[0]
        device = features.device
        dtype = features.dtype
        dim = self.d_model
        
        # 确定最大 token 数量
        token_counts = [sr.num_tokens for sr in split_results]
        max_tokens = max(token_counts) if token_counts else 1
        total_tokens = sum(token_counts)
        
        if total_tokens == 0:
            tokens = torch.zeros(B, 1, dim, device=device, dtype=dtype)
            levels_info = torch.zeros(B, 1, self.max_depth + 1, dtype=torch.long, device=device)
            return self.patch_embed.norm(tokens), levels_info
        
        # 收集所有 region 的 boxes 和 depths
        all_boxes = []
        all_depths = []
        all_levels_info = []
        batch_indices = []
        token_indices = []
        
        p = self.base_patch_size
        for b, sr in enumerate(split_results):
            for i, token in enumerate(sr.tokens):
                fx1 = token.region.x1 / p
                fy1 = token.region.y1 / p
                fx2 = token.region.x2 / p
                fy2 = token.region.y2 / p
                
                fx2 = max(fx1 + 0.5, fx2)
                fy2 = max(fy1 + 0.5, fy2)
                
                all_boxes.append([b, fx1, fy1, fx2, fy2])
                all_depths.append(min(token.depth, self.max_depth))
                all_levels_info.append(token.to_levels_info(self.max_depth))
                batch_indices.append(b)
                token_indices.append(i)
        
        boxes_tensor = torch.tensor(all_boxes, device=device, dtype=dtype)
        depths_tensor = torch.tensor(all_depths, device=device, dtype=torch.long)
        
        # ROI-Align
        try:
            from torchvision.ops import roi_align
            pooled = roi_align(
                features,
                boxes_tensor,
                output_size=(1, 1),
                spatial_scale=1.0,
                aligned=True,
            ).squeeze(-1).squeeze(-1)
        except ImportError:
            pooled = self.patch_embed._fallback_roi_pool(
                features, boxes_tensor, 
                features.shape[2], features.shape[3]
            )
        
        # 深度编码
        scales = self.patch_embed.depth_scale[depths_tensor]
        embeds = self.patch_embed.depth_embed(depths_tensor)
        all_tokens = pooled * scales.unsqueeze(-1) + embeds
        
        # 分配到输出 buffer (P-PERF-3: 向量化分配)
        tokens = torch.zeros(B, max_tokens, dim, device=device, dtype=dtype)
        levels_info = torch.zeros(B, max_tokens, self.max_depth + 1, dtype=torch.long, device=device)
        
        # 使用高级索引进行向量化分配
        batch_idx_tensor = torch.tensor(batch_indices, device=device, dtype=torch.long)
        token_idx_tensor = torch.tensor(token_indices, device=device, dtype=torch.long)
        
        # 向量化 token 分配
        tokens[batch_idx_tensor, token_idx_tensor] = all_tokens
        
        # 向量化 levels_info 分配
        levels_info_tensor = torch.tensor(all_levels_info, device=device, dtype=torch.long)
        levels_info[batch_idx_tensor, token_idx_tensor] = levels_info_tensor
        
        return self.patch_embed.norm(tokens), levels_info
    
    def forward(self, images: torch.Tensor) -> TokenizerOutput:
        """前向传播，等价于 tokenize."""
        return self.tokenize(images)
    
    @torch.no_grad()
    def get_split_stats(self) -> Optional[Dict[str, Any]]:
        """获取最近一次分割的统计信息."""
        if self._last_split_stats is None:
            return None
        
        # 添加 depth_entropy 到统计信息
        stats = self._last_split_stats.copy()
        stats['depth_entropy'] = self.get_scale_entropy()
        return stats
    
    def get_entropy_loss(self) -> Optional[torch.Tensor]:
        """获取熵正则化损失 (用于可学习分割器)."""
        if not self._use_learnable_split:
            return None
        
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        
        if self._last_split_stats is None:
            return None
        
        # 构造 SplitResult 列表用于熵计算
        from .split_adaptive import SplitResult, SplitToken, Region
        results = []
        for i, dist in enumerate(self._last_split_stats['depth_distributions']):
            # 重建简化的 SplitResult
            tokens = []
            for depth, count in dist.items():
                for _ in range(count):
                    tokens.append(SplitToken(
                        region=Region(0, 0, 1, 1),  # 占位
                        depth=depth,
                        path=[],
                        hilbert_idx=0,
                        complexity=0.0,
                    ))
            results.append(SplitResult(tokens=tokens))
        
        return self.splitter.get_entropy_loss(results)
    
    def get_learnable_split_loss(
        self,
        lambda_entropy: float = 0.1,
        lambda_budget: float = 0.01,
        target_tokens: int = 64,
    ) -> Optional[torch.Tensor]:
        """获取可学习分割器的辅助损失.
        
        数学形式化:
            L_split = λ₁ · L_entropy + λ₂ · L_budget + L_reg
            
        其中:
            L_entropy = -H(depth_distribution)  # 鼓励多尺度
            L_budget = ReLU(N - N_target)²      # 预算约束
            L_reg = 阈值正则化                   # 防止坍塌
        
        Args:
            lambda_entropy: 熵损失权重
            lambda_budget: 预算损失权重  
            target_tokens: 目标 token 数
            
        Returns:
            可微分的辅助损失 (用于多任务训练)
        """
        if not self._use_learnable_split:
            return None
        
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        
        # 阈值正则化 (可微分)
        reg_loss = self.splitter.get_threshold_regularization_loss()
        
        # 深度损失 (需要特征图)
        if self._last_features is not None:
            depth_loss = self.splitter.get_differentiable_depth_loss(
                self._last_features, 
                self.image_size,
            )
        else:
            depth_loss = torch.tensor(0.0, device=reg_loss.device)
        
        return lambda_entropy * depth_loss + reg_loss
    
    def get_scale_entropy(self) -> Optional[float]:
        """获取尺度分布熵值."""
        if self._last_split_stats is None:
            return None
        
        total_dist: Dict[int, int] = {}
        for dist in self._last_split_stats['depth_distributions']:
            for d, count in dist.items():
                total_dist[d] = total_dist.get(d, 0) + count
        
        total = sum(total_dist.values())
        if total == 0:
            return None
        
        entropy = 0.0
        for count in total_dist.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log(p)
        
        return entropy
    
    @torch.no_grad()
    def compute_scale_distribution(self, images: torch.Tensor) -> Dict[str, Any]:
        """计算深度分布统计信息."""
        _ = self.tokenize(images)
        
        if self._last_split_stats is None:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }
        
        total_dist: Dict[int, int] = {}
        for dist in self._last_split_stats['depth_distributions']:
            for d, count in dist.items():
                total_dist[d] = total_dist.get(d, 0) + count
        
        total_tokens = sum(total_dist.values())
        if total_tokens == 0:
            return {
                'scale_ratios': {},
                'entropy': 0.0,
                'max_entropy': 0.0,
                'dominant_scale': self.base_patch_size,
                'depth_distribution': {},
            }
        
        scale_ratios: Dict[int, float] = {}
        for depth, count in total_dist.items():
            ps = self.base_patch_size * (2 ** (self.max_depth - depth))
            scale_ratios[ps] = count / total_tokens
        
        entropy = 0.0
        for count in total_dist.values():
            p = count / total_tokens
            if p > 0:
                entropy -= p * math.log(p)
        
        num_depths = self.max_depth + 1
        max_entropy = math.log(num_depths) if num_depths > 1 else 0.0
        
        dominant_depth = max(total_dist.keys(), key=lambda d: total_dist[d])
        dominant_scale = self.base_patch_size * (2 ** (self.max_depth - dominant_depth))
        
        return {
            'scale_ratios': scale_ratios,
            'entropy': entropy,
            'max_entropy': max_entropy,
            'dominant_scale': dominant_scale,
            'depth_distribution': total_dist,
        }
    
    def get_training_stats(self) -> Dict[str, Any]:
        """获取训练状态统计信息."""
        stats = {
            'tokenizer_version': 'v3_unified',
            'architecture': 'shared_conv + learnable_split + hilbert_embed',
            'split_scheme': 'learnable',  # Only LearnableSplitter remains
            'max_depth': self.max_depth,
            'learnable_split': True,  # Always true after Scheme B/C removal
        }
        
        if self._last_split_stats:
            avg_tokens = sum(self._last_split_stats['num_tokens']) / len(self._last_split_stats['num_tokens'])
            stats['avg_tokens_per_image'] = avg_tokens
            stats['depth_entropy'] = self.get_scale_entropy()
        
        # Always use LearnableSplitter
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        splitter_stats = self.splitter.get_split_statistics()
        stats['learnable_thresholds'] = splitter_stats['thresholds'].tolist()
        stats['learnable_temperature'] = splitter_stats['temperature'].item()
        
        return stats
    
    def set_split_temperature(self, temperature: float) -> None:
        """设置可学习分割器的温度 (用于退火调度).
        
        数学形式化:
            T(t) = T_start · (T_end / T_start)^(t / T_total)
        """
        if self._use_learnable_split:
            from .split_adaptive import LearnableSplitter
            assert isinstance(self.splitter, LearnableSplitter)
            self.splitter.set_temperature(temperature)
    
    def reset_split_statistics(self) -> None:
        """重置可学习分割器的统计信息."""
        if self._use_learnable_split:
            from .split_adaptive import LearnableSplitter
            assert isinstance(self.splitter, LearnableSplitter)
            self.splitter.reset_statistics()
    
    def get_temperature_scheduler(
        self,
        T_start: float = 1.0,
        T_end: float = 0.1,
        schedule: str = 'exponential',
        warmup_steps: int = 0,
    ):
        """获取温度退火调度器 (仅可学习分割器).
        
        数学形式化:
            T(t) = T_start · (T_end / T_start)^(t / total_steps)
            
        推荐参数 (基于梯度分析):
            T_start = 1.0: ∂p/∂C = 0.25 at C=τ，探索充分
            T_end = 0.1:   ∂p/∂C = 2.50 at C=τ，近确定性但不梯度消失
            
        用法:
            scheduler = tokenizer.get_temperature_scheduler()
            scheduler.set_total_steps(epochs * steps_per_epoch)
            
            for step in training_loop:
                loss = model(batch)
                ...
                scheduler.step()  # 自动更新温度
        
        Args:
            T_start: 初始温度 (默认 1.0)
            T_end: 最终温度 (默认 0.1)
            schedule: 调度策略 ('exponential', 'linear', 'cosine')
            warmup_steps: 热身步数，期间保持 T_start
            
        Returns:
            TemperatureScheduler 实例
            
        Raises:
            ValueError: 如果不是可学习分割器
        """
        from .split_adaptive import LearnableSplitter, TemperatureScheduler
        assert isinstance(self.splitter, LearnableSplitter)
        
        return TemperatureScheduler(
            splitter=self.splitter,
            T_start=T_start,
            T_end=T_end,
            schedule=schedule,
            warmup_steps=warmup_steps,
        )

    def get_multi_layer_depth_loss(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        max_eval_depth: Optional[int] = None,
        weight_decay_factor: float = 0.5,
        target_entropy: float = 0.693,
        return_details: bool = False,
    ):
        """获取多层可微分深度损失 (仅可学习分割器).
        
        数学形式化:
            L_multi = Σ_d w_d · (H_target - H̄_d)²
            
        为什么需要多层损失:
            单层损失仅在根区域评估，深层阈值 τ₁, τ₂, τ₃... 无梯度信号。
            多层损失在每层的规则网格评估，确保所有阈值可学习。
            
        梯度覆盖率 (β=0.5, D=3):
            d=0: 51.6%, d=1: 25.8%, d=2: 12.9%, d=3: 6.5%
            总覆盖: 96.8%
            
        用法:
            loss, details = tokenizer.get_multi_layer_depth_loss(
                features, image_size, return_details=True
            )
            
            # TensorBoard 可视化
            for d, (l, e, p) in enumerate(zip(
                details['loss_per_depth'],
                details['entropy_per_depth'], 
                details['p_split_per_depth']
            )):
                writer.add_scalar(f'depth/loss_d{d}', l, step)
                writer.add_scalar(f'depth/entropy_d{d}', e, step)
                writer.add_scalar(f'depth/p_split_d{d}', p, step)
                
        Args:
            features: [B, C, H', W'] 特征图
            image_size: (H, W) 图像尺寸，默认使用 self.image_size
            max_eval_depth: 最大评估深度 (默认 min(max_depth, 3))
            weight_decay_factor: 权重衰减因子 β (默认 0.5)
            target_entropy: 目标熵 (默认 0.693 = ln(2))
            return_details: 是否返回各层详情
            
        Returns:
            loss: 标量损失
            details (if return_details): 各层损失详情字典
            
        Raises:
            ValueError: 如果不是可学习分割器
        """
        from .split_adaptive import LearnableSplitter
        assert isinstance(self.splitter, LearnableSplitter)
        
        if image_size is None:
            image_size = self.image_size
        
        return self.splitter.get_multi_layer_depth_loss(
            features=features,
            image_size=image_size,
            max_eval_depth=max_eval_depth,
            weight_decay_factor=weight_decay_factor,
            target_entropy=target_entropy,
            return_details=return_details,
        )
