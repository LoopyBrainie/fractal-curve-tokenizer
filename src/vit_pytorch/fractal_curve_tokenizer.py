# -*- coding: utf-8 -*-
"""Fractal Hilbert Curve Tokenizer.

This module implements a hierarchical image tokenizer based on fractal 
partitioning and Hilbert curve traversal. It supports learnable split 
decisions using reinforcement learning (REINFORCE algorithm).
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import (
    DEFAULT_MAX_LEVEL,
    EXTRA_DEPTH_CAP,
    LOGITS_CLAMP_MAX,
    LOGITS_CLAMP_MIN,
)
from .hilbert import HilbertCurve, get_quadrant_order
from .tokenization import BaseTokenizer, TokenSequence, TokenizerOutput
from .utils import sanitize_tensor

logger = logging.getLogger(__name__)


@dataclass
class PatchInfo:
    """追踪 BFS 中每个 patch 的元信息"""
    patch: torch.Tensor      # [C, H, W]
    level: int               # 当前递归深度
    coord: List[int]         # Hilbert 路径
    dfs_order: float         # DFS 顺序索引，用于最终排序
    

class MiniCNN(nn.Module):
    """轻量级CNN特征提取器，用于分割决策。
    
    .. deprecated:: 0.3.0
        MiniCNN 已废弃，消融实验表明其增加训练开销 45% 但无准确率收益。
        建议使用 `use_cnn=False`（默认值），仅使用 6 维手工特征进行分割决策。
        此类将在 v1.0 中移除。
    
    将输入 patch 通过两层卷积和全局池化转换为固定维度的特征向量。
    
    Args:
        in_channels: 输入通道数（RGB=3，灰度=1）
        hidden_dim: 隐藏层维度
        out_dim: 输出特征维度
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 16,
        out_dim: int = 32,
    ) -> None:
        warnings.warn(
            "MiniCNN 已废弃，消融实验表明其增加训练开销 45% 但无准确率收益。"
            "建议使用 use_cnn=False（默认值）。此类将在 v1.0 中移除。",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__()
        self.net = nn.Sequential(
            nn.InstanceNorm2d(in_channels), # 归一化输入，防止数值过大
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, stride=2),  # 下采样
            nn.ReLU(),
            nn.InstanceNorm2d(hidden_dim), # 中间层归一化
            nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1, stride=2),  # 再次下采样
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局池化
            nn.Flatten(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            x: 输入张量，形状为 [B, C, H, W] 或 [C, H, W]
            
        Returns:
            特征向量，形状为 [B, out_dim]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return self.net(x)


class LearnableSplitDecision(nn.Module):
    """可学习分割决策网络。
    
    该网络接收 patch 的手工特征（层级、尺寸、方差等），可选地加入 CNN 特征，
    输出两个 logits 表示「不分割」和「分割」的倾向。
    
    Args:
        patch_features: 手工特征维度（默认6：level, height, width, variance, mean, edge_density）
        cnn_features: CNN 提取的特征维度，0 表示不使用 CNN
        hidden_dim: 隐藏层维度
    """

    def __init__(
        self,
        patch_features: int = 6,
        cnn_features: int = 0,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        # patch_features: [level, height, width, variance, mean, edge_density]
        # cnn_features: 来自MiniCNN的特征维度，0 表示不使用
        self.cnn_features = cnn_features
        input_dim = patch_features + cnn_features
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 2),  # 输出2个logits: [not_split, split]
        )

        # 初始化偏置，让初始分割倾向于继续分割 (index 1)
        with torch.no_grad():
            self.net[-1].bias[1] += 2.0

    def forward(self, combined_features: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        Args:
            combined_features: 组合特征，形状为 [batch_size, total_features]
            
        Returns:
            logits，形状为 [batch_size, 2]，分别表示不分割和分割的倾向
        """
        return self.net(combined_features)


class FractalHilbertTokenizer(BaseTokenizer):
    """基于分形 Hilbert 曲线的图像 Tokenizer。
    
    该 Tokenizer 使用自适应分形分割将图像递归分解为多尺度 tokens，
    并按 Hilbert 曲线顺序遍历以保持空间局部性。
    
    Attributes:
        min_patch_size: 最小 patch 尺寸 (height, width)
        max_level: 最大递归层数
        learnable_split: 是否使用可学习的分割决策
        adaptive_threshold: 自适应分割的方差阈值
        channels: 输入图像通道数
        use_cnn: 是否使用 CNN 特征（默认 False，用于 benchmark 对比）
        cnn_encoder: CNN 特征提取器（仅 use_cnn=True 时）
        split_decision: 分割决策网络（仅 learnable_split=True 时）
    """
    
    def __init__(
        self,
        min_patch_size: Tuple[int, int] = (1, 1),
        max_level: Optional[int] = None,
        learnable_split: bool = True,
        adaptive_threshold: float = 0.5,
        channels: int = 3,
        use_cnn: bool = False,
    ) -> None:
        """初始化 FractalHilbertTokenizer。
        
        Args:
            min_patch_size: 最小整数 patch 尺寸 (min_h, min_w)，默认到像素级别
            max_level: 最大递归层数，None 表示无限制（只受 min_patch_size 限制）
            learnable_split: 是否使用可学习的分割决策
            adaptive_threshold: 自适应分割阈值
            channels: 输入图像的通道数（RGB=3，灰度=1）
            use_cnn: 是否使用 CNN 特征提取（默认 False，仅用手工特征）
        """
        super().__init__()
        self.min_patch_size = min_patch_size
        self.max_level = max_level  # 可以为None，表示无限制
        self.learnable_split = learnable_split
        self.adaptive_threshold = adaptive_threshold
        self.channels = channels
        self.use_cnn = use_cnn

        if learnable_split:
            # CNN 特征提取器（已废弃，默认禁用）
            if use_cnn:
                warnings.warn(
                    "use_cnn=True 已废弃。消融实验表明 CNN 特征增加 45% 训练开销但无准确率收益，"
                    "且降低 Token 自适应性（范围从 60 降至 33）。建议使用 use_cnn=False（默认值）。"
                    "此参数将在 v1.0 中移除。",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self.cnn_encoder: Optional[MiniCNN] = MiniCNN(in_channels=channels, hidden_dim=16, out_dim=32)
                cnn_dim = 32
            else:
                self.cnn_encoder = None
                cnn_dim = 0
            # 分割决策网络
            self.split_decision: Optional[LearnableSplitDecision] = LearnableSplitDecision(
                patch_features=6, cnn_features=cnn_dim, hidden_dim=64
            )
        else:
            self.cnn_encoder = None
            self.split_decision = None

        # 用于存储REINFORCE所需的log_probs
        self.saved_log_probs: List[torch.Tensor] = []
        self.saved_entropies: List[torch.Tensor] = []
        
        # 预注册 Sobel 核为 buffer，避免每次调用时重复创建
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def clear_saved_actions(self) -> None:
        """清空保存的动作记录（log_probs 和 entropies）。"""
        self.saved_log_probs = []
        self.saved_entropies = []

    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """将图像批次转换为 token 序列。
        
        Args:
            images: 输入图像批次，形状为 [B, C, H, W]
            
        Returns:
            TokenizerOutput 包含每个图像的 token 序列
            
        Raises:
            ValueError: 如果输入形状不是 4D 张量
        """
        if images.dim() != 4:
            raise ValueError(
                f"FractalHilbertTokenizer.tokenize expects 4D input [B, C, H, W], "
                f"got {images.dim()}D tensor with shape {tuple(images.shape)}. "
                f"Hint: Use images.unsqueeze(0) for single image input."
            )

        batch_size, channels, height, width = images.shape
        estimated_max_level = self._estimate_max_possible_level(height, width)
        dynamic_depth_cap = self.max_level if self.max_level is not None else max(estimated_max_level + 5, 12)

        if self.learnable_split and self.split_decision is not None:
            # 确保可学习分割网络与输入位于同一设备
            self.split_decision = self.split_decision.to(images.device)
            if self.use_cnn and self.cnn_encoder is not None:
                self.cnn_encoder = self.cnn_encoder.to(images.device)

        # 清空之前的动作记录
        self.clear_saved_actions()

        min_patch_dim = max(1, min(self.min_patch_size))
        longest_edge = max(height, width)
        additional_levels = int(math.ceil(math.log2(longest_edge / min_patch_dim))) if min_patch_dim > 0 else 0
        max_info_len = max(dynamic_depth_cap + additional_levels + 4, 16)

        sequences = []

        flattened_patch_dim = channels * self.min_patch_size[0] * self.min_patch_size[1]

        # 根据是否启用可学习分割决定使用批处理还是递归
        use_batch = self.learnable_split and self.split_decision is not None
        
        for b in range(batch_size):
            image = images[b]
            sample_device = image.device

            if use_batch:
                # 使用BFS批处理版本 - 减少GPU kernel调用
                tokens_raw, levels_raw = self.fractal_partition_batched(
                    image,
                    max_info_len=max_info_len,
                    depth_limit=dynamic_depth_cap,
                )
            else:
                # 使用原始递归版本 - 用于无可学习分割的场景
                tokens_raw, levels_raw = self.fractal_partition(
                    image,
                    level=0,
                    coord=[],
                    max_info_len=max_info_len,
                    depth_limit=dynamic_depth_cap,
                )

            if len(tokens_raw) == 0:
                logger.warning(
                    "Empty token sequence generated for image %d (shape: %s)",
                    b, tuple(image.shape)
                )
                empty_tokens = torch.empty(0, flattened_patch_dim, device=sample_device)
                empty_levels = torch.empty(0, max_info_len, dtype=torch.long, device=sample_device)
                sequences.append(TokenSequence(tokens=empty_tokens, metadata={"levels": empty_levels}))
                continue

            tokens = torch.stack(tokens_raw)
            if tokens.device != sample_device:
                tokens = tokens.to(sample_device)

            levels = torch.tensor(levels_raw, dtype=torch.long, device=sample_device)
            sequences.append(TokenSequence(tokens=tokens, metadata={"levels": levels}))

        return TokenizerOutput(sequences)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """前向传播，返回 legacy 格式的输出。
        
        Args:
            images: 输入图像批次，形状为 [B, C, H, W]
            
        Returns:
            (tokens, levels) 元组，与 tokenize_legacy 相同
        """
        output = self.tokenize(images)
        legacy = output.to_legacy()
        return legacy.tokens, legacy.levels

    def tokenize_legacy(
        self, images: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """将图像转换为 token 序列（legacy 格式）。
        
        Args:
            images: 输入图像批次，形状为 [B, C, H, W]
            
        Returns:
            (tokens, levels) 元组：
            - tokens: 每个图像的 token 张量列表
            - levels: 每个图像的层级信息张量列表
        """
        output = self.tokenize(images)
        legacy = output.to_legacy()
        return legacy.tokens, legacy.levels

    def _estimate_max_possible_level(self, h: int, w: int) -> int:
        """估算给定尺寸下可能达到的最大层级。
        
        Args:
            h: 图像高度
            w: 图像宽度
            
        Returns:
            最大可能的递归层级
        """
        min_h, min_w = self.min_patch_size
        max_level_h = 0
        max_level_w = 0

        # 计算高度方向最大可能层级
        temp_h = h
        while temp_h > min_h:
            temp_h = temp_h // 2
            max_level_h += 1
            if temp_h <= 0:
                break

        # 计算宽度方向最大可能层级
        temp_w = w
        while temp_w > min_w:
            temp_w = temp_w // 2
            max_level_w += 1
            if temp_w <= 0:
                break

        return max(max_level_h, max_level_w)

    def _decide_should_stop(
        self,
        patch: torch.Tensor,
        level: int,
        depth_limit: int,
        can_split: bool,
    ) -> bool:
        """决定是否应该停止分割。
        
        Args:
            patch: 输入 patch [C, H, W]
            level: 当前递归层级
            depth_limit: 最大递归深度
            can_split: 是否可以物理分割
            
        Returns:
            True 表示应该停止分割
        """
        C, H, W = patch.shape
        
        # 基本停止条件
        level_limit_reached = level >= depth_limit
        basic_stop = not can_split or level_limit_reached

        # 安全检查：patch 太小或层级太深
        extra_depth_cap = depth_limit + EXTRA_DEPTH_CAP
        safety_check = (H <= 1 and W <= 1) or (level >= extra_depth_cap)

        if basic_stop or safety_check:
            if level_limit_reached:
                logger.debug(
                    "Depth limit reached at level %d (limit: %d), patch size: %dx%d",
                    level, depth_limit, H, W
                )
            return True

        # 使用可学习分割决策
        if self.learnable_split and self.split_decision is not None:
            features = self._extract_enhanced_patch_features(patch, level, H, W)
            
            # 可选：添加 CNN 特征
            if self.use_cnn and self.cnn_encoder is not None:
                # CNN 需要最小尺寸 (InstanceNorm 要求空间尺寸 > 1)
                MIN_CNN_SIZE = 4
                if H >= MIN_CNN_SIZE and W >= MIN_CNN_SIZE:
                    cnn_feat = self.cnn_encoder(patch)
                else:
                    cnn_feat = torch.zeros(1, 32, device=patch.device)
                combined = torch.cat([features, cnn_feat], dim=1)
            else:
                combined = features
            
            combined = sanitize_tensor(combined)

            logits = self.split_decision(combined)
            logits = torch.clamp(logits, min=LOGITS_CLAMP_MIN, max=LOGITS_CLAMP_MAX)
            
            if self.training:
                logits = sanitize_tensor(
                    logits, posinf_value=LOGITS_CLAMP_MAX, neginf_value=LOGITS_CLAMP_MIN
                )

                dist = torch.distributions.Categorical(logits=logits)
                action = dist.sample()
                
                self.saved_log_probs.append(dist.log_prob(action))
                self.saved_entropies.append(dist.entropy())
                
                return action.item() == 0
            else:
                action = torch.argmax(logits, dim=-1)
                return action.item() == 0
        else:
            # 默认策略
            if self.adaptive_threshold is not None and self.adaptive_threshold > 0:
                patch_var = torch.var(patch)
                return bool(patch_var.item() < self.adaptive_threshold)
            else:
                return level >= DEFAULT_MAX_LEVEL

    def fractal_partition(
        self,
        patch: torch.Tensor,
        level: int,
        coord: List[int],
        max_info_len: int,
        depth_limit: int,
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """递归执行分形分割。
        
        Args:
            patch: 输入 patch，形状为 [C, H, W]
            level: 当前递归层级
            coord: 当前 Hilbert 路径坐标
            max_info_len: levels_info 的最大长度
            depth_limit: 最大递归深度
            
        Returns:
            (tokens, levels) 元组：
            - tokens: 扁平化的 token 张量列表
            - levels: 层级信息列表（每个元素为 int 列表）
        """
        C, H, W = patch.shape
        min_h, min_w = self.min_patch_size
        tokens: List[torch.Tensor] = []
        levels: List[List[int]] = []

        can_split_h = H > min_h
        can_split_w = W > min_w
        can_split = can_split_h or can_split_w

        should_stop = self._decide_should_stop(patch, level, depth_limit, can_split)

        if should_stop:
            return self._create_token_output(patch, level, coord, max_info_len)

        sub_patches = self._adaptive_split(patch, H, W, can_split_h, can_split_w)

        if not sub_patches:
            return self._create_token_output(patch, level, coord, max_info_len)

        traversal_order = self._determine_traversal_order(level, H, W, len(sub_patches))

        for patch_idx in traversal_order:
            if patch_idx < len(sub_patches):
                sub = sub_patches[patch_idx].contiguous()
                if sub.numel() > 0:
                    tks, lvls = self.fractal_partition(
                        sub,
                        level + 1,
                        coord + [patch_idx],
                        max_info_len,
                        depth_limit,
                    )
                    tokens.extend(tks)
                    levels.extend(lvls)
        return tokens, levels

    def _create_token_output(
        self,
        patch: torch.Tensor,
        level: int,
        coord: List[int],
        max_info_len: int,
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """将 patch 转换为 token 输出。
        
        Args:
            patch: 输入 patch [C, H, W]
            level: 当前层级
            coord: Hilbert 路径坐标
            max_info_len: levels_info 的最大长度
            
        Returns:
            (tokens, levels) 单元素列表
        """
        _, H, W = patch.shape
        processed_patch = self._process_patch_to_fixed_size(patch, H, W)
        patch_flat = processed_patch.reshape(-1)

        path = coord.copy()
        padded = [level] + path + [0] * (max_info_len - 1 - len(path))
        
        return [patch_flat], [padded[:max_info_len]]

    def _determine_traversal_order(
        self, level: int, h: int, w: int, num_patches: int
    ) -> List[int]:
        """确定子 patch 的遍历顺序。
        
        Args:
            level: 当前递归层级
            h: patch 高度
            w: patch 宽度
            num_patches: 子 patch 数量
            
        Returns:
            遍历顺序索引列表
        """
        if num_patches == 4:
            # 使用统一的 Hilbert 模块获取遍历顺序
            return get_quadrant_order(level, h, w)

        if num_patches == 2:
            # 对于高度分割，按从上到下；宽度分割，从左到右
            return [0, 1]

        return list(range(num_patches))

    def _adaptive_split(
        self,
        patch: torch.Tensor,
        h: int,
        w: int,
        can_split_h: bool,
        can_split_w: bool,
    ) -> List[torch.Tensor]:
        """自适应分割 patch。
        
        根据 patch 的高度和宽度是否可分割，决定分割方式：
        - 两个维度都可分割：四分法
        - 只有高度可分割：水平二分法
        - 只有宽度可分割：垂直二分法
        
        Args:
            patch: 输入 patch，形状为 [C, H, W]
            h: patch 高度
            w: patch 宽度
            can_split_h: 高度是否可分割
            can_split_w: 宽度是否可分割
            
        Returns:
            子 patch 列表
        """
        if can_split_h and can_split_w:
            return self._intelligent_quadrant_split(patch, h, w)

        if can_split_h:
            split_index = self._calculate_split_index(h, self.min_patch_size[0], w)
            split_index = max(1, min(h - 1, split_index))
            top = patch[:, :split_index, :]
            bottom = patch[:, split_index:, :]
            return [top, bottom]

        if can_split_w:
            split_index = self._calculate_split_index(w, self.min_patch_size[1], h)
            split_index = max(1, min(w - 1, split_index))
            left = patch[:, :, :split_index]
            right = patch[:, :, split_index:]
            return [left, right]

        return []

    def _extract_enhanced_patch_features(
        self, patch: torch.Tensor, level: int, h: int, w: int
    ) -> torch.Tensor:
        """提取增强的 patch 特征用于分割决策。
        
        Args:
            patch: 输入 patch，形状为 [C, H, W]
            level: 当前递归层级
            h: patch 高度
            w: patch 宽度
            
        Returns:
            特征向量，形状为 [1, 6]
        """
        import torch.nn.functional as F
        device = patch.device

        # 基础统计特征
        patch_var = torch.var(patch)
        patch_mean = torch.mean(patch)

        # 边缘密度特征（使用预注册的Sobel算子）
        gray_patch = torch.mean(patch, dim=0, keepdim=True).unsqueeze(0)  # [1, 1, H, W]

        if h >= 3 and w >= 3:  # 确保patch足够大来应用卷积
            # 确保 Sobel 核与输入在同一设备
            sobel_x = self.sobel_x.to(device)
            sobel_y = self.sobel_y.to(device)
            edge_x = F.conv2d(gray_patch, sobel_x, padding=1)
            edge_y = F.conv2d(gray_patch, sobel_y, padding=1)
            edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
            edge_density = torch.mean(edge_magnitude)
        else:
            edge_density = torch.tensor(0.0, device=device)

        # 纹理复杂度（基于局部方差）
        if h >= 2 and w >= 2:
            local_patches = F.unfold(gray_patch, kernel_size=2, stride=1)  # 2x2局部patch
            local_vars = torch.var(local_patches, dim=1)
            texture_complexity = torch.mean(local_vars)
        else:
            texture_complexity = patch_var

        # 关键修复：特征归一化和对数变换，防止数值过大
        # 1. 归一化尺寸和层级
        norm_level = float(level) / 10.0  # 假设最大层级约10
        norm_h = float(h) / 256.0         # 假设最大尺寸约256
        norm_w = float(w) / 256.0

        # 2. 对数变换处理方差和边缘密度（这些值可能跨度很大）
        log_var = torch.log1p(patch_var).clamp(max=10.0)
        log_edge = torch.log1p(edge_density).clamp(max=10.0)
        
        # 3. 均值归一化 (假设输入已经大致在0-1或-1-1之间，但为了保险起见)
        norm_mean = torch.clamp(patch_mean, -3.0, 3.0)

        # 构建特征向量: [level, height, width, variance, mean, edge_density]
        feature_values = torch.stack(
            [
                torch.tensor(norm_level, device=device),
                torch.tensor(norm_h, device=device),
                torch.tensor(norm_w, device=device),
                log_var.float(),
                norm_mean.float(),
                log_edge.float(),
            ]
        )

        features = feature_values.view(1, -1)

        return features

    def _process_patch_to_fixed_size(
        self, patch: torch.Tensor, h: int, w: int
    ) -> torch.Tensor:
        """将 patch 处理为固定尺寸，支持任意输入尺寸。
        
        使用 padding 和裁剪/插值将 patch 调整到 min_patch_size。
        
        Args:
            patch: 输入 patch，形状为 [C, H, W]
            h: 当前 patch 高度
            w: 当前 patch 宽度
            
        Returns:
            调整后的 patch，形状为 [C, min_h, min_w]
        """
        min_h, min_w = self.min_patch_size

        if h == min_h and w == min_w:
            return patch

        # 先padding到至少目标大小，避免整除导致的尺寸丢失
        if h < min_h or w < min_w:
            pad_h_total = max(0, min_h - h)
            pad_w_total = max(0, min_w - w)

            pad_top = pad_h_total // 2
            pad_bottom = pad_h_total - pad_top
            pad_left = pad_w_total // 2
            pad_right = pad_w_total - pad_left

            patch = F.pad(patch, (pad_left, pad_right, pad_top, pad_bottom))
            h = patch.shape[-2]
            w = patch.shape[-1]

        if h == min_h and w == min_w:
            return patch

        if h > min_h or w > min_w:
            start_h = max(0, (h - min_h) // 2)
            end_h = start_h + min_h
            start_w = max(0, (w - min_w) // 2)
            end_w = start_w + min_w
            patch = patch[:, start_h:end_h, start_w:end_w]

        # 再次确保尺寸一致（整数除法可能导致偏差）
        if patch.shape[-2:] != (min_h, min_w):
            patch = F.interpolate(
                patch.unsqueeze(0),
                size=(min_h, min_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return patch

    def _intelligent_quadrant_split(
        self, patch: torch.Tensor, h: int, w: int
    ) -> List[torch.Tensor]:
        """智能四分法分割，确保每个子 patch 都是整数尺寸。
        
        Args:
            patch: 输入 patch，形状为 [C, H, W]
            h: patch 高度
            w: patch 宽度
            
        Returns:
            四个子 patch 的列表，顺序为 [左上, 右上, 左下, 右下]
        """
        mid_h = self._calculate_split_index(h, self.min_patch_size[0], w)
        mid_w = self._calculate_split_index(w, self.min_patch_size[1], h)

        mid_h = max(1, min(h - 1, mid_h))
        mid_w = max(1, min(w - 1, mid_w))

        sub_patches = [
            patch[:, :mid_h, :mid_w],
            patch[:, :mid_h, mid_w:],
            patch[:, mid_h:, :mid_w],
            patch[:, mid_h:, mid_w:],
        ]

        return sub_patches

    def _calculate_split_index(
        self, length: int, min_size: int, secondary_length: int
    ) -> int:
        """计算分割索引位置。
        
        根据当前维度长度、最小尺寸和另一维度长度，计算最佳分割位置。
        
        Args:
            length: 当前维度长度
            min_size: 该维度的最小尺寸
            secondary_length: 另一维度的长度（用于宽高比调整）
            
        Returns:
            分割索引位置
        """
        if length <= 1:
            return 1

        if length <= min_size:
            return length - 1 if length > 1 else 1

        if length <= min_size * 2:
            candidate = length // 2
        else:
            aspect_ratio = length / max(secondary_length, 1)
            if aspect_ratio > 1.5:
                ratio = 0.5 + min(0.2, 0.1 * (aspect_ratio - 1.5))
            elif aspect_ratio < 0.67:
                ratio = 0.5 - min(0.2, 0.1 * (0.67 - aspect_ratio) / max(aspect_ratio, 1e-3))
            else:
                ratio = 0.5

            candidate = int(round(length * ratio))

        lower_bound = max(1, min_size)
        upper_bound = max(lower_bound, length - lower_bound)
        candidate = max(lower_bound, min(candidate, upper_bound))
        return candidate

    # 注意: Hilbert 曲线相关方法已迁移至 hilbert.py 模块
    # 使用 from .hilbert import HilbertCurve, get_quadrant_order

    def default_should_split(
        self, patch: torch.Tensor, level: int
    ) -> bool:
        """默认分割策略：只看层数和 patch 大小。
        
        Args:
            patch: 输入 patch，形状为 [C, H, W]
            level: 当前递归层级
            
        Returns:
            是否应该继续分割
        """
        _channels, height, width = patch.shape
        min_h, min_w = self.min_patch_size
        max_level = self.max_level if self.max_level is not None else 50
        return level < max_level and (height > min_h or width > min_w)

    # ==================== 批处理优化方法 ====================
    
    def fractal_partition_batched(
        self,
        patch: torch.Tensor,
        max_info_len: int,
        depth_limit: int,
    ) -> Tuple[List[torch.Tensor], List[List[int]]]:
        """
        批处理版本的分形分割，使用 BFS 实现层级批处理。
        
        核心优化：对同一层级的所有 patches 批量计算 CNN 特征和分割决策，
        而非逐个递归计算。保持 Hilbert 曲线遍历顺序不变。
        
        Args:
            patch: 输入图像 [C, H, W]
            max_info_len: levels_info 的最大长度
            depth_limit: 最大递归深度
            
        Returns:
            (tokens, levels): 与原 fractal_partition 相同的输出格式
        """
        device = patch.device
        min_h, min_w = self.min_patch_size
        
        # 初始化 BFS 队列
        # dfs_order 用于最终排序，初始为 0
        queue: List[PatchInfo] = [PatchInfo(patch=patch, level=0, coord=[], dfs_order=0.0)]
        
        # 最终输出的 token 列表（需要按 dfs_order 排序）
        final_tokens: List[Tuple[float, torch.Tensor, List[int]]] = []
        
        while queue:
            # 按层级分组，批量处理同一层的 patches
            current_level = queue[0].level
            
            # 收集当前层的所有 patches
            current_batch: List[PatchInfo] = []
            remaining: List[PatchInfo] = []
            
            for info in queue:
                if info.level == current_level:
                    current_batch.append(info)
                else:
                    remaining.append(info)
            
            queue = remaining
            
            if not current_batch:
                continue
            
            # 批量决策：哪些 patches 应该分割
            split_decisions = self._batch_decide_splits(current_batch, depth_limit)
            
            # 处理每个 patch
            for i, info in enumerate(current_batch):
                should_split = split_decisions[i]
                C, H, W = info.patch.shape
                
                can_split_h = H > min_h
                can_split_w = W > min_w
                can_split = can_split_h or can_split_w
                
                # 如果不能分割或决策为停止，输出为 token
                if not can_split or not should_split:
                    processed = self._process_patch_to_fixed_size(info.patch, H, W)
                    token = processed.reshape(-1)
                    
                    # 构建 levels_info
                    depth = info.level
                    path = info.coord.copy()
                    padded = [depth] + path + [0] * (max_info_len - 1 - len(path))
                    levels = padded[:max_info_len]
                    
                    final_tokens.append((info.dfs_order, token, levels))
                else:
                    # 分割并加入队列
                    sub_patches = self._adaptive_split(info.patch, H, W, can_split_h, can_split_w)
                    
                    if not sub_patches:
                        # 分割失败，作为 token 输出
                        processed = self._process_patch_to_fixed_size(info.patch, H, W)
                        token = processed.reshape(-1)
                        depth = info.level
                        path = info.coord.copy()
                        padded = [depth] + path + [0] * (max_info_len - 1 - len(path))
                        levels = padded[:max_info_len]
                        final_tokens.append((info.dfs_order, token, levels))
                    else:
                        # 获取 Hilbert 遍历顺序
                        traversal_order = self._determine_traversal_order(
                            info.level, H, W, len(sub_patches)
                        )
                        
                        # 计算子 patch 的 DFS 顺序
                        # 使用分数索引保持正确的 DFS 顺序
                        num_children = len(sub_patches)
                        for child_idx, patch_idx in enumerate(traversal_order):
                            if patch_idx < len(sub_patches):
                                sub = sub_patches[patch_idx].contiguous()
                                if sub.numel() > 0:
                                    # DFS 顺序：父节点顺序 + 子节点在遍历中的位置
                                    child_dfs_order = info.dfs_order + child_idx / (num_children * (10 ** (info.level + 1)))
                                    new_info = PatchInfo(
                                        patch=sub,
                                        level=info.level + 1,
                                        coord=info.coord + [patch_idx],
                                        dfs_order=child_dfs_order,
                                    )
                                    queue.append(new_info)
        
        # 按 DFS 顺序排序
        final_tokens.sort(key=lambda x: x[0])
        
        # 提取排序后的 tokens 和 levels
        sorted_tokens = [item[1] for item in final_tokens]
        sorted_levels = [item[2] for item in final_tokens]
        
        return sorted_tokens, sorted_levels
    
    def _batch_decide_splits(
        self,
        batch: List[PatchInfo],
        depth_limit: int,
    ) -> List[bool]:
        """
        批量决定一组 patches 是否应该分割。
        
        对于 learnable_split=True，批量计算 CNN 特征和分割网络输出。
        对于非学习模式，使用方差阈值。
        
        Args:
            batch: 同一层级的 PatchInfo 列表
            depth_limit: 最大深度限制
            
        Returns:
            布尔列表，True 表示应该分割
        """
        decisions: List[bool] = []
        device = batch[0].patch.device if batch else None
        min_h, min_w = self.min_patch_size
        
        # 收集可分割的 patches 用于批量处理
        splittable_indices: List[int] = []
        splittable_patches: List[PatchInfo] = []
        
        for i, info in enumerate(batch):
            C, H, W = info.patch.shape
            can_split_h = H > min_h
            can_split_w = W > min_w
            can_split = can_split_h or can_split_w
            level_limit_reached = info.level >= depth_limit
            
            # 安全检查
            extra_depth_cap = depth_limit + 5
            safety_check = (H <= 1 and W <= 1) or (info.level >= extra_depth_cap)
            
            if not can_split or level_limit_reached or safety_check:
                decisions.append(False)
            else:
                splittable_indices.append(i)
                splittable_patches.append(info)
                decisions.append(True)  # 暂时标记为 True，后面更新
        
        if not splittable_patches:
            return decisions
        
        # 批量决策
        if self.learnable_split and self.split_decision is not None:
            # 批量计算特征和分割决策
            batch_decisions = self._batch_learnable_decision(splittable_patches)
            
            # 更新决策
            for idx, decision in zip(splittable_indices, batch_decisions):
                decisions[idx] = decision
        else:
            # 非学习模式：使用方差阈值
            for idx, info in zip(splittable_indices, splittable_patches):
                if self.adaptive_threshold is not None and self.adaptive_threshold > 0:
                    patch_var = torch.var(info.patch)
                    should_stop = patch_var < self.adaptive_threshold
                    decisions[idx] = not should_stop
                else:
                    # 默认继续分割，但有层级限制
                    decisions[idx] = info.level < 10
        
        return decisions
    
    def _batch_learnable_decision(
        self,
        patches: List[PatchInfo],
    ) -> List[bool]:
        """
        使用学习网络批量决定分割。
        
        批处理优化：将多个 patches 的特征提取合并为批量操作。
        
        Args:
            patches: 待决策的 PatchInfo 列表
            
        Returns:
            布尔列表，True 表示应该分割
        """
        if not patches:
            return []
        
        device = patches[0].patch.device
        batch_size = len(patches)
        
        # 1. 批量提取手工特征 (这部分仍需逐个计算，因为尺寸可能不同)
        features_list = []
        for info in patches:
            C, H, W = info.patch.shape
            features = self._extract_enhanced_patch_features(info.patch, info.level, H, W)
            features_list.append(features)
        
        # Stack features: [batch_size, 6]
        batch_features = torch.cat(features_list, dim=0)  # [batch_size, 6]
        
        # 2. 可选：批量计算 CNN 特征
        if self.use_cnn and self.cnn_encoder is not None:
            # 由于 patches 尺寸可能不同，需要逐个处理
            # CNN 需要最小尺寸 (InstanceNorm 要求空间尺寸 > 1)
            MIN_CNN_SIZE = 4
            cnn_features_list = []
            for info in patches:
                C, H, W = info.patch.shape
                if H >= MIN_CNN_SIZE and W >= MIN_CNN_SIZE:
                    cnn_feat = self.cnn_encoder(info.patch)  # [1, 32]
                else:
                    # Patch 太小，使用零填充代替 CNN 特征
                    cnn_feat = torch.zeros(1, 32, device=device)
                cnn_features_list.append(cnn_feat)
            batch_cnn_features = torch.cat(cnn_features_list, dim=0)  # [batch_size, 32]
            combined = torch.cat([batch_features, batch_cnn_features], dim=1)  # [batch_size, 38]
        else:
            combined = batch_features  # [batch_size, 6]
        
        combined = sanitize_tensor(combined)
        
        # 4. 批量获取 logits
        assert self.split_decision is not None  # 类型守卫
        logits = self.split_decision(combined)  # [batch_size, 2]
        logits = torch.clamp(logits, min=LOGITS_CLAMP_MIN, max=LOGITS_CLAMP_MAX)
        logits = sanitize_tensor(
            logits, posinf_value=LOGITS_CLAMP_MAX, neginf_value=LOGITS_CLAMP_MIN
        )
        
        # 5. 决策
        decisions: List[bool] = []
        
        if self.training:
            # REINFORCE: 采样动作
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()  # [batch_size]
            
            # 保存 log_probs 和 entropies (批量)
            log_probs = dist.log_prob(actions)  # [batch_size]
            entropies = dist.entropy()  # [batch_size]
            
            for i in range(batch_size):
                self.saved_log_probs.append(log_probs[i])
                self.saved_entropies.append(entropies[i])
                # action 0: stop (not split), action 1: split
                decisions.append(actions[i].item() == 1)
        else:
            # 推理模式：取最大概率
            actions = torch.argmax(logits, dim=-1)  # [batch_size]
            for i in range(batch_size):
                decisions.append(actions[i].item() == 1)
        
        return decisions
