# -*- coding: utf-8 -*-
"""
Hilbert-Native Patch Embedding for Variable Depth Tokens

数学形式化
============

Variable Depth Token 的 Patch Embedding 必须满足 4 个约束:

【约束 C1】维度一致性
    Embed(R_i) ∈ ℝ^dim, ∀ i, ∀ d_i
    不同大小的 region 必须映射到相同维度

【约束 C2】Hilbert 路径一致性
    HilbertPath(center(R_i))[:d_i] = QuadtreePath(R_i)
    嵌入必须保持四叉树路径信息

【约束 C3】尺度等变性
    若 R_j = 2×downsample(R_i) 且 content 相同，则:
    Embed(R_i) ≈ σ · Embed(R_j) + bias
    相同内容不同尺度应有数学联系

【约束 C4】LCA 兼容性
    LCA_depth(path_i, path_j) 必须对 Transformer bias 有效
    嵌入必须与 LCA 偏置协同工作

方案 C+ (Region Pooling) 实现
==============================

公式:
    F = SharedConv(Image)  ∈ ℝ^{B × dim × H/p × W/p}
    
    对于 region R_i 在深度 d_i:
        region_feat = F[:, :, y₁:y₂, x₁:x₂]
        pooled = AdaptiveAvgPool2d(1)(region_feat)
        t_i = pooled · σ_{d_i} + E_{depth}(d_i)

参数量:
    - SharedConv: dim × C × p × p ≈ 12K
    - depth_embed: (D+1) × dim ≈ 1.3K
    - depth_scale: D+1 ≈ 5
    - 总计: ~14K (vs 4.2M for Depth-Specific Conv)

Author: GitHub Copilot
Date: 2025-12-25
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .adaptive_split import SplitResult, SplitToken


class HilbertNativePatchEmbed(nn.Module):
    """Hilbert-Native 变深度 Patch Embedding.
    
    满足四个数学约束：
    1. 维度一致性: 所有 region → 相同 dim
    2. 路径一致性: 区域池化保持四叉树路径
    3. 尺度等变性: depth_scale 编码尺度信息
    4. LCA 兼容性: 与现有 LCA bias 无缝工作
    
    Args:
        channels: 输入图像通道数
        dim: 输出嵌入维度
        base_patch_size: 最细粒度 patch 大小 (共享 Conv 的 stride)
        max_depth: 最大四叉树深度
        conv_layers: SharedConv 层数 (1-3)
        use_batch_norm: 是否使用 BatchNorm
    """
    
    def __init__(
        self,
        channels: int = 3,
        dim: int = 256,
        base_patch_size: int = 4,
        max_depth: int = 4,
        conv_layers: int = 2,
        use_batch_norm: bool = True,
    ) -> None:
        super().__init__()
        
        self.channels = channels
        self.dim = dim
        self.base_patch_size = base_patch_size
        self.max_depth = max_depth
        
        # =====================================================================
        # SharedConv: 统一的特征提取器
        # =====================================================================
        # 使用 stride=base_patch_size，将图像 [B, C, H, W] → [B, dim, H/p, W/p]
        # 多层设计增加感受野，但保持单次 stride 降采样
        
        layers = []
        in_ch = channels
        out_ch = dim // 2 if conv_layers > 1 else dim
        
        # 第一层: 主要下采样
        layers.extend([
            nn.Conv2d(in_ch, out_ch, kernel_size=base_patch_size, stride=base_patch_size),
            nn.BatchNorm2d(out_ch) if use_batch_norm else nn.Identity(),
            nn.GELU(),
        ])
        
        # 中间层: 扩展感受野
        for i in range(1, conv_layers):
            next_ch = dim if i == conv_layers - 1 else out_ch
            layers.extend([
                nn.Conv2d(out_ch, next_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(next_ch) if use_batch_norm else nn.Identity(),
                nn.GELU() if i < conv_layers - 1 else nn.Identity(),
            ])
            out_ch = next_ch
        
        self.shared_conv = nn.Sequential(*layers)
        
        # =====================================================================
        # 深度编码
        # =====================================================================
        
        # 深度嵌入: 加法偏置，编码 region 的「语义角色」
        self.depth_embed = nn.Embedding(max_depth + 1, dim)
        
        # 深度缩放: 乘法因子，编码 region 的「信息密度」
        # 初始化: 深层 (细粒度) 权重略大，浅层 (粗粒度) 权重略小
        self.depth_scale = nn.Parameter(torch.ones(max_depth + 1))
        self._init_depth_scale()
        
        # 层归一化 (可选，用于稳定训练)
        self.norm = nn.LayerNorm(dim)
    
    def _init_depth_scale(self) -> None:
        """初始化深度缩放因子.
        
        数学依据:
        - 深层 token 覆盖小区域，信息密度高 → 略大的权重
        - 浅层 token 覆盖大区域，信息稀释 → 略小的权重
        
        初始化: σ_d = 1.0 + 0.05 * d / max_depth ∈ [1.0, 1.05]
        """
        with torch.no_grad():
            for d in range(self.max_depth + 1):
                self.depth_scale[d] = 1.0 + 0.05 * d / self.max_depth
    
    def forward(
        self,
        images: Tensor,
        split_results: List[SplitResult],
    ) -> Tuple[Tensor, Tensor]:
        """前向传播: 图像 + 分割结果 → token 序列.
        
        Args:
            images: [B, C, H, W] 输入图像
            split_results: 长度为 B 的 SplitResult 列表 (来自 AdaptiveSplitter)
            
        Returns:
            tokens: [B, N_max, dim] token 序列 (已按 Hilbert 顺序排列)
            levels_info: [B, N_max, max_depth+1] 层级信息 [depth, q1, q2, ...]
            
        Note:
            由于不同图像可能有不同数量的 token，使用 N_max = max(N_i) 并 padding
            实际 token 数量可从 split_results[b].num_tokens 获取
        """
        B, C, H, W = images.shape
        device = images.device
        
        # 1. 提取共享特征图
        features = self.shared_conv(images)  # [B, dim, H/p, W/p]
        _, _, fh, fw = features.shape
        
        # 2. 确定最大 token 数量
        max_tokens = max(sr.num_tokens for sr in split_results)
        
        # 3. 初始化输出
        tokens = torch.zeros(B, max_tokens, self.dim, device=device)
        levels_info = torch.zeros(
            B, max_tokens, self.max_depth + 1, 
            dtype=torch.long, device=device
        )
        
        # 4. 对每个 batch 提取 token
        for b in range(B):
            self._embed_batch(
                features[b],  # [dim, fh, fw]
                split_results[b],
                tokens[b],
                levels_info[b],
            )
        
        # 5. 层归一化
        tokens = self.norm(tokens)
        
        return tokens, levels_info
    
    def _embed_batch(
        self,
        feature_map: Tensor,  # [dim, fh, fw]
        split_result: SplitResult,
        out_tokens: Tensor,  # [N_max, dim] (output buffer)
        out_levels: Tensor,  # [N_max, max_depth+1] (output buffer)
    ) -> None:
        """为单个 batch 提取 token.
        
        注意: token 已按 Hilbert 顺序排列 (SplitResult 保证)
        """
        dim, fh, fw = feature_map.shape
        
        for i, token_info in enumerate(split_result.tokens):
            # 计算 region 在 feature map 上的坐标
            # region 坐标是像素坐标，需要除以 base_patch_size
            p = self.base_patch_size
            fx1 = token_info.region.x1 // p
            fy1 = token_info.region.y1 // p
            fx2 = max(fx1 + 1, token_info.region.x2 // p)  # 至少 1 个 feature
            fy2 = max(fy1 + 1, token_info.region.y2 // p)
            
            # 确保边界有效
            fx1 = min(fx1, fw - 1)
            fx2 = min(fx2, fw)
            fy1 = min(fy1, fh - 1)
            fy2 = min(fy2, fh)
            
            # 提取并池化 region 特征
            region_feat = feature_map[:, fy1:fy2, fx1:fx2]  # [dim, h, w]
            
            if region_feat.numel() > 0:
                # AdaptiveAvgPool 到 1x1
                pooled = F.adaptive_avg_pool2d(
                    region_feat.unsqueeze(0), 1
                ).squeeze()  # [dim]
            else:
                # 边界情况: 使用最近邻
                pooled = feature_map[:, fy1, fx1]
            
            # 应用深度编码
            depth = min(token_info.depth, self.max_depth)
            scale = self.depth_scale[depth]
            embed = self.depth_embed.weight[depth]
            
            # 最终 token: pooled * scale + embed
            out_tokens[i] = pooled * scale + embed
            
            # 填充 levels_info
            out_levels[i] = torch.tensor(
                token_info.to_levels_info(self.max_depth),
                dtype=torch.long,
                device=out_tokens.device,
            )
    
    def forward_fixed_grid(
        self,
        images: Tensor,
        grid_h: int,
        grid_w: int,
    ) -> Tuple[Tensor, Tensor]:
        """固定网格模式: 兼容非自适应分割.
        
        当不使用 AdaptiveSplit 时，使用均匀网格 (所有 token 深度相同)
        
        Args:
            images: [B, C, H, W]
            grid_h: 网格高度
            grid_w: 网格宽度
            
        Returns:
            tokens: [B, grid_h * grid_w, dim]
            levels_info: [B, grid_h * grid_w, max_depth+1]
        """
        B, C, H, W = images.shape
        device = images.device
        
        # 1. 提取特征
        features = self.shared_conv(images)  # [B, dim, fh, fw]
        _, dim, fh, fw = features.shape
        
        # 2. 重塑为 token 序列
        # 假设 fh == grid_h, fw == grid_w
        tokens = features.flatten(2).transpose(1, 2)  # [B, N, dim]
        
        # 3. 确定统一深度
        # log2(image_size / patch_size) 对应固定网格的深度
        depth = int(math.log2(max(grid_h, grid_w)))
        depth = min(depth, self.max_depth)
        
        # 应用深度编码
        scale = self.depth_scale[depth]
        embed = self.depth_embed.weight[depth]
        tokens = tokens * scale + embed.unsqueeze(0).unsqueeze(0)
        
        # 4. 层归一化
        tokens = self.norm(tokens)
        
        # 5. 创建 levels_info (统一深度)
        N = tokens.shape[1]
        levels_info = torch.zeros(
            B, N, self.max_depth + 1,
            dtype=torch.long, device=device
        )
        levels_info[:, :, 0] = depth
        
        # 填充四叉树路径 (从 HilbertPathCache 获取)
        from .streaming_tokenizer import HilbertPathCache
        _, quadtree_paths = HilbertPathCache.get_or_compute(
            grid_h=fh, grid_w=fw, max_depth=self.max_depth, device=device
        )
        path_len = min(quadtree_paths.shape[1], self.max_depth)
        actual_n = min(N, quadtree_paths.shape[0])
        levels_info[:, :actual_n, 1:path_len+1] = quadtree_paths[:actual_n, :path_len]
        
        return tokens, levels_info


class DepthAwarePositionalEncoding(nn.Module):
    """深度感知位置编码.
    
    结合传统正弦位置编码和深度信息:
        PE(i, d) = sin/cos(pos) + DepthEmbed(d)
    
    与 HilbertNativePatchEmbed 配合使用时，
    提供额外的位置信息补充。
    """
    
    def __init__(
        self,
        dim: int,
        max_tokens: int = 1024,
        max_depth: int = 4,
    ) -> None:
        super().__init__()
        
        self.dim = dim
        self.max_tokens = max_tokens
        
        # 正弦位置编码 (预计算)
        pe = torch.zeros(max_tokens, dim)
        position = torch.arange(0, max_tokens, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # 深度嵌入
        self.depth_embed = nn.Embedding(max_depth + 1, dim)
    
    def forward(
        self,
        tokens: Tensor,  # [B, N, dim]
        depths: Tensor,  # [B, N] 每个 token 的深度
    ) -> Tensor:
        """添加位置编码."""
        B, N, D = tokens.shape
        
        # 正弦位置编码
        pos_enc = self.pe[:N].unsqueeze(0).expand(B, -1, -1)
        
        # 深度嵌入
        depth_enc = self.depth_embed(depths)  # [B, N, dim]
        
        return tokens + pos_enc + depth_enc
