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

方案 C+ (Region Pooling) 实现 - 向量化版本
==========================================

公式:
    F = SharedConv(Image)  ∈ ℝ^{B × dim × H/p × W/p}
    
    向量化池化 (使用 ROI-Align):
        boxes = [(b, x1/p, y1/p, x2/p, y2/p) for all regions]
        pooled = roi_align(F, boxes, output_size=(1,1))  # [N_total, dim, 1, 1]
        T = pooled · σ_d + E_d  # 批量深度编码
    
    复杂度:
        - 时间: O(1) GPU kernel 调用 (vs O(N) for Python loop)
        - 空间: O(N × dim) 
        - Kernel 启动: 1 次 (vs N 次)

参数量:
    - SharedConv: dim × C × p × p ≈ 12K
    - depth_embed: (D+1) × dim ≈ 1.3K
    - depth_scale: D+1 ≈ 5
    - 总计: ~14K (vs 4.2M for Depth-Specific Conv)

深度缩放范围 (P6-1 改进)
==========================

数学形式化:
    σ_d = σ_min + (σ_max - σ_min) · sigmoid(γ_d)
    
    其中:
    - σ_min = 0.5, σ_max = 2.0 (默认)
    - γ_d 是可学习参数
    - 动态范围: 4x (vs 原始 1.2x)
    
约束验证:
    【C1 信息保持】σ_min ≥ 0.5 确保浅层至少保留 50% 信息
    【C2 梯度稳定】σ_max ≤ 2.0 确保梯度放大不超过 2x
    【C3 区分度】σ_max/σ_min = 4x 提供充足的深度区分能力
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# I100-4: 强制依赖 torchvision
# 移除 _fallback_roi_pool 回退实现，统一使用 torchvision.ops.roi_align
from torchvision.ops import roi_align  # 强制依赖，无回退

# I97-9: SplitResult 和 SplitToken 已从 split_adaptive.py 移除
# 仅保留类型注解用于文档，实际使用 TensorSplitResult
from typing import List, Protocol, Any

class SplitToken:
    """Legacy type - 仅用于类型注解，实际使用 GumbelTopKSplitter."""
    def __init__(self, **kwargs):
        pass

class SplitResult:
    """Legacy type - 仅用于类型注解，实际使用 TensorSplitResult."""
    def __init__(self, **kwargs):
        pass


class HilbertNativePatchEmbed(nn.Module):
    """Hilbert-Native 变深度 Patch Embedding.

    满足四个数学约束：
    1. 维度一致性: 所有 region → 相同 dim
    2. 路径一致性: 区域池化保持四叉树路径
    3. 尺度等变性: depth_scale 编码尺度信息
    4. LCA 兼容性: 与现有 LCA bias 无缝工作

    I30-17-EXT: 支持动态深度计算
        - 新 API: 使用 min_patch_size + max_level_limit
        - 旧 API: 直接指定 max_level (自动转换)

    Args:
        channels: 输入图像通道数
        dim: 输出嵌入维度
        base_patch_size: 最细粒度 patch 大小 (共享 Conv 的 stride)
        min_patch_size: [新 API] 目标最小 patch 大小，用于动态计算 max_level
        max_level_limit: [新 API] max_level 硬上限
        max_level: [旧 API] 最大四叉树深度 (直接指定)
        image_size: [新 API] 输入图像尺寸，用于计算 max_level
        conv_layers: SharedConv 层数 (1-3)
        use_batch_norm: 是否使用 BatchNorm
        depth_scale_beta: [已废弃] 使用 depth_scale_range 替代
        depth_scale_range: 深度缩放范围 (σ_min, σ_max)，使用 sigmoid 参数化
                          默认 (0.5, 2.0) 提供 4x 动态范围
                          设为 None 使用旧版固定初始化 (向后兼容)
    """

    def __init__(
        self,
        channels: int = 3,
        dim: int = 256,
        base_patch_size: int = 4,
        # I30-17-EXT: 新 API 参数
        min_patch_size: Optional[int] = None,
        max_level_limit: int = 8,
        # 旧 API 参数 (直接指定 max_level)
        max_level: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        conv_layers: int = 2,
        use_batch_norm: bool = True,
        depth_scale_beta: float = 0.2,
        depth_scale_range: Optional[Tuple[float, float]] = (0.5, 2.0),
    ) -> None:
        super().__init__()

        self.channels = channels
        self.dim = dim
        self.base_patch_size = base_patch_size
        self.depth_scale_beta = depth_scale_beta
        self.depth_scale_range = depth_scale_range

        # I30-17-EXT: 处理新旧 API
        if max_level is not None:
            # 旧 API: 直接使用指定的 max_level
            self.max_level = max_level
        elif min_patch_size is not None and image_size is not None:
            # 新 API: 从 min_patch_size 计算 max_level
            from vit_pytorch.core.depth_utils import compute_max_level
            H, W = image_size
            self.max_level = compute_max_level(
                (H, W), min_patch_size, max_level_limit
            )
        else:
            # 默认值
            self.max_level = max_level_limit
        
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
        # FPN 金字塔 (FPN-style Multi-scale Feature Sampling)
        # =====================================================================
        # I-PHASE3: 添加 FPN 以解决单尺度特征截断问题
        #
        # 数学形式:
        #   FPN_l = Conv(Resize(F_{l-1})) + F_l
        #
        # 其中 F_l 是第 l 层的特征，Resize 是 2x 上采样
        #
        # 深度-特征层映射:
        #   depth 1-2 (大区域) → FPN_2 (最粗糙, 感受野最大)
        #   depth 3-4 (中区域) → FPN_1 (中等感受野)
        #   depth 5+  (小区域) → FPN_0 (最精细, 感受野最小)
        #
        # 这解决了原始方案的问题:
        #   - 单尺度特征导致小 Token (<1px) 产生 Aliasing

        self.fpn_levels = min(conv_layers, 3)  # 最多3层金字塔
        self.fpn_conv = nn.ModuleList()

        # 构建 FPN: 所有层使用 dim 通道以保持一致性
        for i in range(self.fpn_levels):
            self.fpn_conv.append(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            )
        
        # =====================================================================
        # 深度编码
        # =====================================================================
        
        # 深度嵌入: 加法偏置，编码 region 的「语义角色」
        # I24: 使用较小的初始化标准差 (0.02)，避免淹没 pooled features
        # 原问题: nn.Embedding 默认初始化 std~1.0，而 ROI-Align pooled std~0.2
        # 这导致 96% 的 token (同一 depth) 共享几乎相同的表示
        self.depth_embed = nn.Embedding(max_level + 1, dim)
        nn.init.normal_(self.depth_embed.weight, mean=0.0, std=0.02)
        
        # 深度缩放: 乘法因子，编码 region 的「信息密度」
        # P6-1 改进: 使用 sigmoid 参数化，扩展动态范围到 4x
        if depth_scale_range is not None:
            # 新版: 可学习 sigmoid 参数化
            # σ_d = σ_min + (σ_max - σ_min) · sigmoid(γ_d)
            self._depth_scale_raw = nn.Parameter(torch.zeros(max_level + 1))
            self._init_depth_scale_learnable()
        else:
            # 旧版: 固定线性初始化 (向后兼容)
            self._depth_scale_raw = None
            self._depth_scale_fixed = nn.Parameter(torch.ones(max_level + 1))
            self._init_depth_scale_legacy()
        
        # 层归一化 (可选，用于稳定训练)
        self.norm = nn.LayerNorm(dim)
    
    def _init_depth_scale_learnable(self) -> None:
        """初始化可学习深度缩放因子 (P6-1 改进).
        
        数学形式化:
            σ_d = σ_min + (σ_max - σ_min) · sigmoid(γ_d)
            
        初始化策略:
            保持与旧版语义一致 (1.0 → 1.0+β)，但允许学习到 [σ_min, σ_max]
            
        计算:
            目标 σ_d = 1.0 + β·d/D_max
            sigmoid(γ_d) = (σ_d - σ_min) / (σ_max - σ_min)
            γ_d = logit(sigmoid_target)
        """
        assert self.depth_scale_range is not None
        sigma_min, sigma_max = self.depth_scale_range
        
        with torch.no_grad():
            for d in range(self.max_level + 1):
                # 目标值: 与旧版初始化一致
                target_sigma = 1.0 + self.depth_scale_beta * d / self.max_level
                # 裁剪到有效范围
                target_sigma = max(sigma_min + 0.01, min(sigma_max - 0.01, target_sigma))
                # 计算 sigmoid 目标值
                sigmoid_target = (target_sigma - sigma_min) / (sigma_max - sigma_min)
                # 计算 logit (sigmoid 逆函数)
                self._depth_scale_raw[d] = math.log(sigmoid_target / (1 - sigmoid_target))
    
    def _init_depth_scale_legacy(self) -> None:
        """旧版初始化 (向后兼容).
        
        初始化: σ_d = 1.0 + β * d / max_level ∈ [1.0, 1.0+β]
        """
        assert self._depth_scale_fixed is not None
        with torch.no_grad():
            for d in range(self.max_level + 1):
                self._depth_scale_fixed[d] = 1.0 + self.depth_scale_beta * d / self.max_level
    
    @property
    def depth_scale(self) -> torch.Tensor:
        """获取深度缩放因子.

        P6-1 改进: 使用 sigmoid 参数化确保值在 [σ_min, sigma_max] 范围内

        Returns:
            shape: (max_level + 1,) 的缩放因子张量
        """
        if self._depth_scale_raw is not None:
            # 新版: sigmoid 参数化
            assert self.depth_scale_range is not None
            sigma_min, sigma_max = self.depth_scale_range
            return sigma_min + (sigma_max - sigma_min) * torch.sigmoid(self._depth_scale_raw)
        else:
            # 旧版: 直接返回固定参数
            return self._depth_scale_fixed

    def _build_fpn_features(self, features: torch.Tensor) -> list:
        """构建 FPN 特征金字塔

        数学形式:
            FPN_0 = features (最精细)
            FPN_l = Conv(Resize(FPN_{l-1})) + features_l

        Args:
            features: [B, C, H, W] 共享卷积特征

        Returns:
            fpn_features: [FPN_levels] 特征金字塔列表
        """
        import torch.nn.functional as F

        fpn_features = [features]

        # 从最细到最粗构建金字塔
        # 注意: 只需要 2 层上采样，因为 conv_layers=2 时只有 2 层特征
        for i in range(1, min(self.fpn_levels, 2)):
            # 上采样 2x
            prev_feat = fpn_features[-1]
            upsampled = F.interpolate(
                prev_feat,
                scale_factor=2.0,
                mode='nearest'
            )

            # 3x3 卷积减少混叠 (通道数相同，无需调整)
            convolved = self.fpn_conv[i](upsampled)
            fpn_features.append(convolved)

        return fpn_features

    def _select_fpn_level(self, depths: torch.Tensor) -> torch.Tensor:
        """根据深度选择 FPN 层级

        数学映射:
            depth 1-2 → level 2 (最粗糙，感受野最大)
            depth 3-4 → level 1 (中等)
            depth 5+  → level 0 (最精细)

        Args:
            depths: [N] token 深度

        Returns:
            level_indices: [N] 对应的 FPN 层级
        """
        # 映射: depth → level
        # level = max_level - depth (深token用细特征)
        levels = torch.clamp(self.max_level - depths, min=0, max=self.fpn_levels - 1)
        return levels

    def _dynamic_roi_align(
        self,
        features: Tensor,
        boxes: Tensor,
        depths: Tensor,
    ) -> Tensor:
        """动态 sampling_ratio 的 ROI-Align

        I106-1: 根据 Token 深度动态调整 sampling_ratio
        深层 Token 感受野小，使用更细致的采样防止特征模糊

        数学形式:
            sampling_ratio(d_i) = {
                1,  d_i <= 2   (浅层，全局特征)
                2,  2 < d_i <= 4  (中层，中等细节)
                4,  d_i > 4    (深层，需要更细采样)
            }

        Args:
            features: [B, D, H, W] 特征图
            boxes: [N, 5] ROI boxes [batch_idx, x1, y1, x2, y2]
            depths: [N] 每个 ROI 的深度

        Returns:
            pooled: [N, D] 池化后的特征
        """
        import torch.nn.functional as F

        # 定义深度区间和对应的 sampling_ratio
        depth_bins = [0, 2, 4, self.max_level + 1]
        sampling_ratios = [1, 2, 4]

        # 按深度分组
        pooled_list = []
        indices_list = []

        for i in range(len(depth_bins) - 1):
            low, high = depth_bins[i], depth_bins[i + 1]
            mask = (depths >= low) & (depths < high)
            if not mask.any():
                continue

            indices = mask.nonzero(as_tuple=True)[0]
            group_boxes = boxes[indices]

            # 调用 ROI-Align，使用对应的 sampling_ratio
            pooled = roi_align(
                features,
                group_boxes,
                output_size=(1, 1),
                spatial_scale=1.0,
                sampling_ratio=sampling_ratios[i],
                aligned=True,
            )  # [N_group, D, 1, 1]
            pooled = pooled.squeeze(-1).squeeze(-1)

            pooled_list.append(pooled)
            indices_list.append(indices)

        # 合并结果
        if len(pooled_list) == 0:
            # 无 token 时的边界情况
            return torch.zeros(0, features.shape[1], device=features.device, dtype=features.dtype)

        # 按原始顺序合并
        all_pooled = torch.zeros(len(depths), features.shape[1], device=features.device, dtype=features.dtype)
        for pooled, indices in zip(pooled_list, indices_list):
            all_pooled[indices] = pooled

        return all_pooled

    def forward(
        self,
        images: Tensor,
        split_results: List[SplitResult],
    ) -> Tuple[Tensor, Tensor]:
        """前向传播: 图像 + 分割结果 → token 序列 (向量化实现).
        
        数学形式化:
            T = ROI-Align(F, boxes) · σ_d + E_d
            
            其中:
            - F = SharedConv(I) ∈ ℝ^{B × D × H' × W'}
            - boxes = [(b, x1', y1', x2', y2')] 转换后的 ROI 坐标
            - σ_d = depth_scale[d] 深度缩放因子
            - E_d = depth_embed[d] 深度嵌入向量
        
        复杂度:
            - 时间: O(N_total × D) where N_total = Σ N_b
            - Kernel 调用: 1 次 (vs O(N_total) for Python loop)
        
        Args:
            images: [B, C, H, W] 输入图像
            split_results: 长度为 B 的 SplitResult 列表 (来自 AdaptiveSplitter)
            
        Returns:
            tokens: [B, N_max, dim] token 序列 (已按 Hilbert 顺序排列)
            levels_info: [B, N_max, max_level+1] 层级信息 [depth, q1, q2, ...]
        """
        B, C, H, W = images.shape
        device = images.device
        dtype = images.dtype
        
        # 1. 提取共享特征图
        features = self.shared_conv(images)  # [B, dim, H/p, W/p]
        _, dim, fh, fw = features.shape
        
        # 2. 确定最大 token 数量和总 token 数
        token_counts = [sr.num_tokens for sr in split_results]
        max_tokens = max(token_counts)
        total_tokens = sum(token_counts)
        
        if total_tokens == 0:
            # 边界情况: 无 token
            tokens = torch.zeros(B, 1, self.dim, device=device, dtype=dtype)
            levels_info = torch.zeros(B, 1, self.max_level + 1, dtype=torch.long, device=device)
            return self.norm(tokens), levels_info
        
        # 3. 收集所有 region 的 boxes 和 depths (向量化准备)
        all_boxes = []  # [batch_idx, x1, y1, x2, y2] in feature map coords
        all_depths = []
        all_levels_info = []
        batch_indices = []
        token_indices = []  # 每个 token 在其 batch 内的索引
        
        p = self.base_patch_size
        for b, sr in enumerate(split_results):
            for i, token in enumerate(sr.tokens):
                # 将像素坐标转换为 feature map 坐标
                # ROI-Align 使用浮点坐标，格式为 [batch_idx, x1, y1, x2, y2]
                fx1 = token.region.x1 / p
                fy1 = token.region.y1 / p
                fx2 = token.region.x2 / p
                fy2 = token.region.y2 / p
                
                # 确保有效的 ROI (至少 1 个像素)
                fx2 = max(fx1 + 0.5, fx2)
                fy2 = max(fy1 + 0.5, fy2)
                
                all_boxes.append([b, fx1, fy1, fx2, fy2])
                all_depths.append(min(token.depth, self.max_level))
                all_levels_info.append(token.to_levels_info(self.max_level))
                batch_indices.append(b)
                token_indices.append(i)
        
        # 4. 转换为 tensor
        boxes_tensor = torch.tensor(all_boxes, device=device, dtype=dtype)  # [N_total, 5]
        depths_tensor = torch.tensor(all_depths, device=device, dtype=torch.long)  # [N_total]
        
        # 5. ROI-Align 批量池化 (支持动态 sampling_ratio)
        # I106-1: 深层 Token 使用更细致的采样，防止特征模糊
        # sampling_ratio(d) = 1 (d<=2), 2 (2<d<=4), 4 (d>4)
        pooled = self._dynamic_roi_align(
            features, boxes_tensor, depths_tensor
        )  # [N_total, D]

        # 6. 批量应用深度编码
        # t_i = pooled_i * σ_{d_i} + E_{d_i}
        scales = self.depth_scale[depths_tensor]  # [N_total]
        embeds = self.depth_embed(depths_tensor)  # [N_total, D]
        all_tokens = pooled * scales.unsqueeze(-1) + embeds  # [N_total, D]
        
        # 7. 分配到输出 buffer
        tokens = torch.zeros(B, max_tokens, self.dim, device=device, dtype=dtype)
        levels_info = torch.zeros(B, max_tokens, self.max_level + 1, dtype=torch.long, device=device)
        
        for idx, (b, i) in enumerate(zip(batch_indices, token_indices)):
            tokens[b, i] = all_tokens[idx]
            levels_info[b, i] = torch.tensor(
                all_levels_info[idx], dtype=torch.long, device=device
            )
        
        # 8. 层归一化
        tokens = self.norm(tokens)
        
        return tokens, levels_info

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
            levels_info: [B, grid_h * grid_w, max_level+1]
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
        # I109-7: 使用 bit_length() 替代 int(math.log2(...)) 避免浮点精度问题
        depth = max(grid_h, grid_w).bit_length() - 1
        depth = min(depth, self.max_level)
        
        # 应用深度编码
        scale = self.depth_scale[depth]
        embed = self.depth_embed.weight[depth]
        tokens = tokens * scale + embed.unsqueeze(0).unsqueeze(0)
        
        # 4. 层归一化
        tokens = self.norm(tokens)
        
        # 5. 创建 levels_info (统一深度)
        N = tokens.shape[1]
        levels_info = torch.zeros(
            B, N, self.max_level + 1,
            dtype=torch.long, device=device
        )
        levels_info[:, :, 0] = depth
        
        # 填充四叉树路径 (从 HilbertPathCache 获取)
        from vit_pytorch.core.hilbert_indexer import HilbertPathCache
        _, quadtree_paths = HilbertPathCache.get_or_compute(
            grid_h=fh, grid_w=fw, max_level=self.max_level, device=device
        )
        path_len = min(quadtree_paths.shape[1], self.max_level)
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
        max_level: int = 4,
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
        self.depth_embed = nn.Embedding(max_level + 1, dim)
    
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
