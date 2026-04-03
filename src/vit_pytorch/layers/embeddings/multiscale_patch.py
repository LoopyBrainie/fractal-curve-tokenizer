# -*- coding: utf-8 -*-
"""
多尺度 Patch 编码器

数学形式化
============

多尺度卷积金字塔:
    F_s = Conv_s(I), s ∈ {1, ..., S}
    每个尺度: kernel_size = stride = patch_size_s

输出:
    {patch_size: (features [B, D, H/ps, W/ps], grid_shape)}
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn


class MultiScalePatchEncoder(nn.Module):
    """多尺度 Patch 编码器.
    
    使用不同大小的卷积核提取多尺度特征，替代 BFS 递归分割。
    每个尺度独立处理，最后融合。
    
    Args:
        channels: 输入图像通道数
        d_model: 输出嵌入维度
        patch_sizes: 支持的 patch 大小列表
    """
    
    def __init__(
        self,
        channels: int = 3,
        d_model: int = 256,
        patch_sizes: Tuple[int, ...] = (4, 8, 16),
    ) -> None:
        super().__init__()
        self.channels = channels
        self.d_model = d_model
        self.patch_sizes = patch_sizes
        self.num_scales = len(patch_sizes)
        
        # 每个尺度的编码器
        # 使用 Conv2d: kernel_size = stride = patch_size
        self.encoders = nn.ModuleDict({
            f"scale_{ps}": nn.Sequential(
                nn.Conv2d(channels, d_model // 2, kernel_size=ps, stride=ps),
                nn.BatchNorm2d(d_model // 2),
                nn.GELU(),
                nn.Conv2d(d_model // 2, d_model, kernel_size=1),
                nn.BatchNorm2d(d_model),
            )
            for ps in patch_sizes
        })
        
        # 尺度级别嵌入
        self.scale_embedding = nn.Embedding(self.num_scales, d_model)
        
    def forward(
        self,
        images: torch.Tensor,
    ) -> Dict[int, Tuple[torch.Tensor, Tuple[int, int]]]:
        """提取多尺度特征.
        
        Args:
            images: [B, C, H, W] 输入图像
            
        Returns:
            features_dict: {patch_size: (features, (grid_h, grid_w))}
                - features: [B, D, grid_h, grid_w]
        """
        B, C, H, W = images.shape
        features_dict = {}
        
        for scale_idx, ps in enumerate(self.patch_sizes):
            # 检查图像是否足够大
            if H >= ps and W >= ps:
                feat = self.encoders[f"scale_{ps}"](images)  # [B, D, H/ps, W/ps]
                grid_h, grid_w = feat.shape[2], feat.shape[3]
                
                # 添加尺度嵌入
                scale_emb = self.scale_embedding(
                    torch.tensor([scale_idx], device=images.device)
                )  # [1, D]
                feat = feat + scale_emb.view(1, -1, 1, 1)
                
                features_dict[ps] = (feat, (grid_h, grid_w))

        return features_dict

    @property
    def embed_output(self) -> Dict[str, Any]:
        """MultiScalePatchEncoder 诊断输出

        命名空间:
            embed/params/*: 可学习参数统计
        """
        output: Dict[str, Any] = {}

        # embed/params/* - 尺度嵌入统计
        if hasattr(self, 'scale_embedding') and self.scale_embedding is not None:
            w = self.scale_embedding.weight
            output["params/scale_emb_norm"] = float(w.norm().item())
            output["params/scale_emb_mean"] = float(w.mean().item())
            output["params/scale_emb_std"] = float(w.std().item())

        # embed/params/* - 各尺度编码器参数统计
        if hasattr(self, 'encoders') and self.encoders is not None:
            for ps in self.patch_sizes:
                key = f"scale_{ps}"
                if key in self.encoders:
                    encoder = self.encoders[key]
                    # 统计第一个 Conv 层的权重范数
                    first_conv = encoder[0]  # Conv2d
                    output[f"params/scale_{ps}_conv_norm"] = float(first_conv.weight.norm().item())

        return output
