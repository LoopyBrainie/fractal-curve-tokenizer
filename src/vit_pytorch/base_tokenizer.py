# -*- coding: utf-8 -*-
"""
Tokenization 抽象基类模块

数学形式化
============

Tokenizer 抽象定义:
    T: R^{B × C × H × W} → TokenizerOutput
    
    TokenizerOutput 包含:
    - sequences: List[TokenSequence], 长度为 B
    - 每个 TokenSequence:
        - tokens: R^{N_i × D}, token 嵌入
        - levels: Z^{N_i × info_len}, 层级信息

levels_info 格式:
    col 0: depth (深度值 d ∈ {0, ..., L_max})
    col 1+: path (路径索引 q^(j) ∈ {0, 1, 2, 3})

类对照表
----------
+-------------------+--------------------------------------+
| 类                 | 用途                                  |
+===================+======================================+
| TokenSequence     | 单个样本的 token 序列                  |
| TokenizerOutput   | 批次输出，包含多个 TokenSequence       |
| LegacyTokenizerOutput| 向后兼容的 (tokens, levels) 格式    |
| BaseTokenizer     | Tokenizer 抽象基类                    |
| BaseTokenProcessor| TokenProcessor 抽象基类              |
+-------------------+--------------------------------------+
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class TokenSequence:
    tokens: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def get_levels(self) -> Optional[torch.Tensor]:
        levels = self.metadata.get("levels")
        if isinstance(levels, torch.Tensor):
            return levels
        return None

    @property
    def device(self) -> torch.device:
        return self.tokens.device

    def clone(self) -> "TokenSequence":
        metadata = {k: v for k, v in self.metadata.items()}
        return TokenSequence(tokens=self.tokens.clone(), metadata=metadata)


@dataclass
class TokenizerOutput:
    """批次 Tokenizer 输出容器。

    提供便捷属性访问堆叠的 tokens 和 levels_info，
    同时保持对底层 TokenSequence 列表的完整访问。

    P9-5 优化: 支持预填充缓存，避免重复 padding 操作。
    当 Tokenizer 内部已经有 padding 后的张量时，可直接传入缓存，
    消除 model._prepare_tokens 中的 O(B) Python 循环。

    P11-3 改进: 新增 _regions_cache 和 _image_size_cache 用于
    直接从区域边界计算正确的四叉树 LCA 偏置。

    P12-2 优化: _lengths_cache 存储为 Tensor 而非 List[int]，
    避免 _create_attention_mask 中的 Python 列表到 Tensor 转换开销。

    I30-11: 新增 _split_probs_cache 用于加权池化 (Weighted Mean Pooling)

    Attributes:
        sequences: 各样本的 TokenSequence 列表
        _padded_tokens_cache: 预填充的 tokens [B, MaxN, D] (可选缓存)
        _padded_levels_cache: 预填充的 levels [B, MaxN, info_dim] (可选缓存)
        _lengths_cache: 每个样本的实际 token 数量 Tensor[B] (可选缓存)
        _regions_cache: (P11-3) 预填充的 regions [B, MaxN, 4] (可选缓存)
        _image_size_cache: (P11-3) 图像边长 (可选缓存)
        _split_probs_cache: (I30-11) 预填充的分割概率 [B, MaxN] (可选缓存)

    Properties:
        tokens: 堆叠的 tokens [B, N, D]（假设所有样本 token 数量相同）
        levels_info: 堆叠的 levels_info [B, N, K] 或 None
        split_probs: 堆叠的分割概率 [B, N] 或 None (I30-11)
        batch_size: 批次大小
    """
    sequences: List[TokenSequence]
    _padded_tokens_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _padded_levels_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _lengths_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _regions_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _image_size_cache: Optional[int] = field(default=None, repr=False)
    _split_probs_cache: Optional[torch.Tensor] = field(default=None, repr=False)

    def __iter__(self) -> Iterator[TokenSequence]:
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)
    
    @property
    def batch_size(self) -> int:
        """获取批次大小。"""
        return len(self.sequences)
    
    @property
    def tokens(self) -> torch.Tensor:
        """获取堆叠的 tokens [B, N, D]。
        
        假设所有样本的 token 数量相同（padding 后）。
        
        Returns:
            堆叠的 tokens 张量，形状为 [B, N, D]
            
        Raises:
            ValueError: 当序列为空时
        """
        if len(self.sequences) == 0:
            raise ValueError("Cannot get tokens from empty TokenizerOutput")
        return torch.stack([seq.tokens for seq in self.sequences])
    
    @property
    def levels_info(self) -> Optional[torch.Tensor]:
        """获取堆叠的 levels_info [B, N, K] 或 None。

        如果所有样本都没有 levels 信息，返回 None。
        否则返回堆叠的张量（对缺失的 levels 用零填充）。

        I32-2: 如果有 _padded_levels_cache 缓存，直接返回缓存
               缓存中 padding 位置使用 -1 sentinel 标识

        Returns:
            堆叠的 levels_info 张量或 None
        """
        if len(self.sequences) == 0:
            return None

        # I32-2: 优先使用缓存
        if self._padded_levels_cache is not None:
            return self._padded_levels_cache

        levels_list = [seq.get_levels() for seq in self.sequences]

        # 如果所有 levels 都是 None，返回 None
        if all(l is None for l in levels_list):
            return None

        # 找到最大维度
        device = self.sequences[0].device
        max_len = max(l.shape[0] if l is not None else 0 for l in levels_list)

        if max_len == 0:
            return None

        # 确定 info_dim（处理 1D 和 2D 情况）
        info_dims = [l.shape[1] if l is not None and l.dim() > 1 else 1 for l in levels_list]
        info_dim = max(info_dims)
        
        # 创建填充后的张量
        stacked = torch.zeros(
            len(self.sequences), max_len, info_dim,
            dtype=torch.long, device=device
        )
        
        for i, levels in enumerate(levels_list):
            if levels is not None:
                if levels.dim() == 1:
                    stacked[i, :levels.shape[0], 0] = levels
                else:
                    stacked[i, :levels.shape[0], :levels.shape[1]] = levels
        
        return stacked

    def tokens_list(self) -> List[torch.Tensor]:
        return [seq.tokens for seq in self.sequences]

    def get_padded_tokens(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取预填充的 tokens 和长度张量 (P9-5/P12-2 优化).
        
        如果有缓存，直接返回缓存的张量，避免重复 padding。
        否则使用 pad_sequence 进行填充。
        
        P12-2 优化: lengths 返回 Tensor 而非 List[int]，
        支持后续 attention mask 的向量化创建，避免 O(B) Python 循环。
        
        Returns:
            (padded_tokens, lengths):
            - padded_tokens: [B, MaxN, D] 填充后的 tokens
            - lengths: Tensor[B] 每个样本的实际 token 数量
        """
        if self._padded_tokens_cache is not None and self._lengths_cache is not None:
            return self._padded_tokens_cache, self._lengths_cache
        
        # 回退: 使用 pad_sequence
        tokens_list = self.tokens_list()
        lengths_list = [t.shape[0] for t in tokens_list]
        device = tokens_list[0].device if len(tokens_list) > 0 else torch.device('cpu')
        lengths = torch.tensor(lengths_list, dtype=torch.long, device=device)
        
        # I24-14: 回退路径也需要 clamp，防止空序列
        lengths = lengths.clamp(min=1)
        
        padded_tokens = torch.nn.utils.rnn.pad_sequence(
            tokens_list, batch_first=True, padding_value=0.0
        )
        return padded_tokens, lengths

    def get_padded_levels(self, info_dim: int) -> torch.Tensor:
        """获取预填充的 levels 信息 (P9-5 优化).
        
        如果有缓存，直接返回缓存的张量，避免 O(B) Python 循环。
        否则回退到标准 padding 逻辑。
        
        Args:
            info_dim: 目标 info 维度 (通常是 max_level + 4)
            
        Returns:
            padded_levels: [B, MaxN, info_dim] 填充后的 levels
        """
        if self._padded_levels_cache is not None:
            cache = self._padded_levels_cache
            # 检查是否需要扩展 info_dim
            if cache.shape[2] >= info_dim:
                return cache[:, :, :info_dim]
            else:
                # 需要扩展
                B, MaxN, _ = cache.shape
                device = cache.device
                padded = torch.zeros(B, MaxN, info_dim, dtype=torch.long, device=device)
                padded[:, :, :cache.shape[2]] = cache
                return padded
        
        # 回退: 使用 levels_info 属性 (包含 Python 循环)
        levels = self.levels_info
        if levels is None:
            B = len(self.sequences)
            device = self.sequences[0].device if B > 0 else torch.device('cpu')
            return torch.zeros(B, 1, info_dim, dtype=torch.long, device=device)
        
        if levels.shape[2] >= info_dim:
            return levels[:, :, :info_dim]
        else:
            B, MaxN, _ = levels.shape
            device = levels.device
            padded = torch.zeros(B, MaxN, info_dim, dtype=torch.long, device=device)
            padded[:, :, :levels.shape[2]] = levels
            return padded

    def levels_list(self) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for seq in self.sequences:
            levels = seq.get_levels()
            if levels is None:
                result.append(torch.empty(0, dtype=torch.long, device=seq.device))
            else:
                result.append(levels)
        return result

    def get_padded_regions(self) -> Tuple[Optional[torch.Tensor], Optional[int]]:
        """获取预填充的 regions 和 image_size (P11-3 新增).

        用于直接从区域边界计算正确的四叉树 LCA 偏置，
        绕过 levels_info 中全为 0 的路径问题。

        Returns:
            (regions, image_size):
            - regions: [B, MaxN, 4] 填充后的 regions，格式 [x1, y1, x2, y2]
                       或 None (如果未设置缓存)
            - image_size: 图像边长，或 None
        """
        return self._regions_cache, self._image_size_cache

    @property
    def split_probs(self) -> Optional[torch.Tensor]:
        """获取堆叠的分割概率 [B, N] 或 None (I30-11).

        用于 Weighted Mean Pooling，每个 token 的重要性权重。

        Returns:
            堆叠的分割概率张量或 None
        """
        return self._split_probs_cache

    def get_padded_split_probs(self) -> Optional[torch.Tensor]:
        """获取预填充的分割概率 (I30-11 新增).

        用于加权池化，利用 GumbelTopKSplitter 的分割概率作为权重。
        如果有缓存，直接返回；否则返回 None。

        Returns:
            [B, MaxN] 填充后的分割概率，或 None (如果未设置缓存)
        """
        return self._split_probs_cache

    def to_legacy(self) -> "LegacyTokenizerOutput":
        return LegacyTokenizerOutput(
            tokens=self.tokens_list(),
            levels=self.levels_list(),
        )


@dataclass
class LegacyTokenizerOutput:
    tokens: List[torch.Tensor]
    levels: List[torch.Tensor]


class BaseTokenizer(nn.Module):
    """Abstract base class for tokenizers."""

    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> Any:  # type: ignore[override]
        return self.tokenize(images)


class BaseTokenProcessor(nn.Module):
    """Base class for modules that post-process tokenizer outputs."""

    def process(self, batch: TokenizerOutput) -> TokenizerOutput:
        raise NotImplementedError

    def forward(self, batch: TokenizerOutput) -> TokenizerOutput:  # type: ignore[override]
        return self.process(batch)
