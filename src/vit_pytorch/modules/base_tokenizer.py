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

from abc import abstractmethod
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

    I98-5: 新增 _levels_info_cache 用于 LevelsInfo 强类型缓存

    P-OPT-11 Phase 9: 延迟构造 TokenSequence。
    支持 sequences=None 创建，首次访问时从缓存构造。

    Attributes:
        sequences: 各样本的 TokenSequence 列表 (可为 None，实现延迟构造)
        _padded_tokens_cache: 预填充的 tokens [B, MaxN, D] (可选缓存)
        _padded_levels_cache: 预填充的 levels [B, MaxN, info_dim] (可选缓存)
        _lengths_cache: 每个样本的实际 token 数量 Tensor[B] (可选缓存)
        _regions_cache: (P11-3) 预填充的 regions [B, MaxN, 4] (可选缓存)
        _image_size_cache: (P11-3) 图像边长 (可选缓存)
        _split_probs_cache: (I30-11) 预填充的分割概率 [B, MaxN] (可选缓存)
        _levels_info_cache: (I98-5) LevelsInfo 强类型缓存 (可选缓存)

    Properties:
        tokens: 堆叠的 tokens [B, N, D]（假设所有样本 token 数量相同）
        levels_info: (I98-5) LevelsInfo 强类型，包含运行时契约验证
        split_probs: 堆叠的分割概率 [B, N] 或 None (I30-11)
        batch_size: 批次大小
    """
    _sequences_data: Optional[List[TokenSequence]] = None
    _padded_tokens_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _padded_levels_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _lengths_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _regions_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _image_size_cache: Optional[int] = field(default=None, repr=False)
    _split_probs_cache: Optional[torch.Tensor] = field(default=None, repr=False)
    _lazy_sequences_built: bool = field(default=False, repr=False)
    # I98-5: LevelsInfo 强类型缓存
    _levels_info_cache: Optional["LevelsInfo"] = field(default=None, repr=False)

    def _build_sequences_from_cache(self) -> None:
        """从缓存张量构造 TokenSequence 列表 (P-OPT-11 Phase 9).

        当 _sequences_data=None 但有 _padded_tokens_cache 时，
        延迟构造 TokenSequence 列表供迭代使用。
        """
        if self._sequences_data is not None or self._padded_tokens_cache is None:
            return  # 已有序列或无缓存

        B = self._padded_tokens_cache.shape[0]
        self._sequences_data = []

        for b in range(B):
            # 从 padded tensor 提取该样本的 tokens
            token_count = int(self._lengths_cache[b]) if self._lengths_cache is not None else self._padded_tokens_cache.shape[1]
            tokens_b = self._padded_tokens_cache[b, :token_count]

            # 构建 metadata
            metadata: Dict[str, Any] = {}
            if self._padded_levels_cache is not None:
                levels_b = self._padded_levels_cache[b, :token_count]
                metadata["levels"] = levels_b

            # 添加 split_stats (与原始 tokenize 行为一致)
            metadata["split_stats"] = {
                "num_tokens": token_count,
                "depth_distribution": {},  # 延迟构建，不在这里计算
            }

            seq = TokenSequence(tokens=tokens_b, metadata=metadata)
            self._sequences_data.append(seq)

        self._lazy_sequences_built = True

    def __iter__(self) -> Iterator[TokenSequence]:
        # P-OPT-11: 延迟构造 sequences
        if self.sequences is None and self._padded_tokens_cache is not None:
            self._build_sequences_from_cache()
        return iter(self.sequences) if self.sequences else iter([])

    def __len__(self) -> int:
        # P-OPT-11: 支持延迟构造时的长度获取
        if self.sequences is not None:
            return len(self.sequences)
        elif self._padded_tokens_cache is not None:
            return self._padded_tokens_cache.shape[0]
        return 0

    @property
    def batch_size(self) -> int:
        """获取批次大小。"""
        return len(self)

    @property
    def sequences(self) -> Optional[List["TokenSequence"]]:
        """获取 TokenSequence 列表，支持延迟构建 (P-OPT-11).

        当 _sequences_data 为 None 但有缓存时，自动触发延迟构建。

        Returns:
            TokenSequence 列表或 None (如果无缓存)
        """
        if self._sequences_data is None and self._padded_tokens_cache is not None:
            self._build_sequences_from_cache()
        return self._sequences_data

    @sequences.setter
    def sequences(self, value: Optional[List["TokenSequence"]]) -> None:
        """设置 sequences，支持 None 值触发延迟构建."""
        self._sequences_data = value

    @property
    def tokens(self) -> torch.Tensor:
        """获取堆叠的 tokens [B, N, D]。

        假设所有样本的 token 数量相同（padding 后）。

        P-OPT-11: 优先使用 _padded_tokens_cache，避免从 sequences 堆叠。

        Returns:
            堆叠的 tokens 张量，形状为 [B, N, D]

        Raises:
            ValueError: 当序列为空时
        """
        # P-OPT-11: 优先使用缓存
        if self._padded_tokens_cache is not None:
            return self._padded_tokens_cache

        if len(self) == 0:
            raise ValueError("Cannot get tokens from empty TokenizerOutput")
        return torch.stack([seq.tokens for seq in self])
    
    def _build_levels_info_tensor(self) -> Optional[torch.Tensor]:
        """内部方法：构建 levels_info 原始张量。

        I98-5: 用于内部实现，返回 raw tensor 供缓存构造使用。

        Returns:
            堆叠的 levels_info 张量或 None
        """
        if len(self) == 0:
            return None

        # 优先使用缓存
        if self._padded_levels_cache is not None:
            return self._padded_levels_cache

        # 使用迭代器 (会自动触发 lazy 构建)
        levels_list = [seq.get_levels() for seq in self]

        # 如果所有 levels 都是 None，返回 None
        if all(l is None for l in levels_list):
            return None

        # 找到最大维度
        if self._padded_tokens_cache is not None:
            device = self._padded_tokens_cache.device
        else:
            device = next(iter(self)).device
        max_len = max(l.shape[0] if l is not None else 0 for l in levels_list)

        if max_len == 0:
            return None

        # 确定 info_dim（处理 1D 和 2D 情况）
        info_dims = [l.shape[1] if l is not None and l.dim() > 1 else 1 for l in levels_list]
        info_dim = max(info_dims)

        # 创建填充后的张量
        stacked = torch.zeros(
            len(self), max_len, info_dim,
            dtype=torch.long, device=device
        )

        for i, levels in enumerate(levels_list):
            if levels is not None:
                if levels.dim() == 1:
                    stacked[i, :levels.shape[0], 0] = levels
                else:
                    stacked[i, :levels.shape[0], :levels.shape[1]] = levels

        return stacked

    @property
    def levels_info(self) -> Optional["LevelsInfo"]:
        """获取 LevelsInfo 强类型（I98-5 增强）。

        直接返回 LevelsInfo 实例，包含四叉树契约验证：
        - C1: depth 范围 [-1, D]
        - C2: path 范围 [0, 3]
        - C3: 路径长度一致性

        使用缓存避免重复构造。

        Returns:
            LevelsInfo 实例或 None（当序列为空时）
        """
        from vit_pytorch.core.levels_info import LevelsInfo

        if len(self) == 0:
            return None

        # 使用缓存
        if self._levels_info_cache is not None:
            return self._levels_info_cache

        # 从缓存张量构造 LevelsInfo
        if self._padded_levels_cache is not None:
            D = self._padded_levels_cache.shape[-1] - 1
            info = LevelsInfo(data=self._padded_levels_cache, max_level=D)
        else:
            # 从 sequences 构造
            all_levels = self._build_levels_info_tensor()
            if all_levels is None:
                return None
            D = all_levels.shape[-1] - 1
            info = LevelsInfo(data=all_levels, max_level=D)

        self._levels_info_cache = info
        return info

    def tokens_list(self) -> List[torch.Tensor]:
        # P-OPT-11: 使用迭代器 (会自动触发 lazy 构建)
        return [seq.tokens for seq in self]

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
        
        # 回退: 使用 _build_levels_info_tensor (I98-5: 避免调用 levels_info 属性)
        levels = self._build_levels_info_tensor()
        if levels is None:
            B = len(self)
            # P-OPT-11: 使用 _padded_tokens_cache 获取 device
            if self._padded_tokens_cache is not None:
                device = self._padded_tokens_cache.device
            else:
                device = next(iter(self)).device if B > 0 else torch.device('cpu')
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
        # P-OPT-11: 使用迭代器 (会自动触发 lazy 构建)
        result: List[torch.Tensor] = []
        for seq in self:
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

    def get_levels_info(self, max_level: int) -> "LevelsInfo":
        """获取 LevelsInfo 实例 (I98-4 新增, I98-5 简化).

        I98-5: 现在直接使用 levels_info 属性（返回 LevelsInfo）。
        仅在 levels_info 为 None 时创建新的 LevelsInfo。

        Args:
            max_level: 四叉树最大深度

        Returns:
            LevelsInfo 实例
        """
        info = self.levels_info
        if info is not None:
            return info

        # I98-5: levels_info 为 None 时创建默认 LevelsInfo
        from vit_pytorch.core.levels_info import LevelsInfo
        B = self.batch_size
        N = 1
        all_levels = torch.zeros(B, N, max_level + 1, dtype=torch.long, device=self.device)
        return LevelsInfo(data=all_levels, max_level=max_level)

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
    """Abstract base class for tokenizers.

    Mathematical Contract
    ====================
    T: R^{B × C × H × W} → TokenizerOutput

    All concrete tokenizers must implement:
    - get_tokens_from_regions(): Support non-streaming tokenization
    - forward(): Alias for tokenize() following nn.Module convention
    """

    @abstractmethod
    def get_tokens_from_regions(
        self,
        images: torch.Tensor,
        regions: torch.Tensor
    ) -> TokenizerOutput:
        """Generate tokens directly from region boundaries.

        Enables non-streaming tokenizer implementations.

        Args:
            images: Input images [B, C, H, W]
            regions: Region boundaries [B, N, 4] in (x1, y1, x2, y2) format

        Returns:
            TokenizerOutput with tokens and metadata
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        """Main tokenization entry point (streaming or batch).

        Args:
            images: Input images [B, C, H, W]

        Returns:
            TokenizerOutput with tokens and metadata
        """
        raise NotImplementedError

    def forward(self, images: torch.Tensor) -> TokenizerOutput:  # type: ignore[override]
        return self.tokenize(images)


class BaseTokenProcessor(nn.Module):
    """Base class for modules that post-process tokenizer outputs.

    Mathematical Contract
    ====================
    P: TokenizerOutput → TokenizerOutput

    Post-processors can:
    - Filter/merge tokens
    - Compute attention masks
    - Aggregate metadata
    """

    @abstractmethod
    def process(self, batch: TokenizerOutput) -> TokenizerOutput:
        """Process tokenizer output.

        Args:
            batch: TokenizerOutput from tokenizer

        Returns:
            Processed TokenizerOutput
        """
        raise NotImplementedError

    def forward(self, batch: TokenizerOutput) -> TokenizerOutput:  # type: ignore[override]
        return self.process(batch)
