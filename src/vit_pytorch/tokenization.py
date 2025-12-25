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
from typing import Any, Dict, Iterator, List, Optional

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
    
    Attributes:
        sequences: 各样本的 TokenSequence 列表
        
    Properties:
        tokens: 堆叠的 tokens [B, N, D]（假设所有样本 token 数量相同）
        levels_info: 堆叠的 levels_info [B, N, K] 或 None
        batch_size: 批次大小
    """
    sequences: List[TokenSequence]

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
        
        Returns:
            堆叠的 levels_info 张量或 None
        """
        if len(self.sequences) == 0:
            return None
        
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

    def levels_list(self) -> List[torch.Tensor]:
        result: List[torch.Tensor] = []
        for seq in self.sequences:
            levels = seq.get_levels()
            if levels is None:
                result.append(torch.empty(0, dtype=torch.long, device=seq.device))
            else:
                result.append(levels)
        return result

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
