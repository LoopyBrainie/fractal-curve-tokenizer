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
    sequences: List[TokenSequence]

    def __iter__(self) -> Iterator[TokenSequence]:
        return iter(self.sequences)

    def __len__(self) -> int:
        return len(self.sequences)

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
