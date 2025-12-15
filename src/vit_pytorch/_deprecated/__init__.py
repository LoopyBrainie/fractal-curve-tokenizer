# -*- coding: utf-8 -*-
"""废弃模块包。

此包包含已废弃的模块，保留仅为向后兼容。
这些模块将在 v1.0 中移除。

废弃模块:
    - fractal_curve_tokenizer: BFS + REINFORCE tokenizer
    - token_processor: 独立的 token 处理器

推荐替代:
    - StreamingFractalTokenizer: 统一的流式 tokenizer
    - StreamingFractalTokenizerV2: 带 Gumbel-Softmax 的增强版

迁移指南:
    # 旧代码
    from vit_pytorch import FractalHilbertTokenizer, EnhancedFractalTokenProcessor
    tokenizer = FractalHilbertTokenizer(...)
    processor = EnhancedFractalTokenProcessor(...)
    
    # 新代码
    from vit_pytorch import StreamingFractalTokenizerV2
    tokenizer = StreamingFractalTokenizerV2(...)
    # 无需单独的 processor，功能已集成
"""

import warnings

# 发出导入警告
warnings.warn(
    "正在导入废弃模块。请迁移到 StreamingFractalTokenizer。"
    "详见 examples/training/STREAMING_TOKENIZER_GUIDE.md",
    DeprecationWarning,
    stacklevel=2,
)

from .fractal_curve_tokenizer import (
    FractalHilbertTokenizer,
    LearnableSplitDecision,
    MiniCNN,
    PatchInfo,
)
from .token_processor import EnhancedFractalTokenProcessor

__all__ = [
    "FractalHilbertTokenizer",
    "EnhancedFractalTokenProcessor",
    "LearnableSplitDecision",
    "MiniCNN",
    "PatchInfo",
]
