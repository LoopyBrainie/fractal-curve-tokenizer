from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AdvancedFractalPositionEmbedding(nn.Module):
    """
    高级分形位置编码，完全对齐增强tokenizer
    支持动态层级、Hilbert路径编码和多尺度空间感知
    """

    def __init__(
        self,
        dim: int,
        max_level: int = 50,
        max_seq_len: int = 10000,
        use_hilbert_encoding: bool = True,
        use_spatial_encoding: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_level = max_level
        self.max_seq_len = max_seq_len
        self.use_hilbert_encoding = use_hilbert_encoding
        self.use_spatial_encoding = use_spatial_encoding

        self.depth_embedding = nn.Embedding(max_level + 1, dim)

        if use_hilbert_encoding:
            self.path_embedding = nn.Embedding(max_seq_len, dim)
            self.path_encoder = nn.LSTM(
                input_size=dim,
                hidden_size=dim // 2,
                num_layers=2,
                batch_first=True,
                bidirectional=True,
            )

        if use_spatial_encoding:
            self.spatial_embedding_2d = nn.Parameter(torch.randn(1, max_seq_len, dim))
            self.scale_embedding = nn.Embedding(max_level + 1, dim)

        input_dim = dim
        if use_hilbert_encoding:
            input_dim += dim
        if use_spatial_encoding:
            input_dim += dim * 2

        self.fusion_network = nn.Sequential(
            nn.Linear(input_dim, dim * 2),
            nn.LayerNorm(dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

        self.encoding_weights = nn.Parameter(torch.ones(4))
        self.level_attention_bias = nn.Parameter(torch.zeros(max_level + 1, max_level + 1))

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.depth_embedding.weight, std=0.02)
        if self.use_hilbert_encoding:
            nn.init.normal_(self.path_embedding.weight, std=0.02)
        if self.use_spatial_encoding:
            nn.init.normal_(self.spatial_embedding_2d, std=0.02)
            nn.init.normal_(self.scale_embedding.weight, std=0.02)

        nn.init.uniform_(self.level_attention_bias, -0.1, 0.1)

    def forward(
        self,
        levels_info: torch.Tensor,
        sequence_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if levels_info.numel() == 0:
            return torch.zeros(0, self.dim, device=levels_info.device, dtype=torch.float32)

        device = levels_info.device
        seq_len = levels_info.shape[0]

        depths = levels_info[:, 0].clamp(0, self.max_level)
        depth_emb = self.depth_embedding(depths)

        embeddings = [depth_emb]

        if self.use_hilbert_encoding and levels_info.shape[1] > 1:
            path_info = levels_info[:, 1:]
            path_embeddings = []
            for i in range(path_info.shape[1]):
                path_indices = path_info[:, i].clamp(0, self.max_seq_len - 1)
                path_emb = self.path_embedding(path_indices)
                path_embeddings.append(path_emb)

            if path_embeddings:
                path_sequence = torch.stack(path_embeddings, dim=1)
                path_encoded, _ = self.path_encoder(path_sequence)
                path_final = path_encoded[:, -1, :]
                embeddings.append(path_final)
            else:
                embeddings.append(torch.zeros_like(depth_emb))

        if self.use_spatial_encoding:
            if sequence_positions is not None:
                seq_pos = sequence_positions.clamp(0, self.max_seq_len - 1)
            else:
                seq_pos = torch.arange(seq_len, device=device).clamp(0, self.max_seq_len - 1)

            spatial_emb = self.spatial_embedding_2d[0, seq_pos, :]
            embeddings.append(spatial_emb)

            scale_emb = self.scale_embedding(depths)
            embeddings.append(scale_emb)

        combined_emb = torch.cat(embeddings, dim=-1)
        fused_emb = self.fusion_network(combined_emb)

        if len(embeddings) == 4:
            weighted_emb = (
                embeddings[0] * self.encoding_weights[0]
                + embeddings[1] * self.encoding_weights[1]
                + embeddings[2] * self.encoding_weights[2]
                + embeddings[3] * self.encoding_weights[3]
            ) / self.encoding_weights.sum()
        else:
            weighted_emb = fused_emb

        final_emb = fused_emb + weighted_emb * 0.1
        return final_emb

    def get_attention_bias(self, depths: torch.Tensor) -> torch.Tensor:
        num_tokens = len(depths)
        bias = torch.zeros(num_tokens, num_tokens, device=depths.device)

        for i in range(num_tokens):
            for j in range(num_tokens):
                depth_i = depths[i].clamp(0, self.max_level).long()
                depth_j = depths[j].clamp(0, self.max_level).long()
                bias[i, j] = self.level_attention_bias[depth_i, depth_j]

        return bias
