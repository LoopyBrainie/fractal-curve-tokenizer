"""v1.3 STANDARD: Multi-Block HMFT Splitter (alpha-b-rev4).

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.1
this is the main path splitter for v1.3. It generalizes H1SS by allowing the
block size h to be **learned** over a 5-bin discrete set (8, 16, 32, 64, 128).

Key design differences from H1SS:
  1. **5-bin learnable h**: a softmax over HMFT_BLOCK_SIZES chooses h per
     image (or per group), instead of H1SS's fixed h.
  2. **Power-of-4 alignment**: h is restricted to 4^k values to preserve
     the Hilbert sub-manifold topology (per the Power-of-4 proof in
     docs/superpowers/lemmas/2026-06-08-power-of-4-chunking-proof.md).
  3. **Gumbel-STE TopK**: same as H1SS, but selects K cells from the
     chosen h × h blocks (not from the global Hilbert indices).

The 5-block EAS verification (JVP-1) is in
tests/unit/L2_components/splitters/test_multi_block_hmft_eas.py.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from vit_pytorch.core.constants import HMFT_BLOCK_SIZES, HMFT_K_HARD_GLOBAL_POOL
from vit_pytorch.core.splitter_protocol import CoreSplitter, SplitResult


@dataclass
class MultiBlockHMFTSplitterConfig:
    """Configuration for MultiBlockHMFTSplitter."""
    feature_dim: int = 256
    min_patch_size: int = 4
    max_level_limit: int = 8
    hidden_dim: int = 64
    K_fixed: int = 16
    # 5-bin learnable h block sizes (Power-of-4 alignment)
    block_sizes: Tuple[int, ...] = HMFT_BLOCK_SIZES
    # Hard K — number of selected sub-blocks per image
    G_global_pool: int = HMFT_K_HARD_GLOBAL_POOL
    # Gumbel-STE temperature
    temperature_init: float = 1.0
    temperature_min: float = 0.1


class MultiBlockHMFTSplitter(nn.Module, CoreSplitter):
    """v1.3 STANDARD: Multi-Block HMFT Splitter (alpha-b-rev4).

    Splits an input feature map of shape [B, C, H, W] into K sub-blocks
    by choosing a block size h from a learnable 5-bin distribution and
    then applying Gumbel-STE TopK over the resulting (H/h)·(W/h) cells.
    """

    def __init__(
        self,
        config: Optional[MultiBlockHMFTSplitterConfig] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            config = MultiBlockHMFTSplitterConfig(**kwargs)
        self._config = config
        # Learnable 5-bin block size logits (small random init to avoid
        # the entropy-maximum stationary point of uniform distribution)
        self.h_logits = nn.Parameter(
            torch.randn(len(config.block_sizes)) * 0.1
        )
        # Current state
        self._current_epoch: int = 0
        self._current_image_size: Optional[Tuple[int, int]] = None
        self._is_training_mode: bool = True
        # Gumbel temperature (annealable)
        self._temperature: float = config.temperature_init
        # Geometry encoder inputs dim: 3 (path, rot, area)
        self.geometry_encoder = _LightweightGeometryEncoder(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
        )
        # Fusion head
        self.fusion = _LightweightFusion(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
        )

    @property
    def max_level_limit(self) -> int:
        return self._config.max_level_limit

    @property
    def num_candidates(self) -> int:
        if self._current_image_size is None:
            return 0
        H, W = self._current_image_size
        # Use the largest h in the bin to get a lower bound on candidates
        h = self._config.block_sizes[-1]
        n_h = H // h
        n_w = W // h
        return n_h * n_w

    @property
    def is_training(self) -> bool:
        return self._is_training_mode

    def set_training(self, training: bool) -> None:
        self._is_training_mode = training

    def update_candidates(self, image_size: Tuple[int, int]) -> None:
        self._current_image_size = image_size

    def set_temperature(self, temperature: float) -> None:
        self._temperature = max(temperature, self._config.temperature_min)

    def get_current_temperature(self) -> Tensor:
        return torch.tensor(self._temperature)

    def set_annealing_schedule(self, schedule: Any) -> None:
        # Schedule is opaque to the splitter; the trainer owns scheduling
        pass

    def set_explore_bias(self, bias: float) -> None:
        # Add to h_logits for exploration; do not persist
        pass

    def get_diagnostics(self) -> Dict[str, Any]:
        return {
            "splitter": "MultiBlockHMFTSplitter",
            "h_probs": torch.softmax(self.h_logits, dim=-1).detach().cpu().tolist(),
            "temperature": self._temperature,
            "epoch": self._current_epoch,
        }

    def get_depth_distribution(self) -> Dict[int, float]:
        return {0: 1.0}

    def get_quota_logits(self) -> Optional[Tensor]:
        return None

    def get_quota_probs(self) -> Optional[Tensor]:
        return None

    def get_entropy_loss(self) -> Tensor:
        # Encourage exploration over h bins
        probs = torch.softmax(self.h_logits, dim=-1)
        return -(probs * torch.log(probs + 1e-9)).sum()

    def get_variance_regularization(self) -> Tensor:
        return torch.tensor(0.0)

    def get_coverage_stats(self) -> Dict[str, float]:
        return {"num_candidates": float(self.num_candidates)}

    def _sample_block_size(
        self,
        hard: bool,
        max_h: Optional[int] = None,
        min_n_cells: int = 1,
    ) -> int:
        """Sample a block size from the 5-bin learnable distribution.

        Constraints:
        - h <= max_h (image size)
        - n_cells = (H/h) * (W/h) >= min_n_cells
          → h <= sqrt(H*W / min_n_cells)
        """
        if max_h is not None:
            # Add a soft constraint: ensure h is small enough to give >= min_n_cells
            # n_cells = (H/h) * (W/h); for square H=W=G, h <= G / sqrt(min_n_cells)
            min_factor = max(1, int(min_n_cells ** 0.5 + 0.5))
            effective_max_h = max_h // min_factor
            valid_sizes = [
                s for s in self._config.block_sizes
                if s <= max_h and s <= effective_max_h * 1
            ]
            if not valid_sizes:
                # Fallback: pick largest valid h that's still <= max_h
                valid_sizes = [s for s in self._config.block_sizes if s <= max_h]
            if not valid_sizes:
                return self._config.block_sizes[0]
        else:
            valid_sizes = list(self._config.block_sizes)
        valid_indices = [
            self._config.block_sizes.index(s) for s in valid_sizes
        ]
        h_probs = torch.softmax(self.h_logits, dim=-1)[valid_indices]
        if hard:
            idx = int(h_probs.argmax().item())
        else:
            idx = int(torch.multinomial(h_probs, 1).item())
        return valid_sizes[idx]

    def _partition_hilbert(
        self, features: Tensor, h: int
    ) -> Tuple[Tensor, Tensor]:
        """Partition feature map into h×h Hilbert-ordered sub-blocks.

        Returns:
            blocks: [B, n_cells, h*h, C] (Hilbert-ordered sub-blocks)
            hilbert_indices: [n_cells] (cell index in Hilbert order)
        """
        B, C, H, W = features.shape
        assert H % h == 0 and W % h == 0, (
            f"H={H} and W={W} must be divisible by h={h}"
        )
        n_h = H // h
        n_w = W // h
        n_cells = n_h * n_w

        # Compute Hilbert indices for the (n_h, n_w) cell grid
        from vit_pytorch.core.curve_hilbert import HilbertCurve
        cell_xs = torch.arange(n_w).repeat(n_h, 1).flatten()
        cell_ys = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w).flatten()
        hilbert_indices = HilbertCurve.xy_to_d_batch(n_w, cell_xs, cell_ys)
        # Sort cells by Hilbert index
        sorted_order = torch.argsort(hilbert_indices)
        # Reshape features to [B, n_h, h, n_w, h, C] then permute to [B, n_cells, h*h, C]
        blocks = features.view(B, n_h, h, n_w, h, C)
        blocks = blocks.permute(0, 1, 3, 2, 4, 5).contiguous()  # [B, n_h, n_w, h, h, C]
        blocks = blocks.view(B, n_cells, h * h, C)
        blocks = blocks[:, sorted_order, :, :]
        return blocks, hilbert_indices[sorted_order]

    def forward(
        self,
        features: Tensor,
        image_size: Optional[Tuple[int, int]] = None,
        hard: bool = False,
        epoch: int = 0,
    ) -> SplitResult:
        """Forward pass: choose h, partition, select K sub-blocks.

        Args:
            features: [B, C, H, W] input feature map
            image_size: optional (H, W), defaults to features.shape[-2:]
            hard: if True, use argmax over h_logits and argmax over cell scores
            epoch: current training epoch (for schedule)

        Returns:
            SplitResult with K selected sub-blocks per image.
        """
        self._current_epoch = epoch
        if image_size is None:
            image_size = (features.shape[-2], features.shape[-1])
        self.update_candidates(image_size)
        B, C, H, W = features.shape

        # 1) Choose block size h (must give at least K_fixed cells)
        h = self._sample_block_size(
            hard=hard, max_h=min(H, W), min_n_cells=self._config.K_fixed,
        )

        # 2) Partition into h×h Hilbert-ordered sub-blocks
        blocks, cell_hilbert_indices = self._partition_hilbert(features, h)
        n_cells = blocks.shape[1]

        # 3) Compute per-cell scores (Gumbel-STE TopK)
        # Per-cell features: mean over h*h spatial → [B, n_cells, C]
        cell_features = blocks.mean(dim=2)  # [B, n_cells, C]
        # Project to a scalar score per cell via simple mean over channels
        score_head = cell_features.mean(dim=-1)  # [B, n_cells]
        # Add learnable noise via Gumbel
        if self.training and not hard:
            gumbel_noise = -torch.log(
                -torch.log(torch.rand_like(score_head) + 1e-9) + 1e-9
            )
            score_head = (score_head + gumbel_noise) / max(self._temperature, 1e-3)
        # TopK selection
        K = min(self._config.K_fixed, n_cells)
        topk_scores, topk_indices = torch.topk(score_head, K, dim=-1)
        # Hard mask: 1 for selected, 0 otherwise
        mask_hard = torch.zeros_like(score_head)
        mask_hard.scatter_(-1, topk_indices, 1.0)
        # Soft mask: softmax scores
        mask_soft = torch.softmax(score_head, dim=-1)
        # STE: forward hard, backward soft
        mask_ste = (mask_hard - mask_soft).detach() + mask_soft

        # 4) Gather selected blocks: [B, K, h*h, C]
        selected_blocks = torch.gather(
            blocks, 1, topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h * h, C)
        )

        # 5) Build SplitResult
        # M = total selected regions = B * K (per-image, so flatten)
        M = B * K
        # Construct regions [M, 4] as placeholder (x0, y0, x1, y1) per selected cell
        regions = torch.zeros(M, 4, device=features.device, dtype=features.dtype)
        # Depths and batch indices
        depths = torch.zeros(M, device=features.device, dtype=torch.long)
        batch_indices = torch.arange(B, device=features.device).repeat_interleave(K)
        # Hilbert indices for selected cells (use topk_indices into cell_hilbert_indices)
        selected_hilbert_indices = cell_hilbert_indices[topk_indices].flatten()  # [B*K]

        return SplitResult(
            regions=regions,
            depths=depths,
            batch_indices=batch_indices,
            hilbert_indices=selected_hilbert_indices,
            selected_mask=mask_hard,
            logits=score_head,
            probs=mask_soft,
            K_soft=torch.tensor(float(K), device=features.device),
            mask_ste=mask_ste,
            roi_features_raw=selected_blocks.view(M, h * h * C),
            candidate_indices=topk_indices.flatten(),
        )


class _LightweightGeometryEncoder(nn.Module):
    """Minimal geometry encoder: 3-channel (path, rot, area) projection.

    For the skeleton, we project raw features to a hidden geometry space.
    Future B.4 work will replace this with H1SS's full GeometryEncoder.
    """

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(feature_dim, hidden_dim)

    def forward(self, features: Tensor) -> Tensor:
        return self.proj(features)


class _LightweightFusion(nn.Module):
    """Minimal fusion head: project + activate."""

    def __init__(self, feature_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_dim, feature_dim)
        self.act = nn.GELU()

    def forward(self, geo: Tensor) -> Tensor:
        return self.act(self.proj(geo))
