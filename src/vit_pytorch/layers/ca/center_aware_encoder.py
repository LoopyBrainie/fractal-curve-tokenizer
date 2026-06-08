"""CenterAwareEncoder — gates the Polar Voronoi secondary path.

Mathematical form
=================

The Polar Voronoi splitter is a *secondary* path that complements H1SS.
It is only useful when the local feature activation pattern is
"center-biased" — i.e., the brightest cell sits at the center of a
5-cell Voronoi mask and the four NSEW neighbors have comparable
magnitude. In arbitrary natural-image regions this condition fails
(edges, corners, low-frequency backgrounds) and Voronoi selection would
inject noise.

We therefore attach a small ``CenterAwareEncoder`` that, given pooled
ROI features, returns:

    is_center_biased: bool
        Hard decision — used to switch the secondary path on/off.
    confidence:       float in [0, 1]
        Soft confidence — used to scale the secondary loss.

The encoder is a depthwise conv (spatial Laplacian-like 3x3) followed
by a single linear projection. The bias of the linear layer is
initialised so that ``sigmoid(b) ≈ 0.5`` at init — i.e., the model
starts agnostic and only learns to gate once the data is observed.

The encoder is intentionally small (< 1K params) and lives in
``src/vit_pytorch/layers/ca/`` (L2 layer, *not* L1) so that it can
import from ``vit_pytorch.core.constants`` and the splitter protocol
without circular dependency.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# 3x3 Laplacian-like kernel that fires when center > mean(NSEW).
# This is the same shape as the "5-cell Voronoi" mask (center + 4
# NSEW neighbours) used in WBA evaluation, so the encoder's hard
# decision is geometrically consistent with the WBA computation.
_CENTER_BIAS_KERNEL = torch.tensor(
    [
        [0.0, -0.25, 0.0],
        [-0.25, 1.0, -0.25],
        [0.0, -0.25, 0.0],
    ],
    dtype=torch.float32,
).view(1, 1, 3, 3)


class CenterAwareEncoder(nn.Module):
    """Lightweight center-bias detector for the Polar Voronoi path.

    Args:
        in_channels: feature channels of the input ROI features
            (typically 256, matching HilbertOptimalSplitter.feature_dim).
        threshold: confidence threshold above which ``is_center_biased``
            flips to True. Default 0.5 (sigmoid midpoint of the bias
            init below).

    Forward:
        features: [B, C, H, W] ROI features pooled from a 5-cell
            Voronoi mask window.

    Returns:
        is_center_biased: [B] bool tensor — hard gating decision.
        confidence:        [B] float tensor in [0, 1] — soft scaling.
    """

    def __init__(self, in_channels: int = 256, threshold: float = 0.5) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.threshold = threshold

        # Depthwise spatial filter — single channel input/output.
        # Initialised with the 3x3 center-bias kernel above so that the
        # encoder starts as a *fixed* Laplacian, not random noise.
        # Trained as a per-channel scaling (depthwise) to allow the
        # encoder to up- or down-weight specific feature channels.
        self.spatial = nn.Conv2d(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        with torch.no_grad():
            self.spatial.weight.copy_(_CENTER_BIAS_KERNEL)

        # Channel mixer (1x1 conv) + linear projection to logit.
        # We project the spatially-filtered response to a single
        # scalar per batch element.
        self.channel_mix = nn.Conv2d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=1,
            bias=False,
        )
        nn.init.zeros_(self.channel_mix.weight)

        # Per-batch logit: positive -> center-biased, negative -> not.
        # Bias init = 0 -> sigmoid(0) = 0.5 -> threshold-based decision
        # is well-defined at init.
        self.logit_bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Return (is_center_biased, confidence)."""
        # features: [B, C, H, W]
        if features.dim() != 4:
            raise ValueError(
                f"CenterAwareEncoder expects [B, C, H, W]; got {tuple(features.shape)}"
            )

        # Spatial filter (3x3) applied per-channel via depthwise group conv.
        # Replicate the same Conv2d(1,1,k=3) across channels with groups=C.
        B, C, H, W = features.shape
        weight = self.spatial.weight  # [1, 1, 3, 3]
        depthwise_weight = weight.expand(C, 1, 3, 3).contiguous()
        spatial_out = F.conv2d(
            features,
            depthwise_weight,
            bias=None,
            stride=1,
            padding=1,
            groups=C,
        )

        # Mix channels down to a single logit per spatial location.
        mixed = self.channel_mix(spatial_out)  # [B, 1, H, W]
        mixed = mixed + self.logit_bias  # broadcast bias
        logit = mixed.mean(dim=(-1, -2))   # [B, 1] -> [B] via mean
        logit = logit.squeeze(-1)

        confidence = torch.sigmoid(logit)
        is_center_biased = confidence > self.threshold
        return is_center_biased, confidence

    def extra_repr(self) -> str:
        return (
            f"CenterAwareEncoder(in_channels={self.in_channels}, "
            f"threshold={self.threshold})"
        )
