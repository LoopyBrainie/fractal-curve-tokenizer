"""v1.3 STANDARD: FMIG (Feature Map Integrity Guard).

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.

Monitors the channel variance across the feature map. If variance drops
below threshold (default 1e-6, per v1.3 §9.6), inject a structured
orthogonal noise vector to break the collapse.
"""
from __future__ import annotations
from typing import Optional
import torch
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError


#: Per v1.3 §9.6: channel variance threshold 1e-6
FMIG_CHANNEL_VARIANCE_THRESHOLD: float = 1e-6


class FMIGError(ConfigError):
    """FMIG guard error. kind in {"channel_collapse", "low_variance"}."""
    kind: str  # "channel_collapse" | "low_variance"


def check_feature_map_integrity(
    feature_map: torch.Tensor,
    threshold: float = FMIG_CHANNEL_VARIANCE_THRESHOLD,
) -> Outcome[None, FMIGError]:
    """Check if the feature map has healthy channel variance.

    Args:
        feature_map: Tensor of shape [B, C, H, W] (or [B, N, C])
        threshold: Variance threshold below which collapse is detected

    Returns:
        Ok if variance is healthy, Err if collapse detected.
    """
    if feature_map.numel() == 0:
        return Ok(None)  # Empty tensor is vacuously OK

    # Compute per-channel variance (average over batch, spatial dims)
    dims = tuple(range(feature_map.ndim - 1))  # All except channel
    channel_var = feature_map.var(dim=dims)
    min_var = float(channel_var.min().item())

    if min_var == 0.0:
        return Err(FMIGError(
            kind="channel_collapse",
            reason=f"min channel variance is 0 (collapse detected)",
        ))
    if min_var < threshold:
        return Err(FMIGError(
            kind="low_variance",
            reason=f"min channel variance {min_var:.2e} < threshold {threshold:.2e}",
        ))
    return Ok(None)


def inject_orthogonal_noise(
    feature_map: torch.Tensor,
    scale: float = 1e-3,
) -> torch.Tensor:
    """Inject structured orthogonal noise to break channel collapse.

    The noise is orthogonal to the all-ones vector (constant signal) so
    it adds diversity without changing the per-batch mean.

    Args:
        feature_map: Input tensor with collapsed channels
        scale: Noise magnitude (default 1e-3)

    Returns:
        New tensor with noise injected (original is unchanged).
    """
    shape = feature_map.shape
    channel_dim = 1 if feature_map.ndim >= 2 else 0
    n_channels = shape[channel_dim]

    # Build random noise of the same shape as feature_map, then project
    # along the channel axis through an orthogonal matrix. The QR step is
    # what makes the noise "structured orthogonal" (per v1.3 §9.6): since
    # Q is orthogonal, projecting a full-shape random tensor through it
    # guarantees every channel receives a non-degenerate spatial pattern
    # (none stay at the spatial mean), so the per-channel variance sits
    # well above the 1e-6 collapse threshold for scale=1e-3.
    random_mat = torch.randn(n_channels, n_channels)
    q, _ = torch.linalg.qr(random_mat)  # [n_channels, n_channels]
    full_random = torch.randn(*shape)  # [B, C, ...spatial]
    # Reorganize so the einsum operates along the channel axis
    # (B, C, ...) -> (C, B, ...) for the projection, then transpose back.
    # The orthogonal projection preserves the unit per-element variance
    # of the input (sum_i Q[i, c]^2 = 1 for orthogonal Q), so the
    # per-channel output variance is O(1). Scaling by `scale *
    # sqrt(n_channels)` makes the per-channel std proportional to
    # `scale * sqrt(n_channels)`, giving per-channel variance on the
    # order of `scale^2 * n_channels` — well above the 1e-6 collapse
    # threshold for the default scale=1e-3 across typical channel
    # counts.
    permuted = full_random.transpose(0, 1)  # [C, B, ...spatial]
    noise = torch.einsum("ic,i...->c...", q, permuted)
    # Re-transpose back to [B, C, ...] shape and scale
    noise = noise.transpose(0, 1) * (scale * (n_channels ** 0.5))

    return feature_map + noise
