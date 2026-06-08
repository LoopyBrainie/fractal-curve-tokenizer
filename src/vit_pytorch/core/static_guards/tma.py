"""v1.3 STANDARD: TMA (Telemetry Monitor Alignment) guard.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.

Verifies that the telemetry interface (xxx_output properties like
monitor_output, embed_output, attn_output, ffn_output) returns a dict
with valid types. Passes vacuously for modules without telemetry.

Note: This is a static structural check. Runtime alignment of telemetry
with shadow pipeline evaluations is enforced by the Shadow Monitor
(docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.7).
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError


class TMAError(ConfigError):
    """TMA guard error. kind in {"non_dict_telemetry", "invalid_telemetry_field"}."""
    kind: str  # "non_dict_telemetry" | "invalid_telemetry_field"


def check_telemetry_alignment(module: nn.Module) -> Outcome[None, TMAError]:
    """Verify module's telemetry interface returns valid dict.

    Convention: any property named `xxx_output` (e.g., `monitor_output`,
    `embed_output`, `attn_output`, `ffn_output`, `splitter_output`)
    should return a dict. The keys should be strings, and values
    should be scalar numerics (int, float, torch.Tensor scalar).
    """
    # Inspect all properties
    for attr_name in dir(module):
        if not attr_name.endswith("_output"):
            continue
        if attr_name.startswith("_"):  # Skip dunder/private
            continue
        try:
            value = getattr(module, attr_name)
        except Exception:
            continue  # Properties that raise are non-blocking
        if callable(value):
            continue  # Skip methods
        if value is None:
            continue  # Skip None (informational)
        if not isinstance(value, dict):
            return Err(TMAError(
                kind="non_dict_telemetry",
                reason=f"{attr_name} must be dict, got {type(value).__name__}",
            ))
        for key, val in value.items():
            if not isinstance(key, str):
                return Err(TMAError(
                    kind="invalid_telemetry_field",
                    reason=f"{attr_name} key must be str, got {type(key).__name__}",
                ))
    return Ok(None)
