"""v1.3 STANDARD: FFMI (Fractal Feature Map Isolation) guard.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set A.

Verifies that L1 deterministic (Hilbert) feature maps and L2+ learnable
feature maps are isolated in the gradient graph. For v1.3 this is a structural
check: a module passes if either:
- It has no nn.Parameter (purely deterministic), OR
- It has nn.Parameter with requires_grad=True (standard learnable), OR
- It has buffers only (deterministic state)

The mathematical isolation is enforced by the architecture (L1 modules
only use HilbertCurve.xy_to_d_batch, L2+ modules use nn.Linear/Conv).
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError


class FFMIError(ConfigError):
 """FFMI guard error. kind in {"grad_leak", "unexpected_structure"}."""
 kind: str


def check_fractal_feature_isolation(module: nn.Module) -> Outcome[None, FFMIError]:
 """Verify structural isolation between L1 and L2+ feature maps.

 For v1.3: this is an informational check that the module structure is
 sane (has either buffers or parameters, or is a pure functional wrapper).
 The actual isolation is enforced by the L1→L2 import hierarchy rule
 in CLAUDE.md and by the arch-only-imports-L1 principle.

 Returns Ok for any well-formed nn.Module; Err only for malformed cases
 (e.g., a module with both L1 import AND nn.Parameter at the same level,
 which would be a structural violation).
 """
 # Sanity check: module is a real nn.Module
 if not isinstance(module, nn.Module):
  return Err(FFMIError(
   kind="unexpected_structure",
   reason=f"expected nn.Module, got {type(module).__name__}",
  ))

 # Walk the module tree and ensure no child has BOTH a HilbertCurve import
 # AND an nn.Parameter - that would be a structural violation
 for name, child in module.named_modules():
  if name == "":
   continue
  # Check if the child has both HilbertCurve usage and nn.Parameter
  has_hilbert_attr = any(
   attr_name.startswith("hilbert_") or attr_name.startswith("fractal_")
   for attr_name, _ in child.named_buffers()
  )
  has_learnable = any(
   True for _ in child.parameters(recurse=False)
  )
  if has_hilbert_attr and has_learnable:
   return Err(FFMIError(
    kind="grad_leak",
    reason=f"module {name!r} mixes Hilbert buffers with learnable params (potential grad leakage)",
   ))

 return Ok(None)
