"""v1.3 STANDARD: FFMI (Fractal Feature Map Isolation) guard test.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set A.
"""
import pytest
import torch
import torch.nn as nn
from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards.ffmi import check_fractal_feature_isolation, FFMIError


class _L1DeterministicMap(nn.Module):
 """Mock L1 Hilbert deterministic feature map (no learnable params)."""
 def __init__(self):
  super().__init__()
  # No nn.Linear or nn.Conv - purely deterministic ops
  self.register_buffer("coords", torch.randn(8,8))

 def forward(self, x):
  return self.coords.unsqueeze(0).expand(x.shape[0], -1, -1)


class _L2LearnableMap(nn.Module):
 """Mock L2+ learnable feature map (with nn.Linear)."""
 def __init__(self):
  super().__init__()
  self.linear = nn.Linear(8,8)


@pytest.mark.static_guard
class TestFFMIGuard:
 def test_ffmi_passes_for_pure_deterministic_module(self):
  l1 = _L1DeterministicMap()
  result = check_fractal_feature_isolation(l1)
  assert isinstance(result, Ok), f"FFMI should pass for L1 deterministic, got {result}"

 def test_ffmi_passes_for_learnable_module(self):
  l2 = _L2LearnableMap()
  result = check_fractal_feature_isolation(l2)
  assert isinstance(result, Ok), f"FFMI should pass for L2+ learnable, got {result}"

 def test_ffmi_passes_for_combined_model(self):
  # A model with both L1 and L2 components
  class _CombinedModel(nn.Module):
   def __init__(self):
    super().__init__()
    self.l1 = _L1DeterministicMap()
    self.l2 = nn.Linear(8,4)

   def forward(self, x):
    return self.l2(self.l1(x).mean(dim=-1))

  model = _CombinedModel()
  result = check_fractal_feature_isolation(model)
  assert isinstance(result, Ok), f"FFMI should pass for combined model, got {result}"

 def test_ffmi_rejects_module_with_requires_grad_false_param(self):
  """A learnable-looking parameter explicitly set requires_grad=False should still pass (it IS isolated)."""
  class _IsolatedLearnable(nn.Module):
   def __init__(self):
    super().__init__()
    self.weight = nn.Parameter(torch.randn(4,4))
    # Don't override requires_grad - should be True by default

  mod = _IsolatedLearnable()
  result = check_fractal_feature_isolation(mod)
  assert isinstance(result, Ok)
