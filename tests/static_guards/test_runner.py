"""v1.3 STANDARD: Static guard runner integration tests.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6.

The runner aggregates all 8 guards. These tests verify the runner:
1. Passes for a well-formed model
2. Catches errors when a guard fails
3. The CI-blocking assertion raises on any failure
"""
import pytest
import torch
import torch.nn as nn

from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards import (
    run_all_guards,
    assert_all_guards_pass,
    GUARD_NAMES,
)


class _WellFormedModel(nn.Module):
 """Mock well-formed model that should pass all 8 guards."""
 def __init__(self):
  super().__init__()
  self.linear = nn.Linear(8, 8)

 def forward(self, x: torch.Tensor) -> torch.Tensor:
  return self.linear(x)


@pytest.mark.static_guard
class TestGuardRunner:
 def test_runner_returns_dict_with_all_8_guards(self):
  """Runner must return a dict with all 8 guard names as keys."""
  model = _WellFormedModel()
  results = run_all_guards(model)
  assert set(results.keys()) == set(GUARD_NAMES)
  assert len(results) == 8

 def test_runner_passes_for_well_formed_model(self):
  """A well-formed model should pass all 8 guards."""
  model = _WellFormedModel()
  results = run_all_guards(model)
  failures = {name: r for name, r in results.items() if isinstance(r, Err)}
  assert len(failures) == 0, f"Unexpected failures: {failures}"

 def test_runner_catches_sma_failure(self):
  """A model with invalid _state should fail SMA."""

  class _BadStateModel(nn.Module):
   def __init__(self):
    super().__init__()
    self._state = "INVALID_STATE"

  model = _BadStateModel()
  results = run_all_guards(model)
  assert isinstance(results["SMA"], Err)
  # Other guards may or may not pass; we only check SMA catches the issue
  assert "invalid_state_value" in str(results["SMA"].error.kind)

 def test_runner_catches_pcar_failure(self):
  """A model with inverted _saturation bounds should fail PCAR."""

  class _InvertedSatModel(nn.Module):
   def __init__(self):
    super().__init__()
    self._saturation_min = 0.95
    self._saturation_max = -0.95  # Inverted!

  model = _InvertedSatModel()
  results = run_all_guards(model)
  assert isinstance(results["PCAR"], Err)

 def test_assert_all_guards_pass_succeeds_for_well_formed(self):
  """assert_all_guards_pass() should NOT raise for a well-formed model."""
  model = _WellFormedModel()
  # Should not raise
  assert_all_guards_pass(model)

 def test_assert_all_guards_pass_raises_on_failure(self):
  """assert_all_guards_pass() should raise RuntimeError on any failure."""

  class _BadStateModel(nn.Module):
   def __init__(self):
    super().__init__()
    self._state = "BAD_STATE"

  model = _BadStateModel()
  with pytest.raises(RuntimeError) as exc_info:
   assert_all_guards_pass(model)
  # Error message should mention which guard(s) failed
  assert "SMA" in str(exc_info.value) or "guard" in str(exc_info.value).lower()

 def test_guard_names_includes_both_sets(self):
  """GUARD_NAMES must include all 4 Set A and 4 Set B guards."""
  set_a = {"SMA", "DGC", "FFMI", "PCAR"}
  set_b = {"TMA", "ADGC", "FMIG", "PCA"}
  assert set_a.issubset(set(GUARD_NAMES))
  assert set_b.issubset(set(GUARD_NAMES))
