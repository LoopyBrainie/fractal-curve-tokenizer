"""v1.3 STANDARD: SMA (State Machine Alignment) guard test.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set A.
"""
import pytest
import torch
import torch.nn as nn
from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards.sma import check_state_machine_alignment, SMAError


class _MockStatefulModule(nn.Module):
    """Mock module with a _state attribute for SMA testing."""
    def __init__(self, state: str = "INIT"):
        super().__init__()
        self._state = state


@pytest.mark.static_guard
class TestSMAGuard:
    def test_sma_passes_valid_state(self):
        mod = _MockStatefulModule(state="INIT")
        result = check_state_machine_alignment(mod)
        assert isinstance(result, Ok), f"SMA should pass for valid state, got {result}"

    def test_sma_passes_all_valid_states(self):
        for state in ["INIT", "RUN", "PAUSED", "ROLLED_BACK", "TERMINATED"]:
            mod = _MockStatefulModule(state=state)
            result = check_state_machine_alignment(mod)
            assert isinstance(result, Ok), f"SMA should pass for {state}"

    def test_sma_rejects_invalid_state(self):
        mod = _MockStatefulModule(state="BOGUS_STATE")
        result = check_state_machine_alignment(mod)
        assert isinstance(result, Err), f"SMA should fail for invalid state"
        assert isinstance(result.error, SMAError)
        assert result.error.kind == "invalid_state_value"

    def test_sma_warns_when_no_state_attribute(self):
        mod = nn.Linear(4, 4)  # No _state attribute
        result = check_state_machine_alignment(mod)
        # Modules without _state should pass (vacuously) — guard only enforces when present
        assert isinstance(result, Ok), "SMA should pass for modules without _state"
