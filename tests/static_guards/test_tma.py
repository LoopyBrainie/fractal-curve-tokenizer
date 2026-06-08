"""v1.3 STANDARD: TMA (Telemetry Monitor Alignment) guard test.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set B.
"""
import pytest
import torch
import torch.nn as nn
from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards.tma import check_telemetry_alignment, TMAError


class _MockTelemetryModule(nn.Module):
    """Mock module with a telemetry interface (xxx_output property)."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self._telemetry_cache = {"wba": 0.85, "gme": 0.65, "mig": 0.05}

    @property
    def monitor_output(self) -> dict:
        return dict(self._telemetry_cache)


@pytest.mark.static_guard
class TestTMAGuard:
    def test_tma_passes_for_module_with_aligned_telemetry(self):
        mod = _MockTelemetryModule()
        result = check_telemetry_alignment(mod)
        assert isinstance(result, Ok), f"TMA should pass, got {result}"

    def test_tma_passes_for_module_without_telemetry(self):
        mod = nn.Linear(4, 4)
        result = check_telemetry_alignment(mod)
        # No telemetry → pass vacuously
        assert isinstance(result, Ok)

    def test_tma_rejects_empty_telemetry_dict(self):
        class _EmptyTelemetry(nn.Module):
            monitor_output = {}

        mod = _EmptyTelemetry()
        result = check_telemetry_alignment(mod)
        # Empty dict is informational — pass
        assert isinstance(result, Ok)

    def test_tma_handles_non_dict_telemetry(self):
        class _BadTelemetry(nn.Module):
            @property
            def monitor_output(self):
                return "not a dict"  # Invalid type

        mod = _BadTelemetry()
        result = check_telemetry_alignment(mod)
        assert isinstance(result, Err)
        assert result.error.kind == "non_dict_telemetry"
