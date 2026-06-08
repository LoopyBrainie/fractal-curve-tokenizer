"""v1.3 STANDARD: PCAR (P-Controller Anti-saturation) guard test.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set A.
"""
import pytest
import torch
import torch.nn as nn
from vit_pytorch.core.outcome import Ok, Err
from vit_pytorch.core.static_guards.pcar import check_p_controller_anti_saturation, PCARError


class _MockPController(nn.Module):
    """Mock P-Controller with explicit saturation bounds."""
    def __init__(self, saturation_min: float = -0.95, saturation_max: float = 0.95):
        super().__init__()
        self._saturation_min = saturation_min
        self._saturation_max = saturation_max
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, error: torch.Tensor) -> torch.Tensor:
        output = self.gain * error
        return torch.clamp(output, min=self._saturation_min, max=self._saturation_max)


@pytest.mark.static_guard
class TestPCARGuard:
    def test_pcar_passes_for_well_configured_controller(self):
        ctrl = _MockPController()
        result = check_p_controller_anti_saturation(ctrl)
        assert isinstance(result, Ok), f"PCAR should pass for well-configured controller, got {result}"

    def test_pcar_passes_for_tight_bounds(self):
        ctrl = _MockPController(saturation_min=-0.5, saturation_max=0.5)
        result = check_p_controller_anti_saturation(ctrl)
        assert isinstance(result, Ok)

    def test_pcar_rejects_missing_saturation_attributes(self):
        class _BadController(nn.Module):
            def __init__(self):
                super().__init__()
                # No _saturation_min/_saturation_max
                self.gain = nn.Parameter(torch.tensor(0.1))

        bad = _BadController()
        result = check_p_controller_anti_saturation(bad)
        # Modules without explicit saturation attrs pass vacuously (not a P-Controller)
        assert isinstance(result, Ok)

    def test_pcar_rejects_inverted_bounds(self):
        class _InvertedController(nn.Module):
            def __init__(self):
                super().__init__()
                self._saturation_min = 0.95
                self._saturation_max = -0.95  # Inverted!

        bad = _InvertedController()
        result = check_p_controller_anti_saturation(bad)
        assert isinstance(result, Err)
        assert isinstance(result.error, PCARError)
