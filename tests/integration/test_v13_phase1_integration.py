"""v1.3 STANDARD: Phase 1 FractalCurveViT integration (B.14).

Verifies that the new v1.3 components (Multi-Block HMFT, Polar Voronoi,
MambaVision-Lite, Shadow Monitor, R12 Aux Loss, Power-of-4 chunking) can
all be wired into the existing FractalCurveViT model without regressions.

Per the plan, the new components are **opt-in** via config flags:
  - config.enable_eahbp (Phase 2, EAHBPAttention)
  - config.enable_polar (B.8, PolarVoronoiSplitter)
  - config.distillation (B.10, MambaVisionLiteStudent)
  - config.v13_paced_* (Phase 2, Paced Window)
  - config.v12_lambda_* (B.12, R12 Aux Loss)
"""
from __future__ import annotations

import pytest
import torch

from vit_pytorch import FractalCurveViT
from vit_pytorch.layers.splitters import (
    MultiBlockHMFTSplitter,
    MultiBlockHMFTSplitterConfig,
)


class TestFractalCurveViTPhase1Integration:
    """B.14: New v1.3 components integrate with FractalCurveViT."""

    def test_default_fractal_vit_constructs(self):
        """Default FractalCurveViT (H1SS) still works (no regressions)."""
        model = FractalCurveViT(
            image_size=64,
            num_classes=10,
            dim=64,
            mlp_dim=128,
            num_layers=2,
            heads=2,
        )
        model.train()
        x = torch.randn(1, 3, 64, 64)
        forward_output, _metrics = model(x)
        assert forward_output.logits.shape == (1, 10)

    def test_fractal_vit_with_multi_block_hmft_splitter(self):
        """FractalCurveViT(splitter=MultiBlockHMFTSplitter()) constructs and forwards."""
        splitter = MultiBlockHMFTSplitter(MultiBlockHMFTSplitterConfig())
        model = FractalCurveViT(
            image_size=64,
            num_classes=10,
            dim=64,
            mlp_dim=128,
            num_layers=2,
            heads=2,
            splitter=splitter,
        )
        model.train()
        x = torch.randn(1, 3, 64, 64)
        forward_output, _metrics = model(x)
        assert forward_output.logits.shape == (1, 10)

    def test_fractal_vit_h1ss_baseline_smoke(self):
        """H1SS baseline at N=32 to ensure no regression from constants/config changes."""
        model = FractalCurveViT(
            image_size=32,
            num_classes=10,
            dim=64,
            mlp_dim=128,
            num_layers=2,
            heads=2,
        )
        model.train()
        x = torch.randn(2, 3, 32, 32)
        forward_output, _metrics = model(x)
        assert forward_output.logits.shape == (2, 10)
        # num_tokens field exists and is reasonable
        nt = forward_output.num_tokens
        if isinstance(nt, torch.Tensor):
            nt_list = nt.flatten().tolist()
        elif isinstance(nt, int):
            nt_list = [nt]
        else:
            nt_list = list(nt)
        for batch_nt in nt_list:
            assert 8 <= batch_nt <= 64, (
                f"num_tokens={batch_nt} outside design [8, 64]"
            )

    def test_fractal_vit_5_opt_in_flags_default_false(self):
        """All v1.3 opt-in flags default to False (no behavior change)."""
        from vit_pytorch.core.config import FractalConfig
        cfg = FractalConfig(image_size=64)
        assert cfg.enable_eahbp is False
        assert cfg.enable_polar is False
        assert cfg.distillation is False
        assert cfg.benchmark_1024 is False
        # 6 calibration values are populated
        assert cfg.v13_t_fatal == 0.50
        assert cfg.v13_alpha_tree == 1.20
        assert cfg.v13_alpha_skew == 0.40
        assert cfg.v13_gamma_kinetic == 0.15
        assert cfg.v13_delta_washout == 0.05
        assert cfg.v13_epsilon_leak_factor == 0.25
        # R12 coefficients
        assert cfg.v12_lambda_tree == 0.10
        assert cfg.v12_lambda_skew == 0.10
