"""v1.3 STANDARD: Trainer CLI flags test (T1.3).

Verifies that train_fractal_vit.build_parser() recognizes the v1.3
opt-in flags. Uses a `build_parser()` helper (extracted from main)
so the test can parse without invoking the full training pipeline.
"""
from __future__ import annotations

import pytest


class TestV13CLIFlags:
    """v1.3 opt-in CLI flags parse correctly."""

    def test_build_parser_exists(self):
        """build_parser is exposed at module level."""
        from src.training.train_fractal_vit import build_parser
        parser = build_parser()
        assert parser is not None

    def test_enable_flags_default_false(self):
        """All enable_* flags default to False when omitted."""
        from src.training.train_fractal_vit import build_parser
        args = build_parser().parse_args([])
        assert args.enable_shadow_monitor is False
        assert args.enable_r12_aux is False
        assert args.enable_eahbp_3gate is False
        assert args.enable_paced_window is False

    def test_enable_flags_parse_true(self):
        """All enable_* flags parse to True when present."""
        from src.training.train_fractal_vit import build_parser
        args = build_parser().parse_args([
            "--enable-shadow-monitor",
            "--enable-r12-aux",
            "--enable-eahbp-3gate",
            "--enable-paced-window",
        ])
        assert args.enable_shadow_monitor is True
        assert args.enable_r12_aux is True
        assert args.enable_eahbp_3gate is True
        assert args.enable_paced_window is True

    def test_calibration_args_parse(self):
        """Calibration args parse to their values."""
        from src.training.train_fractal_vit import build_parser
        args = build_parser().parse_args([
            "--shadow-monitor-interval", "25",
            "--r12-lambda-tree", "0.20",
            "--r12-lambda-skew", "0.15",
            "--paced-window-fatal-streak", "5",
        ])
        assert args.shadow_monitor_interval == 25
        assert args.r12_lambda_tree == pytest.approx(0.20, abs=1e-9)
        assert args.r12_lambda_skew == pytest.approx(0.15, abs=1e-9)
        assert args.paced_window_fatal_streak == 5

    def test_calibration_args_defaults(self):
        """Calibration args have sensible defaults."""
        from src.training.train_fractal_vit import build_parser
        args = build_parser().parse_args([])
        assert args.shadow_monitor_interval == 50
        assert args.r12_lambda_tree == pytest.approx(0.10, abs=1e-9)
        assert args.r12_lambda_skew == pytest.approx(0.10, abs=1e-9)
        assert args.paced_window_fatal_streak == 3
