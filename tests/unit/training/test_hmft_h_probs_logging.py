"""v1.3 STANDARD: HMFT h_probs logging test (T6.1).

Verifies that the trainer can pull the 5-bin block-size distribution
from a MultiBlockHMFTSplitter's get_diagnostics() and log it to the
MetricsCollector for dashboard consumption.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock


class TestHMFTHProbsLogging:
    """T6.1: h_probs are logged per epoch via get_diagnostics()."""

    def test_log_h_probs_helper_exists(self):
        """log_h_probs is importable from epoch_train."""
        from src.training.trainer.epoch_train import log_h_probs
        assert log_h_probs is not None

    def test_log_h_probs_records_5_bins(self):
        """5 h-prob values from a mock splitter land in the collector."""
        from src.training.trainer.epoch_train import log_h_probs
        from src.training.metrics.collector import MetricsCollector

        # Mock splitter with realistic h_probs
        splitter = MagicMock()
        splitter.get_diagnostics.return_value = {
            "h_probs": [0.1, 0.2, 0.3, 0.2, 0.2],
        }
        collector = MetricsCollector()
        log_h_probs(splitter, collector, epoch=1)
        snap = collector.get_summary()
        for i, p in enumerate([0.1, 0.2, 0.3, 0.2, 0.2]):
            assert snap.get(f"hmft/h_prob_bin_{i}") == p, (
                f"Expected h_prob_bin_{i}={p}, got {snap.get(f'hmft/h_prob_bin_{i}')}"
            )

    def test_log_h_probs_handles_missing_h_probs(self):
        """Splitter without h_probs is a no-op (no exception)."""
        from src.training.trainer.epoch_train import log_h_probs
        from src.training.metrics.collector import MetricsCollector

        splitter = MagicMock()
        splitter.get_diagnostics.return_value = {}  # no h_probs
        collector = MetricsCollector()
        log_h_probs(splitter, collector, epoch=2)  # must not raise
        snap = collector.get_summary()
        assert "hmft/h_prob_bin_0" not in snap

    def test_log_h_probs_handles_no_diagnostics_method(self):
        """Splitter without get_diagnostics() is a no-op."""
        from src.training.trainer.epoch_train import log_h_probs
        from src.training.metrics.collector import MetricsCollector

        class BareSplitter:
            pass
        collector = MetricsCollector()
        log_h_probs(BareSplitter(), collector, epoch=3)  # must not raise
        snap = collector.get_summary()
        assert "hmft/h_prob_bin_0" not in snap
