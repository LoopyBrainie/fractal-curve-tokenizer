"""Tests for try_create_dataset() and try_create_hf_dataset() factories (AEH Phase 1).

These factories wrap create_dataset() / create_hf_dataset() to return
Outcome[Dataset, DataError] for known user-input failures, while
propagating programmer errors (FileNotFoundError, network issues, etc.)
unchanged. The two patterns we exercise:

1. try_create_dataset: catches the "Unknown dataset" ValueError → DataError.
2. try_create_hf_dataset: catches the "HF not installed" ImportError
   AND the re-raised ValueError from a missing split.
"""
from __future__ import annotations
import pytest

from vit_pytorch.core.outcome import Err, Ok, DataError


class TestTryCreateDataset:
    """try_create_dataset() returns Outcome[Dataset, DataError]."""

    def test_err_for_unknown_dataset_name(self):
        from training.data import try_create_dataset
        # "unknown_dataset" is not in the supported list (cifar10, cifar100,
        # tiny-imagenet, cub200, mnist). The factory must surface the
        # ValueError as a typed DataError, not propagate the raw raise.
        result = try_create_dataset(name="not_a_real_dataset", split="train")
        assert isinstance(result, Err)
        assert result.error.kind == "unknown_dataset"
        assert "Unknown dataset" in str(result.error)

    def test_factory_returns_outcome_not_raw_raise(self):
        """The factory must NEVER let the underlying ValueError escape.

        The whole point of the Outcome wrapper is to give callers a
        non-raising alternative; if the ValueError leaks, the boundary
        contract is broken.
        """
        from training.data import try_create_dataset
        # Should NOT raise ValueError; should return Err instead.
        try:
            result = try_create_dataset(name="garbage", split="train")
        except ValueError as e:
            pytest.fail(
                f"try_create_dataset leaked ValueError: {e}. "
                f"Expected Outcome[Dataset, DataError]."
            )
        assert isinstance(result, Err)


class TestTryCreateHFDataset:
    """try_create_hf_dataset() returns Outcome[Dataset, DataError]."""

    def test_err_when_hf_not_installed(self):
        """When datasets is not installed, the factory must return Err
        of kind 'hf_not_installed' — never propagate ImportError.

        We simulate the not-installed state by toggling the module-level
        HF_AVAILABLE flag, which is what the import-time try/except sets.
        """
        from training import data as data_mod
        original = data_mod.HF_AVAILABLE
        data_mod.HF_AVAILABLE = False
        try:
            from training.data import try_create_hf_dataset
            result = try_create_hf_dataset(name="any/dataset", split="train")
            assert isinstance(result, Err)
            assert result.error.kind == "hf_not_installed"
            assert isinstance(result.error, DataError)
        finally:
            data_mod.HF_AVAILABLE = original

    def test_does_not_leak_importerror_when_hf_unavailable(self):
        """The boundary contract: ImportError must NOT escape the factory."""
        from training import data as data_mod
        data_mod.HF_AVAILABLE = False
        try:
            from training.data import try_create_hf_dataset
            try:
                result = try_create_hf_dataset(name="any/dataset", split="train")
            except ImportError as e:
                pytest.fail(
                    f"try_create_hf_dataset leaked ImportError: {e}. "
                    f"Expected Outcome[Dataset, DataError]."
                )
            assert isinstance(result, Err)
        finally:
            data_mod.HF_AVAILABLE = True  # restore (the original was True)


class TestDataErrorDistinctFromConfigError:
    """DataError must be a distinct failure class, not a config error."""

    def test_data_error_is_not_value_error(self):
        """Unlike ConfigError, DataError does NOT subclass ValueError.

        Rationale: dataset loading failures are not user-input errors
        (the dataset name was correct, but loading failed for some
        external reason). Callers must pattern-match on the Outcome
        rather than catching ValueError.
        """
        err = DataError(kind="unknown_dataset", reason="test")
        assert not isinstance(err, ValueError)

    def test_data_error_carries_kind_and_reason(self):
        err = DataError(kind="load_failure", reason="network timeout")
        assert err.kind == "load_failure"
        assert err.reason == "network timeout"

    def test_data_error_str_includes_kind(self):
        err = DataError(kind="hf_not_installed", reason="pip install datasets")
        assert "hf_not_installed" in str(err)
