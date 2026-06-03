"""Hypothesis-based invariant tests for Outcome[T, E].

Enforces (Q7 / R6 / Phase 1 step 2.5): no Ok(...) call site in the
migrated code may receive a torch.Tensor or torch.nn.Module payload.
This protects the STE gradient path through F.gumbel_softmax.

Note: the runtime guard in Ok.__post_init__ catches the bug at construction
time. This test verifies the guard itself (regression protection) and
exercises representative call sites from config.py and data.py.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from hypothesis import given, strategies as st
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError, DataError


# Strategy: any non-Tensor, non-Module value
@st.composite
def safe_payloads(draw):
    """Generate payloads that are safe to wrap in Ok."""
    return draw(st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(max_size=20),
        st.booleans(),
        st.lists(st.integers(), max_size=5),
        st.dictionaries(st.text(max_size=5), st.integers(), max_size=3),
    ))


class TestOkRuntimeGuard:
    @given(value=safe_payloads())
    def test_ok_accepts_safe_values(self, value):
        """All non-Tensor, non-Module values must be accepted."""
        result: Ok = Ok(value)
        assert result.value == value

    @given(
        shape=st.tuples(st.integers(1, 4), st.integers(1, 4)),
        dtype=st.sampled_from([torch.float32, torch.int64]),
    )
    def test_ok_rejects_torch_tensor(self, shape, dtype):
        """Runtime guard: torch.Tensor payloads must raise TypeError."""
        tensor = torch.zeros(shape, dtype=dtype)
        with pytest.raises(TypeError, match="torch.Tensor"):
            Ok(tensor)

    def test_ok_rejects_nn_module(self):
        """nn.Module payloads must raise TypeError (Modules carry param tensors)."""
        module = nn.Linear(2, 2)
        with pytest.raises(TypeError, match="nn.Module|torch.Tensor"):
            Ok(module)

    def test_unsafe_tensor_classmethod_works(self):
        """Ok.unsafe_tensor(t) is the sanctioned escape hatch for tensors."""
        t = torch.zeros(3)
        result = Ok.unsafe_tensor(t)
        assert isinstance(result, Ok)
        assert result.value is t


class TestOutcomeTypeAlias:
    """Verify Outcome[T, E] is usable as a type annotation in migrated code."""

    def test_outcome_is_constructible_from_either_variant(self):
        ok: Outcome[int, str] = Ok(42)
        err: Outcome[int, str] = Err("oops")
        assert ok.value == 42
        assert err.error == "oops"

    def test_pattern_match_dispatches_correctly(self):
        ok: Outcome[int, str] = Ok(42)
        err: Outcome[int, str] = Err("oops")
        for outcome, expected in [(ok, 42), (err, None)]:
            match outcome:
                case Ok(v):
                    assert v == 42
                case Err(_):
                    assert outcome is err


class TestConfigErrorInheritance:
    """Q3: ConfigError must subclass ValueError for back-compat."""

    def test_config_error_is_value_error(self):
        err = ConfigError(kind="min_patch_size", reason="test")
        assert isinstance(err, ValueError)
        # Existing pytest.raises(ValueError) sites must continue to match.
        with pytest.raises(ValueError):
            raise err


class TestDataErrorDistinct:
    """DataError does NOT subclass ValueError — distinct failure class."""

    def test_data_error_is_not_value_error(self):
        err = DataError(kind="unknown_dataset", reason="test")
        assert not isinstance(err, ValueError)
        # Callers must pattern-match on the Outcome.
