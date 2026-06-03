"""Tests for the Outcome[T, E] sealed sum primitive."""
from __future__ import annotations
import pytest
import torch
from vit_pytorch.core.outcome import Outcome, Ok, Err, ConfigError


class TestOk:
    def test_ok_holds_value(self):
        ok: Ok[int] = Ok(42)
        assert ok.value == 42

    def test_ok_rejects_torch_tensor(self):
        """Runtime guard: STE gradient safety (Decision Q7)."""
        tensor = torch.zeros(3)
        with pytest.raises(TypeError, match="torch.Tensor"):
            Ok(tensor)

    def test_ok_rejects_torch_nn_module(self):
        """Modules carry parameters (tensors); reject them too."""
        from torch import nn
        m = nn.Linear(2, 2)
        with pytest.raises(TypeError, match="torch.Tensor|nn.Module"):
            Ok(m)


class TestErr:
    def test_err_holds_error(self):
        err: Err[str] = Err("oops")
        assert err.error == "oops"


class TestPatternMatch:
    def test_match_on_ok(self):
        ok: Outcome[int, str] = Ok(42)
        match ok:
            case Ok(value):
                assert value == 42
            case Err(error):
                pytest.fail(f"expected Ok, got Err({error})")

    def test_match_on_err(self):
        err: Outcome[int, str] = Err("oops")
        match err:
            case Ok(value):
                pytest.fail(f"expected Err, got Ok({value})")
            case Err(error):
                assert error == "oops"


class TestTypeAlias:
    def test_outcome_is_a_type_alias(self):
        """PEP 695: Outcome[T, E] = Ok[T] | Err[E] must be usable as a type."""
        def consumer(o: Outcome[int, str]) -> int:
            match o:
                case Ok(v): return v
                case Err(_): return -1
        assert consumer(Ok(42)) == 42
        assert consumer(Err("x")) == -1


class TestConfigError:
    def test_config_error_is_value_error(self):
        """Q3: ConfigError must subclass ValueError so all 14 existing
        pytest.raises(ValueError) sites work unchanged."""
        err = ConfigError(kind="min_patch_size", reason="must be positive")
        assert isinstance(err, ValueError)

    def test_config_error_carries_kind_and_reason(self):
        err = ConfigError(kind="coverage_base", reason="must be in (0, 1]")
        assert err.kind == "coverage_base"
        assert err.reason == "must be in (0, 1]"

    def test_config_error_str_includes_field_name(self):
        """Q3: Chinese/English field names must survive in the message
        so existing test_config.py match= patterns continue to match."""
        err = ConfigError(kind="min_patch_size", reason="must be positive")
        assert "min_patch_size" in str(err)
