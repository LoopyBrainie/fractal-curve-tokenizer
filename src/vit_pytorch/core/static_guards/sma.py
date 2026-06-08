"""v1.3 STANDARD: SMA (State Machine Alignment) guard.

Per docs/superpowers/specs/2026-06-08-fractal-hilbert-vit-best-practice-design.md §9.6 Set A.

Ensures that any module with a `_state` attribute has a state value in the
allowed set. Passes vacuously for modules without `_state`.
"""
from __future__ import annotations
from typing import Any, Literal, Union
from vit_pytorch.core.outcome import Ok, Err, Outcome, ConfigError


SMAStateValue = Literal["INIT", "RUN", "PAUSED", "ROLLED_BACK", "TERMINATED"]
_SMA_VALID_STATES = frozenset({"INIT", "RUN", "PAUSED", "ROLLED_BACK", "TERMINATED"})


class SMAError(ConfigError):
    """SMA guard error. kind in {"missing_state", "invalid_state_value"}."""
    kind: str  # "missing_state" | "invalid_state_value"


def check_state_machine_alignment(obj: Any) -> Outcome[None, SMAError]:
    """Verify obj._state (if present) is in the allowed set.

    Args:
        obj: Any object; typically a nn.Module with a _state attribute.

    Returns:
        Ok(None) if no _state attribute or _state is valid.
        Err(SMAError) if _state has an invalid value.
    """
    state = getattr(obj, "_state", None)
    if state is None:
        return Ok(None)  # Vacuously pass
    if not isinstance(state, str):
        return Err(SMAError(
            kind="invalid_state_value",
            reason=f"_state must be str, got {type(state).__name__}",
        ))
    if state not in _SMA_VALID_STATES:
        return Err(SMAError(
            kind="invalid_state_value",
            reason=f"_state={state!r} not in {sorted(_SMA_VALID_STATES)}",
        ))
    return Ok(None)
