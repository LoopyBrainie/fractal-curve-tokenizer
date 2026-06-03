"""Outcome[T, E] sealed sum type for algebraic error handling.

Phase 1 boundary: introduced only at the config + data + CLI layer.
DO NOT import this module from src/vit_pytorch/layers/** or
src/vit_pytorch/modules/** — the L2/L3 forward graph must stay
unwrapped to protect torch.compile cache and STE gradient flow.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar

import torch

T = TypeVar("T")
E = TypeVar("E")


def _check_no_tensor_in_ok(value: Any) -> None:
    """Runtime guard: prevents wrapping torch.Tensor in Ok.

    Rationale (Q7 / R6): the STE gradient path through F.gumbel_softmax
    requires the success-path tensor in the clear. Wrapping a tensor
    in Ok(...) and consuming via match risks silent detach. This guard
    makes the invariant load-bearing in production code, not just a
    test-time check.
    """
    if isinstance(value, torch.Tensor):
        raise TypeError(
            f"Ok(...) refused torch.Tensor payload (shape={tuple(value.shape)}, "
            f"dtype={value.dtype}). Unwrap the tensor before constructing Ok. "
            f"This guard protects the STE gradient path; see "
            f"src/vit_pytorch/core/outcome.py docstring."
        )
    # torch.nn.Module carries parameters (tensors) — also reject.
    import torch.nn as nn
    if isinstance(value, nn.Module):
        raise TypeError(
            f"Ok(...) refused torch.nn.Module payload ({type(value).__name__}). "
            f"Modules carry parameter tensors; unwrap to plain values first."
        )


@dataclass(slots=True, frozen=True)
class Ok(Generic[T]):
    """Success variant of Outcome[T, E]."""
    value: T

    def __post_init__(self) -> None:
        _check_no_tensor_in_ok(self.value)

    @classmethod
    def unsafe_tensor(cls, tensor: torch.Tensor) -> "Ok[torch.Tensor]":
        """Explicit escape hatch: bypass the torch.Tensor guard.

        The default Ok.__post_init__ guard refuses torch.Tensor payloads
        to protect the STE gradient path (see Q7 / R6). This classmethod
        is the *only* sanctioned way to wrap a tensor in Ok. Use it
        ONLY when the caller can prove the tensor is detached from the
        computation graph (e.g., eval-time logging, post-`detach()`
        metrics, top-K confidence arrays).

        The escape hatch is loud on purpose: every call site becomes a
        docstring anchor for an audit. By construction, a tensor
        entering Ok via `Ok.unsafe_tensor(...)` cannot have entered
        `Ok(...)` (which would have raised) — so the call site IS the
        audit trail.

        Returns:
            Ok[torch.Tensor] wrapping the supplied tensor. The tensor
            is stored verbatim; no copy or detach is performed.
        """
        # Bypass dataclass __init__ + __post_init__ validation by
        # constructing an uninitialized instance and using object.__setattr__
        # (required because the dataclass is frozen=True).
        obj = cls.__new__(cls)
        super(Ok, obj).__init__()  # type: ignore[misc]
        object.__setattr__(obj, "value", tensor)
        return obj


@dataclass(slots=True, frozen=True)
class Err(Generic[E]):
    """Failure variant of Outcome[T, E]."""
    error: E


# PEP 695 type alias (Python ≥ 3.12)
type Outcome[T, E] = Ok[T] | Err[E]


ConfigErrorKind = Literal[
    "min_patch_size",
    "max_level_limit",
    "coverage_base",
    "coverage_order",
    "temperature_min",
    "temperature_init",
    "temperature_anneal",
    "entropy_mode",
    "feature_dim",
    "hidden_dim",
    "diversity_weight",
    "reconstruction_weight",
    "split_threshold",
    "gumbel_temp_order",
    "image_size",
    "divisibility",
    "grid_size",
]


class ConfigError(ValueError):
    """Typed error for config validation failures (Q3: hybrid).

    Subclasses ValueError so all 14 in-scope pytest.raises(ValueError)
    sites continue to work. The `kind` Literal enables type-level
    discrimination for new code; `reason` is the human message.

    Note: deliberately does NOT declare __slots__. The Python built-in
    BaseException / Exception hierarchy already provides __dict__ in
    the C-level instance layout, so a __slots__ subclass cannot
    actually save memory; meanwhile, __slots__ on Exception subclasses
    can cause conflicts with pickle, copy.deepcopy, and certain
    multiple-inheritance patterns. We rely on plain attribute
    assignment (kind, reason) for portability.
    """

    def __init__(self, kind: ConfigErrorKind, reason: str) -> None:
        self.kind = kind
        self.reason = reason
        super().__init__(f"{kind}: {reason}")


DataErrorKind = Literal[
    "unknown_dataset",
    "hf_not_installed",
    "load_failure",
]


class DataError(Exception):
    """Typed error for dataset loading failures (Q5: minimal scope).

    Deliberately does NOT subclass ValueError or RuntimeError — data-loading
    failures are a distinct failure class from config validation.
    Callers pattern-match on `error.kind` via the Outcome.

    Like ConfigError, deliberately does NOT declare __slots__ — see
    ConfigError docstring for the rationale (BaseException already
    provides __dict__; __slots__ on Exception subclasses can break
    pickle/deepcopy/multi-inheritance).
    """

    def __init__(self, kind: DataErrorKind, reason: str) -> None:
        self.kind = kind
        self.reason = reason
        super().__init__(f"{kind}: {reason}")
