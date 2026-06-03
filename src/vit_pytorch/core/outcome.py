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
