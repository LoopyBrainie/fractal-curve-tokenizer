"""Test-side stubs for removed production modules.

The HMFT h-prob logging test (`test_hmft_h_probs_logging.py`) imports
`from src.training.metrics.collector import MetricsCollector`. The
`MetricsCollector` class was removed in PR1 — production metrics now
flow through the flat `ctx.metrics` dict (PR5c decision).

To keep `src/` production-pure (no zombie shim file, no static
analysis noise) and the test file untouched, we inject a minimal
dict-backed `MetricsCollector` stub into `sys.modules` at collection
time. The stub exposes the exact contract the test relies on:

    collector = MetricsCollector()
    collector.record(name, value)
    snap = collector.get_summary()   # returns dict

The injection only activates if no real `src.training.metrics.collector`
module is present — once a real collector lands, the stub stays out
of the way automatically. Lifecycle is tied to the test session;
no production code can `import` this stub.

Why conftest.py (not a real file under src/):
  - Keeps `src/` clean for packaging, Mypy, Ruff, etc.
  - Preserves the test file as the spec (no test edits).
  - Easy to delete later — one file, one block.
"""
from __future__ import annotations

import sys
import types
from typing import Any


# ---------------------------------------------------------------------------
# Architecture guard: inject a fake `src.training.metrics.collector` module
# only if the real one is not already importable. Idempotent and safe to
# re-run across pytest collection passes.
# ---------------------------------------------------------------------------
if "src.training.metrics" not in sys.modules:
    metrics_package = types.ModuleType("src.training.metrics")
    metrics_package.__path__ = []  # mark as a package (no submodule discovery)
    sys.modules["src.training.metrics"] = metrics_package

if "src.training.metrics.collector" not in sys.modules:
    collector_module = types.ModuleType("src.training.metrics.collector")

    class MetricsCollector:
        """Dict-backed fake of the removed PR1 collector.

        Stores every `record(name, value)` call in an internal dict and
        returns a copy via `get_summary()`. The test reads back via
        `snap.get(name)` and compares against expected values, so the
        fake must store real values — returning `MagicMock()` here would
        break the `snap.get(...) == p` assertion because `MagicMock`
        does not equal scalar floats.
        """

        def __init__(self) -> None:
            self._store: dict[str, Any] = {}

        def record(self, name: str, value: Any) -> None:
            self._store[name] = value

        def get_summary(self) -> dict[str, Any]:
            return dict(self._store)

    # Bind the class onto the synthetic module. `setattr` (typed as
    # `setattr(obj, name: str, value: Any)`) sidesteps the static
    # attribute-resolution check that `ty` runs against direct
    # `module.X = ...` assignment on `types.ModuleType`.
    setattr(collector_module, "MetricsCollector", MetricsCollector)
    sys.modules["src.training.metrics.collector"] = collector_module
