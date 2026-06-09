"""Monitor Module

Numerical monitoring and defense.

=== PR2 (trainer refactor) ===
numerical_defense.py deleted (810 lines). Replaced by:
  - callbacks.NaNGuard (判定 is_healthy, 骨架硬依赖)
  - callbacks.NaNDumpCallback (NaN dump, 默认 off)

Public API (legacy NumericalDefender/GradientValidator/etc) 已废弃:
请从 `src.training.callbacks` 导入新实现。

See plan fluffy-watching-turing.md §3 PR2.
"""

from .gradient_monitor import GradientMonitor
from .loss_monitor import LossMonitor, LossTracker, CombinedLossTracker

__all__ = [
    "GradientMonitor",
    "LossMonitor",
    "LossTracker",
    "CombinedLossTracker",
]
