"""Monitor Module

Numerical monitoring and defense.

=== PR2 (trainer refactor) ===
numerical_defense.py deleted (810 lines). Replaced by:
  - callbacks.NaNGuard (判定 is_healthy, 骨架硬依赖)
  - callbacks.NaNDumpCallback (NaN dump, 默认 off)

=== PR3 (trainer refactor) ===
gradient_monitor.py deleted (501 lines, 含 dead EnhancedGradientMonitor +
CF-3 register_full_backward_hook). Replaced by:
  - callbacks.GradientMonitorCallback (单一来源 register_post_accumulate_grad_hook)

Public API (legacy GradientMonitor) 已废弃:
请从 `src.training.callbacks` 导入新实现。

See plan fluffy-watching-turing.md §3 PR2 + PR3.
"""

from .loss_monitor import LossMonitor, LossTracker, CombinedLossTracker

__all__ = [
    "LossMonitor",
    "LossTracker",
    "CombinedLossTracker",
]
