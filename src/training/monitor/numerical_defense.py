"""Numerical Defense Module

Provides numerical stability protection:
- Gradient NaN/Inf detection
- Anomaly detection context
- Automatic gradient skipping on numerical issues
"""

from __future__ import annotations

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from contextlib import contextmanager


class AnomalyDetectionContext:
    """Context manager for PyTorch anomaly detection

    Enables torch.autograd.set_detect_anomaly() within a scope.
    Useful for debugging gradient issues.

    Example:
        with AnomalyDetectionContext(model):
            loss = model(inputs)
            loss.backward()
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._previous_state = None

    def __enter__(self):
        if self.enabled:
            self._previous_state = torch.is_anomaly_enabled()
            torch.set_detect_anomaly(True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled and self._previous_state is not None:
            torch.set_detect_anomaly(self._previous_state)
        return False


class GradientValidator:
    """Validate gradients for numerical issues

    Checks for NaN, Inf, and extreme values in gradients.
    Can automatically skip optimizer steps on detected issues.

    Example:
        validator = GradientValidator(model, skip_on_issue=True)

        # After backward:
        should_skip = validator.check_gradients()
        if not should_skip:
            optimizer.step()
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        skip_on_issue: bool = True,
        log_warnings: bool = True,
    ):
        self.model = model
        self.skip_on_issue = skip_on_issue
        self.log_warnings = log_warnings

        self.issue_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.issues_history: List[Dict[str, Any]] = []

    def check_gradients(self) -> bool:
        """Check all model gradients for issues

        Returns:
            True if gradients are valid (can proceed with optimizer step)
            False if gradients have issues (should skip)
        """
        if self.model is None:
            return True

        has_nan = False
        has_inf = False

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad

                # Check NaN
                if torch.isnan(grad).any():
                    has_nan = True
                    self.nan_count += 1
                    self._record_issue(name, "nan", grad.norm().item())

                # Check Inf
                if torch.isinf(grad).any():
                    has_inf = True
                    self.inf_count += 1
                    self._record_issue(name, "inf", grad.norm().item())

        has_issue = has_nan or has_inf
        if has_issue:
            self.issue_count += 1

            if self.log_warnings:
                self._log_warning(has_nan, has_inf)

        # Return True if valid (proceed), False if skip
        return not (self.skip_on_issue and has_issue)

    def _record_issue(self, name: str, issue_type: str, norm: float):
        """Record issue for history"""
        self.issues_history.append({
            "name": name,
            "type": issue_type,
            "norm": norm,
        })

    def _log_warning(self, has_nan: bool, has_inf: bool):
        """Log warning message"""
        issues = []
        if has_nan:
            issues.append("NaN")
        if has_inf:
            issues.append("Inf")

        print(f"[WARNING] Gradient issue detected: {', '.join(issues)} "
              f"(total issues: {self.issue_count})")

    def get_gradient_stats(self) -> Dict[str, Any]:
        """Get gradient statistics

        Returns:
            Dictionary of statistics
        """
        stats = {
            "total_issues": self.issue_count,
            "nan_count": self.nan_count,
            "inf_count": self.inf_count,
            "has_issues": self.issue_count > 0,
        }

        # Current gradient norms
        if self.model is not None:
            norms = []
            for param in self.model.parameters():
                if param.grad is not None:
                    norms.append(param.grad.norm().item())

            if norms:
                stats["current_grad_norm"] = norms[0] if len(norms) == 1 else norms
                stats["max_grad_norm"] = max(norms)
                stats["min_grad_norm"] = min(norms)

        return stats

    def has_recent_issues(self, window: int = 10) -> bool:
        """Check if there were issues in recent steps

        Args:
            window: Number of recent steps to check

        Returns:
            True if issues found in recent history
        """
        recent = self.issues_history[-window:]
        return len(recent) > 0

    def reset(self) -> None:
        """Reset counters and history"""
        self.issue_count = 0
        self.nan_count = 0
        self.inf_count = 0
        self.issues_history.clear()


class NumericalDefender:
    """Complete numerical defense system

    Combines anomaly detection and gradient validation.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        detect_anomaly: bool = False,
        skip_on_nan: bool = True,
        check_frequency: int = 1,
    ):
        self.model = model
        self.detect_anomaly = detect_anomaly
        self.skip_on_nan = skip_on_nan
        self.check_frequency = check_frequency

        self.validator = GradientValidator(model, skip_on_issue=skip_on_nan)
        self.anomaly_context = AnomalyDetectionContext(enabled=detect_anomaly)

        self.step_count = 0

    def should_check(self) -> bool:
        """Check if we should validate gradients this step"""
        return self.step_count % self.check_frequency == 0

    def pre_backward(self):
        """Hook to call before backward"""
        pass  # Could enable anomaly detection here if needed

    def post_backward(self) -> bool:
        """Hook to call after backward, returns whether to skip optimizer step

        Returns:
            True if should proceed with optimizer step
            False if should skip
        """
        self.step_count += 1

        if self.should_check():
            return self.validator.check_gradients()
        return True

    def pre_step(self):
        """Hook to call before optimizer step"""
        if self.detect_anomaly:
            self.anomaly_context.__enter__()

    def post_step(self):
        """Hook to call after optimizer step"""
        if self.detect_anomaly:
            self.anomaly_context.__exit__(None, None, None)

    def get_stats(self) -> Dict[str, Any]:
        """Get defense statistics"""
        return {
            "validator": self.validator.get_gradient_stats(),
            "step_count": self.step_count,
            "skip_rate": self.validator.issue_count / max(self.step_count, 1),
        }

    def reset(self) -> None:
        """Reset defender state"""
        self.validator.reset()
        self.step_count = 0


def check_tensor_numerical_health(
    tensor: torch.Tensor,
    name: str = "tensor",
    raise_on_issue: bool = False,
) -> Dict[str, Any]:
    """Check a tensor for numerical health

    Args:
        tensor: Tensor to check
        name: Name for logging
        raise_on_issue: Whether to raise exception on issues

    Returns:
        Dictionary of health metrics
    """
    result = {
        "name": name,
        "has_nan": torch.isnan(tensor).any().item(),
        "has_inf": torch.isinf(tensor).any().item(),
        "min": tensor.min().item() if tensor.numel() > 0 else None,
        "max": tensor.max().item() if tensor.numel() > 0 else None,
        "mean": tensor.mean().item() if tensor.numel() > 0 else None,
        "std": tensor.std().item() if tensor.numel() > 0 else None,
    }

    has_issues = result["has_nan"] or result["has_inf"]

    if has_issues and raise_on_issue:
        raise ValueError(f"Numerical issue in {name}: NaN={result['has_nan']}, Inf={result['has_inf']}")

    return result


__all__ = [
    "AnomalyDetectionContext",
    "GradientValidator",
    "NumericalDefender",
    "check_tensor_numerical_health",
]
