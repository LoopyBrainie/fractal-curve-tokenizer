"""Epoch Logging Module

Logs epoch statistics to JSON and console.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json
from datetime import datetime
from torch import Tensor


def _tensor_to_serializable(obj: Any) -> Any:
    """Recursively convert Tensor objects to JSON-serializable Python types.

    Args:
        obj: Object to convert

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, Tensor):
        # Tensor -> Python scalar (item() if 0-dim, else list)
        if obj.dim() == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    elif isinstance(obj, dict):
        return {k: _tensor_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_tensor_to_serializable(item) for item in obj]
    elif isinstance(obj, float):
        return obj
    elif isinstance(obj, int):
        return obj
    elif obj is None:
        return None
    else:
        # Try to convert any remaining objects
        try:
            return float(obj)
        except (TypeError, ValueError):
            return str(obj)


class EpochLogger:
    """Logger for epoch statistics

    Creates epoch_stats.json files with comprehensive metrics.

    Example:
        logger = EpochLogger(output_dir="./outputs")

        # After each epoch:
        logger.log(
            epoch=1,
            train_metrics={"loss": 0.5, "accuracy": 0.8},
            eval_metrics={"loss": 0.6, "accuracy": 0.75, "ece": 0.1},
        )
    """

    def __init__(
        self,
        output_dir: str = "./outputs",
        name: str = "training",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.name = name

        # History
        self.history: list = []

    def log(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, Any]] = None,
        eval_metrics: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Log epoch metrics

        Args:
            epoch: Current epoch number
            train_metrics: Training metrics
            eval_metrics: Evaluation metrics
            extra: Additional metrics

        Returns:
            Merged metrics dictionary
        """
        timestamp = datetime.now().isoformat()

        # Build stats dict
        stats = {
            "epoch": epoch,
            "timestamp": timestamp,
        }

        if train_metrics:
            stats["train"] = train_metrics

        if eval_metrics:
            stats["eval"] = eval_metrics

        if extra:
            stats["extra"] = extra

        # Add to history
        self.history.append(stats)

        # Save to JSON
        self._save_epoch_stats(stats)

        # Also save latest
        self._save_latest_stats()

        # Print summary
        self._print_summary(epoch, train_metrics, eval_metrics)

        return stats

    def _save_epoch_stats(self, stats: Dict[str, Any]) -> None:
        """Save epoch stats to JSON"""
        epoch = stats["epoch"]
        stats_path = self.output_dir / f"epoch_{epoch:04d}_stats.json"

        # Convert Tensor objects to JSON-serializable types
        serializable_stats = _tensor_to_serializable(stats)

        with open(stats_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)

    def _save_latest_stats(self) -> None:
        """Save latest stats"""
        latest_path = self.output_dir / "latest_stats.json"

        # Convert Tensor objects to JSON-serializable types
        serializable_stats = _tensor_to_serializable(self.history[-1])

        with open(latest_path, "w") as f:
            json.dump(serializable_stats, f, indent=2)

    def _print_summary(
        self,
        epoch: int,
        train_metrics: Optional[Dict[str, Any]],
        eval_metrics: Optional[Dict[str, Any]],
    ) -> None:
        """Print summary to console"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} Summary")
        print(f"{'='*60}")

        if train_metrics:
            print(f"Train Loss: {train_metrics.get('loss', 0):.4f} | "
                  f"Acc: {train_metrics.get('accuracy', 0)*100:.2f}%")

        if eval_metrics:
            print(f"Val Loss: {eval_metrics.get('loss', 0):.4f} | "
                  f"Acc: {eval_metrics.get('accuracy', 0)*100:.2f}% | "
                  f"Top5: {eval_metrics.get('top5_accuracy', 0)*100:.2f}%")

            if "ece" in eval_metrics:
                print(f"ECE: {eval_metrics['ece']:.4f}")

        print(f"{'='*60}\n")

    def get_history(self) -> list:
        """Get full history"""
        return self.history

    def get_best_epoch(self, metric: str = "accuracy", mode: str = "max") -> Optional[Dict]:
        """Get best epoch by metric

        Args:
            metric: Metric to check (e.g., "accuracy", "loss")
            mode: "max" or "min"

        Returns:
            Best epoch stats dict
        """
        if not self.history:
            return None

        best_epoch = None
        best_value = float("-inf") if mode == "max" else float("inf")

        for stats in self.history:
            # Check both train and eval
            for prefix in ["train", "eval"]:
                if prefix in stats and metric in stats[prefix]:
                    value = stats[prefix][metric]

                    if (mode == "max" and value > best_value) or \
                       (mode == "min" and value < best_value):
                        best_value = value
                        best_epoch = stats
                        break

        return best_epoch


class MetricsTracker:
    """Track metrics over training with summary statistics"""

    def __init__(self):
        self.metrics: Dict[str, list] = {}

    def add(self, name: str, value: float) -> None:
        """Add a metric value"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_history(self, name: str) -> list:
        """Get history of a metric"""
        return self.metrics.get(name, [])

    def get_latest(self, name: str, default: float = 0.0) -> float:
        """Get latest value of a metric"""
        history = self.metrics.get(name, [])
        return history[-1] if history else default

    def get_best(self, name: str, mode: str = "max") -> float:
        """Get best value of a metric"""
        history = self.metrics.get(name, [])
        if not history:
            return 0.0
        return max(history) if mode == "min" else max(history)

    def get_average(self, name: str, window: Optional[int] = None) -> float:
        """Get average of a metric over window"""
        history = self.metrics.get(name, [])
        if not history:
            return 0.0

        if window is not None:
            history = history[-window:]

        return sum(history) / len(history)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics"""
        summary_dict = {}

        for name, values in self.metrics.items():
            if values:
                summary_dict[name] = {
                    "latest": values[-1],
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "count": len(values),
                }

        return summary_dict


__all__ = [
    "EpochLogger",
    "MetricsTracker",
]
