#!/usr/bin/env python3
"""
I100-2 Temperature Ablation Study Runner (精简版)

优化要点:
- 减少冗余日志输出
- 实时进度显示 (progress bar + ETA)
- 训练日志重定向到文件
- 快速失败检测

使用方法:
    python scripts/run_temp_ablation.py --parallel 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import threading
import shutil

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "temp_ablation"

# Ablation 配置定义 (9组)
ABLATION_CONFIGS = {
    "A": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "基线配置"},
    "B": {"T_start": 1.0, "T_end": 0.3, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_end=0.3"},
    "C": {"T_start": 1.0, "T_end": 0.5, "warmup": 0,  "schedule": "cosine", "learnable": True, "desc": "无 warmup"},
    "D": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "linear", "learnable": True, "desc": "linear"},
    "E": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "exponential", "learnable": True, "desc": "exp"},
    "F": {"T_start": 0.5, "T_end": 0.5, "warmup": 0,  "schedule": "linear", "learnable": False, "desc": "固定T=0.5"},
    "G": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": False, "desc": "非学习温度"},
    "H": {"T_start": 1.5, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_start=1.5"},
    "I": {"T_start": 1.0, "T_end": 0.7, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_end=0.7"},
}

SEEDS = [0, 1, 2]
TOTAL_EXPERIMENTS = len(ABLATION_CONFIGS) * len(SEEDS)


@dataclass
class ExperimentResult:
    config: str
    seed: int
    final_acc: float = 0.0
    best_acc: float = 0.0
    epochs: int = 0
    duration: float = 0.0
    success: bool = False
    error_msg: Optional[str] = None


class ProgressTracker:
    """进度追踪器"""

    def __init__(self, total: int, desc: str = "实验"):
        self.total = total
        self.completed = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.results: List[ExperimentResult] = []
        self.current_task = ""

    def update(self, result: ExperimentResult):
        with self.lock:
            self.completed += 1
            self.results.append(result)

    def print_progress(self, clear: bool = True):
        """打印进度条"""
        elapsed = time.time() - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed) / rate if rate > 0 else 0

        # 进度条
        bar_len = 30
        filled = int(bar_len * self.completed / self.total)
        bar = "█" * filled + "░" * (bar_len - filled)

        # 状态
        status = f"[{bar}] {self.completed}/{self.total}"

        # 当前任务
        if self.current_task:
            status += f" | {self.current_task}"

        # 成功率
        success_count = sum(1 for r in self.results if r.success)
        success_rate = success_count / max(1, self.completed) * 100

        # ETA
        if eta > 0:
            status += f" | ETA: {eta/60:.0f}min | 成功率: {success_rate:.0f}%"

        if clear:
            # 清除上一行
            print(f"\r{' ' * 80}", end="\r")
            print(f"\r{status}", end="\r")
        else:
            print(status)


def get_base_args() -> List[str]:
    """控制变量: 所有实验使用相同的超参数"""
    return [
        "--dataset", "tiny-imagenet",
        "--epochs", "60",
        "--dim", "256",
        "--depth", "8",
        "--heads", "6",
        "--dropout", "0.25",
        "--drop-path", "0.25",
        "--weight-decay", "0.15",
        "--batch-size", "64",
        "--lr", "5e-4",
        "--warmup-epochs", "10",
        "--use-amp",
        "--gradient-checkpoint",
        "--channels-last",
        "--save-interval", "10",
        "--quiet",  # 减少训练日志输出
    ]


def build_args(config_name: str, seed: int) -> List[str]:
    cfg = ABLATION_CONFIGS[config_name]
    args = [
        "--seed", str(seed),
        "--splitter-temp-start", str(cfg["T_start"]),
        "--splitter-temp-end", str(cfg["T_end"]),
        "--splitter-temp-warmup", str(cfg["warmup"]),
        "--temp-schedule", cfg["schedule"],
    ]
    if cfg["learnable"]:
        args.append("--learnable-temperature")
    return get_base_args() + args


def run_experiment(config_name: str, seed: int, log_file: Optional[Path] = None) -> ExperimentResult:
    """运行单个实验"""
    cfg = ABLATION_CONFIGS[config_name]
    output_dir = OUTPUT_DIR / f"config_{config_name}" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 日志文件
    if log_file is None:
        log_file = output_dir / "train.log"

    cmd = [
        "uv", "run", "python", "src/training/train_fractal_vit.py",
        "--save-dir", str(output_dir),
    ] + build_args(config_name, seed)

    start_time = time.time()
    try:
        with open(log_file, "w") as f:
            result = subprocess.run(
                cmd,
                cwd=PROJECT_ROOT,
                stdout=subprocess.DEVNULL,  # 抑制stdout
                stderr=subprocess.STDOUT,    # 重定向到文件
                text=True,
                bufsize=1
            )

        duration = time.time() - start_time

        if result.returncode == 0:
            final_acc, best_acc = parse_training_result(output_dir)
            return ExperimentResult(
                config=config_name, seed=seed,
                final_acc=final_acc, best_acc=best_acc,
                epochs=60, duration=duration, success=True
            )
        else:
            # 读取错误日志
            with open(log_file) as f:
                error = f.read()[-300:] or "Unknown error"
            return ExperimentResult(
                config=config_name, seed=seed,
                duration=duration, success=False, error_msg=error
            )
    except Exception as e:
        duration = time.time() - start_time
        return ExperimentResult(
            config=config_name, seed=seed,
            duration=duration, success=False, error_msg=str(e)
        )


def parse_training_result(output_dir: Path) -> tuple:
    """解析训练结果"""
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            return metrics.get("test_acc", 0.0), metrics.get("best_test_acc", 0.0)
    return 0.0, 0.0


def run_parallel(configs: List[str], seeds: List[int], max_parallel: int,
                 progress: ProgressTracker) -> List[ExperimentResult]:
    """并行执行实验"""
    import concurrent.futures

    tasks = [(cfg, seed) for cfg in configs for seed in seeds]

    def run_task(cfg_seed):
        cfg, seed = cfg_seed
        cfg_desc = ABLATION_CONFIGS[cfg]["desc"]
        progress.current_task = f"{cfg}({cfg_desc}) s{seed}"
        log_file = OUTPUT_DIR / f"config_{cfg}" / f"seed_{seed}" / "train.log"
        return run_experiment(cfg, seed, log_file)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {executor.submit(run_task, t): t for t in tasks}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            progress.update(result)
            results.append(result)
            progress.print_progress()

    return results


def print_summary(results: List[ExperimentResult]):
    """打印结果汇总"""
    print("\n" + "=" * 75)
    print("I100-2 温度消融实验结果")
    print("=" * 75)

    # 按配置分组统计
    config_stats: Dict[str, List[float]] = {}
    for r in results:
        if r.success:
            config_stats.setdefault(r.config, []).append(r.final_acc)

    # 排序输出
    sorted_configs = sorted(
        config_stats.items(),
        key=lambda x: sum(x[1]) / len(x[1]),
        reverse=True
    )

    # 表头
    print(f"\n{'配置':<5} {'T_s':<5} {'T_e':<5} {'W':<3} {'sched':<8} {'L':<3} "
          f"{'Mean':<8} {'Std':<7} {'N':<3} 说明")
    print("-" * 75)

    for config, accs in sorted_configs:
        cfg = ABLATION_CONFIGS[config]
        mean_acc = sum(accs) / len(accs)
        std_acc = (sum((a - mean_acc) ** 2 for a in accs) / len(accs)) ** 0.5 if len(accs) > 1 else 0.0
        print(f"{config:<5} {cfg['T_start']:<5} {cfg['T_end']:<5} {cfg['warmup']:<3} "
              f"{cfg['schedule']:<8} {str(cfg['learnable'])[0]:<3} "
              f"{mean_acc:.4f}   {std_acc:.4f}   {len(accs):<3} {cfg['desc']}")

    # 失败列表
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\n失败 ({len(failures)}个):")
        for f in failures:
            print(f"  {f.config}_s{f.seed}: {f.error_msg[:80] if f.error_msg else 'Unknown'}")

    # 最佳配置
    if sorted_configs:
        best_config, best_accs = sorted_configs[0]
        print(f"\n最佳配置: {best_config} ({ABLATION_CONFIGS[best_config]['desc']}) "
              f"Acc={sum(best_accs)/len(best_accs):.4f}")


def save_results(results: List[ExperimentResult]):
    """保存结果"""
    output_file = OUTPUT_DIR / "ablation_results.json"

    data = {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "success": sum(1 for r in results if r.success),
        "results": [
            {
                "config": r.config,
                "seed": r.seed,
                "final_acc": r.final_acc,
                "best_acc": r.best_acc,
                "duration_s": r.duration,
                "success": r.success,
                "error": r.error_msg,
                "params": ABLATION_CONFIGS[r.config]
            }
            for r in results
        ]
    }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="I100-2 温度消融实验 (精简版)")
    parser.add_argument("--config", type=str, default=None,
                        help="指定配置 (A-I)，默认全部")
    parser.add_argument("--seed", type=int, default=None,
                        help="指定种子，默认全部种子")
    parser.add_argument("--parallel", type=int, default=4,
                        help="并行数 (默认: 4)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="单实验超时(秒)")

    args = parser.parse_args()

    # 确定配置和种子
    configs = [args.config.upper()] if args.config else list(ABLATION_CONFIGS.keys())
    seeds = [args.seed] if args.seed is not None else SEEDS

    total = len(configs) * len(seeds)

    print("=" * 60)
    print("I100-2 温度消融实验")
    print("=" * 60)
    print(f"配置: {configs}")
    print(f"种子: {seeds}")
    print(f"实验数: {total} ({len(configs)}×{len(seeds)})")
    print(f"并行: {args.parallel}")
    print("=" * 60)

    # 进度追踪
    progress = ProgressTracker(total, "消融实验")

    if args.parallel <= 1:
        results = []
        for cfg in configs:
            for seed in seeds:
                progress.current_task = f"{cfg} s{seed}"
                log_file = OUTPUT_DIR / f"config_{cfg}" / f"seed_{seed}" / "train.log"
                result = run_experiment(cfg, seed, log_file)
                progress.update(result)
                progress.print_progress(clear=False)
        print()
    else:
        results = run_parallel(configs, seeds, args.parallel, progress)
        print()  # 换行

    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
