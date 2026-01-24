#!/usr/bin/env python3
"""
I100-2 Temperature Ablation Study Runner (Python版本)

使用Python脚本直接调用训练器进行消融实验，epoch=60加快速度。
通过subprocess调用 `uv run python` 确保依赖环境正确。

配置说明:
- 9组配置 (A-I)，每组3个种子，共27个实验
- 控制变量: 其他所有参数保持一致，仅温度参数变化

使用方法:
    python scripts/run_temp_ablation.py --parallel 4
    python scripts/run_temp_ablation.py --config A --seed 0  # 运行单个配置
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
from typing import List, Optional

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "temp_ablation"

# Ablation 配置定义 (9组)
ABLATION_CONFIGS = {
    "A": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "基线配置"},
    "B": {"T_start": 1.0, "T_end": 0.3, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_end=0.3 (更激进)"},
    "C": {"T_start": 1.0, "T_end": 0.5, "warmup": 0,  "schedule": "cosine", "learnable": True, "desc": "无 warmup"},
    "D": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "linear", "learnable": True, "desc": "linear 调度"},
    "E": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "exponential", "learnable": True, "desc": "exp 调度"},
    "F": {"T_start": 0.5, "T_end": 0.5, "warmup": 0,  "schedule": "linear", "learnable": False, "desc": "固定温度"},
    "G": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": False, "desc": "非学习温度"},
    "H": {"T_start": 1.5, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_start=1.5"},
    "I": {"T_start": 1.0, "T_end": 0.7, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_end=0.7"},
}


@dataclass
class ExperimentResult:
    """单个实验结果"""
    config: str
    seed: int
    final_acc: float
    best_acc: float
    epochs: int
    duration: float
    success: bool
    error_msg: Optional[str] = None


def get_base_args() -> List[str]:
    """
    获取基础训练参数 (控制变量)
    所有消融实验使用相同的超参数，仅温度参数变化
    """
    return [
        "--dataset", "tiny-imagenet",
        "--epochs", "60",           # I100-2: 加快实验速度
        "--dim", "256",             # 适中模型规模
        "--depth", "8",             # 适中深度
        "--heads", "6",             # 适中注意力头数
        "--dropout", "0.25",        # 正则化
        "--drop-path", "0.25",      # 正则化
        "--weight-decay", "0.15",
        "--batch-size", "64",
        "--lr", "5e-4",
        "--warmup-epochs", "10",
        "--use-amp",                # 混合精度
        "--gradient-checkpoint",    # 节省显存
        "--channels-last",
        "--save-interval", "10",
    ]


def build_args(config_name: str, seed: int) -> List[str]:
    """
    构建完整的训练命令行参数
    """
    cfg = ABLATION_CONFIGS[config_name]
    base_args = get_base_args()

    args = [
        "--seed", str(seed),
        "--splitter-temp-start", str(cfg["T_start"]),
        "--splitter-temp-end", str(cfg["T_end"]),
        "--splitter-temp-warmup", str(cfg["warmup"]),
        "--temp-schedule", cfg["schedule"],
    ]

    if cfg["learnable"]:
        args.append("--learnable-temperature")

    return base_args + args


def run_experiment(config_name: str, seed: int, timeout: Optional[int] = None) -> ExperimentResult:
    """
    运行单个消融实验

    Args:
        config_name: 配置名称 (A-I)
        seed: 随机种子
        timeout: 超时时间(秒)，None表示不限制

    Returns:
        ExperimentResult: 实验结果
    """
    cfg = ABLATION_CONFIGS[config_name]
    output_dir = OUTPUT_DIR / f"config_{config_name}" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用 uv run 直接调用 Python 脚本
    cmd = [
        "uv", "run", "python", "src/training/train_fractal_vit.py",
        "--save-dir", str(output_dir),
    ] + build_args(config_name, seed)

    print(f"\n{'='*60}")
    print(f"[I100-2] 运行实验: config={config_name}, seed={seed}")
    print(f"  T_start={cfg['T_start']}, T_end={cfg['T_end']}, "
          f"warmup={cfg['warmup']}, schedule={cfg['schedule']}, "
          f"learnable={cfg['learnable']}")
    print(f"  输出: {output_dir}")
    print(f"  命令: {' '.join(cmd)}")
    print(f"{'='*60}")

    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            timeout=timeout,
            capture_output=True,
            text=True
        )
        duration = time.time() - start_time

        if result.returncode == 0:
            # 解析结果
            final_acc, best_acc = parse_training_result(output_dir)
            print(f"[完成] config={config_name}, seed={seed}, "
                  f"final_acc={final_acc:.4f}, best_acc={best_acc:.4f}, "
                  f"耗时={duration/60:.1f}min")
            return ExperimentResult(
                config=config_name,
                seed=seed,
                final_acc=final_acc,
                best_acc=best_acc,
                epochs=60,
                duration=duration,
                success=True
            )
        else:
            error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
            print(f"[失败] config={config_name}, seed={seed}: {error_msg}")
            return ExperimentResult(
                config=config_name,
                seed=seed,
                final_acc=0.0,
                best_acc=0.0,
                epochs=0,
                duration=duration,
                success=False,
                error_msg=error_msg
            )
    except subprocess.TimeoutExpired:
        duration = time.time() - start_time
        print(f"[超时] config={config_name}, seed={seed}, 耗时>{timeout}s")
        return ExperimentResult(
            config=config_name,
            seed=seed,
            final_acc=0.0,
            best_acc=0.0,
            epochs=0,
            duration=duration,
            success=False,
            error_msg=f"Timeout after {timeout}s"
        )
    except Exception as e:
        duration = time.time() - start_time
        print(f"[错误] config={config_name}, seed={seed}: {str(e)}")
        return ExperimentResult(
            config=config_name,
            seed=seed,
            final_acc=0.0,
            best_acc=0.0,
            epochs=0,
            duration=duration,
            success=False,
            error_msg=str(e)
        )


def parse_training_result(output_dir: Path) -> tuple:
    """从训练输出目录解析最终精度和最佳精度"""
    metrics_file = output_dir / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
            return metrics.get("test_acc", 0.0), metrics.get("best_test_acc", 0.0)
    return 0.0, 0.0


def run_ablation_parallel(configs: List[str], seeds: List[int], max_parallel: int = 4) -> List[ExperimentResult]:
    """
    并行运行消融实验

    Args:
        configs: 要运行的配置列表
        seeds: 随机种子列表
        max_parallel: 最大并行数

    Returns:
        List[ExperimentResult]: 所有实验结果
    """
    import concurrent.futures

    results = []
    tasks = [(cfg, seed) for cfg in configs for seed in seeds]

    print(f"\n[I100-2] 开始消融实验")
    print(f"  配置: {configs}")
    print(f"  种子: {seeds}")
    print(f"  总实验数: {len(tasks)}")
    print(f"  最大并行: {max_parallel}")
    print(f"  输出目录: {OUTPUT_DIR}")

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as executor:
        futures = {
            executor.submit(run_experiment, cfg, seed): (cfg, seed)
            for cfg, seed in tasks
        }

        for future in concurrent.futures.as_completed(futures):
            cfg, seed = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[异常] config={cfg}, seed={seed}: {e}")
                results.append(ExperimentResult(
                    config=cfg, seed=seed,
                    final_acc=0.0, best_acc=0.0,
                    epochs=0, duration=0,
                    success=False, error_msg=str(e)
                ))

    return results


def analyze_results(results: List[ExperimentResult]) -> dict:
    """分析消融实验结果"""
    from collections import defaultdict
    import statistics

    config_results = defaultdict(list)
    for r in results:
        if r.success:
            config_results[r.config].append(r.final_acc)

    analysis = {}
    for config, accs in config_results.items():
        if len(accs) >= 1:
            mean_acc = statistics.mean(accs)
            std_acc = statistics.stdev(accs) if len(accs) > 1 else 0.0
            analysis[config] = {
                "mean": mean_acc,
                "std": std_acc,
                "results": accs,
                "count": len(accs)
            }

    return analysis


def print_summary(results: List[ExperimentResult]):
    """打印结果汇总"""
    print("\n" + "="*70)
    print("I100-2 温度消融实验结果汇总")
    print("="*70)

    analysis = analyze_results(results)

    # 按精度排序
    sorted_configs = sorted(analysis.items(), key=lambda x: x[1]["mean"], reverse=True)

    print(f"\n{'配置':<6} {'T_start':<8} {'T_end':<8} {'warmup':<7} {'schedule':<10} {'Learnable':<10} "
          f"{'Mean Acc':<10} {'Std':<8} {'N':<4}")
    print("-"*85)

    for config, stats in sorted_configs:
        cfg = ABLATION_CONFIGS[config]
        print(f"{config:<6} {cfg['T_start']:<8} {cfg['T_end']:<8} {cfg['warmup']:<7} "
              f"{cfg['schedule']:<10} {str(cfg['learnable']):<10} "
              f"{stats['mean']:.4f}     {stats['std']:.4f}   {stats['count']}")

    # 基线对比
    if "A" in analysis:
        baseline = analysis["A"]["mean"]
        print(f"\n基线配置 A 精度: {baseline:.4f}")
        print("\n相对基线改进:")
        for config, stats in sorted_configs:
            if config != "A":
                diff = stats["mean"] - baseline
                symbol = "+" if diff > 0 else ""
                print(f"  {config} vs A: {symbol}{diff:.4f} ({symbol}{diff/baseline*100:.2f}%)")

    # 失败实验
    failures = [r for r in results if not r.success]
    if failures:
        print(f"\n失败实验 ({len(failures)}个):")
        for f in failures:
            print(f"  {f.config}_seed_{f.seed}: {f.error_msg}")


def save_results(results: List[ExperimentResult]):
    """保存结果到JSON"""
    output_file = OUTPUT_DIR / "ablation_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    results_data = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": sum(1 for r in results if r.success),
        "results": [
            {
                "config": r.config,
                "seed": r.seed,
                "final_acc": r.final_acc,
                "best_acc": r.best_acc,
                "duration": r.duration,
                "success": r.success,
                "error": r.error_msg,
                "params": {
                    **ABLATION_CONFIGS[r.config],
                    "description": ABLATION_CONFIGS[r.config]["desc"]
                }
            }
            for r in results
        ]
    }

    with open(output_file, "w") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\n结果已保存: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="I100-2 温度消融实验运行器")
    parser.add_argument("--config", type=str, default=None,
                        help="指定配置 (A-I)，默认运行所有配置")
    parser.add_argument("--seed", type=int, default=None,
                        help="指定种子，默认运行所有种子 (0,1,2)")
    parser.add_argument("--seeds", type=str, default="0,1,2",
                        help="种子列表，逗号分隔 (默认: 0,1,2)")
    parser.add_argument("--parallel", type=int, default=2,
                        help="最大并行数 (默认: 2)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="单个实验超时时间(秒)")

    args = parser.parse_args()

    # 确定要运行的配置
    if args.config:
        configs = [args.config.upper()]
        if configs[0] not in ABLATION_CONFIGS:
            print(f"错误: 无效配置 '{args.config}', 可选: {list(ABLATION_CONFIGS.keys())}")
            sys.exit(1)
    else:
        configs = list(ABLATION_CONFIGS.keys())

    # 确定要运行的种子
    if args.seed is not None:
        seeds = [args.seed]
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    print(f"[I100-2] 温度消融实验")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  配置: {configs}")
    print(f"  种子: {seeds}")
    print(f"  输出: {OUTPUT_DIR}")

    # 运行实验
    if args.parallel <= 1:
        # 顺序执行
        results = []
        for cfg in configs:
            for seed in seeds:
                result = run_experiment(cfg, seed, args.timeout)
                results.append(result)
    else:
        # 并行执行
        results = run_ablation_parallel(configs, seeds, args.parallel)

    # 分析并保存结果
    print_summary(results)
    save_results(results)


if __name__ == "__main__":
    main()
