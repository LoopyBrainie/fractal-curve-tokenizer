#!/usr/bin/env python3
"""
I100-2: 温度参数 Ablation Study 脚本生成器

生成 9 组温度配置的运行脚本。

Usage:
    python scripts/generate_temp_ablation.py --dry-run
    python scripts/generate_temp_ablation.py --seeds 3 --dataset tiny-imagenet
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "ablation_temp"

# Ablation 配置定义
ABLATION_CONFIGS = {
    "A": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "基线配置"},
    "B": {"T_start": 1.0, "T_end": 0.3, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_end=0.3更激进"},
    "C": {"T_start": 1.0, "T_end": 0.5, "warmup": 0, "schedule": "cosine", "learnable": True, "desc": "无warmup"},
    "D": {"T_start": 1.0, "T_end": 0.5, "warmup": 5, "schedule": "cosine", "learnable": True, "desc": "warmup=5"},
    "E": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "linear", "learnable": True, "desc": "linear退火"},
    "F": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "exponential", "learnable": True, "desc": "exponential退火"},
    "G": {"T_start": 1.0, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": False, "desc": "固定温度"},
    "H": {"T_start": 1.5, "T_end": 0.5, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_start=1.5"},
    "I": {"T_start": 1.0, "T_end": 0.7, "warmup": 10, "schedule": "cosine", "learnable": True, "desc": "T_end=0.7"},
}


def generate_run_script(config_name: str, config: dict, seed: int, dataset: str) -> str:
    """生成单个运行脚本"""

    T_start = config["T_start"]
    T_end = config["T_end"]
    warmup = config["warmup"]
    schedule = config["schedule"]
    learnable = "true" if config["learnable"] else "false"
    desc = config["desc"]

    script = f'''@echo off
REM ============================================================================
REM I100-2 温度参数 Ablation Study
REM 配置: {config_name} - {desc}
REM 种子: {seed}
REM 数据集: {dataset}
REM ============================================================================

echo [I100-2] Ablation 配置 {config_name} (seed={seed})
echo   T_start={T_start}, T_end={T_end}, warmup={warmup}
echo   schedule={schedule}, learnable={learnable}
echo.

REM 设置输出目录
set OUTPUT_DIR={str(RUNS_DIR)}\\{config_name}\\seed_{seed}
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM 运行训练
uv run python src\\training\\train_fractal_vit.py \\
    --dataset {dataset} \\
    --epochs 100 \\
    --seed {seed} \\
    --splitter-temp-start {T_start} \\
    --splitter-temp-end {T_end} \\
    --splitter-temp-warmup {warmup} \\
    --temp-schedule {schedule} \\
    --learnable-temperature {learnable} \\
    --save-dir %OUTPUT_DIR% \\
    --batch-size 32 \\
    --dim 256 \\
    --depth 8 \\
    --heads 8 \\
    --use-amp \\
    --gradient-checkpoint

echo [完成] 配置 {config_name} seed={seed}
if %errorlevel% neq 0 (
    echo [错误] 配置 {config_name} seed={seed} 失败
    exit /b %errorlevel%
)

exit /b 0
'''
    return script


def generate_analysis_script():
    """生成结果分析脚本"""

    script = '''@echo off
REM ============================================================================
REM I100-2 Ablation 结果分析脚本
REM ============================================================================

echo [分析] 温度参数 Ablation 结果
echo.

set RESULTS_DIR=''' + str(RUNS_DIR) + '''

echo 读取各配置结果...
python -c "
import os
import json

configs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
results = {}

for cfg in configs:
    cfg_dir = os.path.join(RESULTS_DIR, cfg)
    if os.path.exists(cfg_dir):
        best_acc = 0
        for subdir in os.listdir(cfg_dir):
            log_file = os.path.join(cfg_dir, subdir, 'training.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        if 'test_acc' in last_line:
                            acc = float(last_line.split('test_acc:')[1].split()[0])
                            best_acc = max(best_acc, acc)
        results[cfg] = best_acc

print('')
print('=== I100-2 温度 Ablation 结果 ===')
print('配置    T_start  T_end    Test Acc')
print('-' * 40)
for cfg, acc in sorted(results.items(), key=lambda x: -x[1]):
    print(cfg + '       -        -        ' + format(acc, '.2%'))
"
'''
    return script


def generate_summary_markdown() -> str:
    """生成结果汇总 Markdown"""

    md = f'''# I100-2 温度参数 Ablation Study 结果

> 生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 实验配置

| 配置 | T_start | T_end | Warmup | Schedule | Learnable | 描述 |
|------|---------|-------|--------|----------|-----------|------|
'''
    for name, cfg in ABLATION_CONFIGS.items():
        learnable_str = "True" if cfg["learnable"] else "False"
        md += f"| {name} | {cfg['T_start']} | {cfg['T_end']} | {cfg['warmup']} | {cfg['schedule']} | {learnable_str} | {cfg['desc']} |\n"

    md += '''
## 结果汇总

| 配置 | Test Accuracy | 相对基线 | 收敛epoch | Token稳定性 |
|------|---------------|----------|-----------|-------------|
| A (基线) | - | - | - | - |
| B | - | - | - | - |
| C | - | - | - | - |
| D | - | - | - | - |
| E | - | - | - | - |
| F | - | - | - | - |
| G | - | - | - | - |
| H | - | - | - | - |
| I | - | - | - | - |

## 分析结论

### 1. 温度范围影响
- T_end=0.3 vs 0.5: ...

### 2. Warmup 影响
- warmup=0 vs 10: ...

### 3. 调度策略影响
- cosine vs linear vs exponential: ...

### 4. 可学习温度影响
- learnable=True vs False: ...

## 推荐配置

基于实验结果，推荐配置: **TBD**

'''
    return md


def main():
    parser = argparse.ArgumentParser(
        description="I100-2 温度参数 Ablation Study 脚本生成器"
    )
    parser.add_argument(
        "--seeds", type=int, default=3, help="随机种子数量 (默认: 3)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tiny-imagenet",
        choices=["tiny-imagenet", "cifar10"],
        help="数据集 (默认: tiny-imagenet)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印生成的脚本，不实际生成文件",
    )

    args = parser.parse_args()

    print(f"=== I100-2 温度参数 Ablation Study ===")
    print(f"配置数量: {len(ABLATION_CONFIGS)}")
    print(f"种子数量: {args.seeds}")
    print(f"总实验数: {len(ABLATION_CONFIGS) * args.seeds}")
    print(f"数据集: {args.dataset}")
    print()

    if args.dry_run:
        print("[Dry Run] 预览生成配置:")
        for config_name, config in ABLATION_CONFIGS.items():
            print(f"  {config_name}: T={config['T_start']}->{config['T_end']}, "
                  f"warmup={config['warmup']}, schedule={config['schedule']}")
        return 0

    # 生成运行脚本
    total_scripts = 0
    for config_name, config in ABLATION_CONFIGS.items():
        config_dir = RUNS_DIR / config_name
        config_dir.mkdir(parents=True, exist_ok=True)

        for seed in range(1, args.seeds + 1):
            script = generate_run_script(config_name, config, seed, args.dataset)
            script_path = config_dir / f"run_seed_{seed}.bat"
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(script)
            total_scripts += 1
            rel_path = script_path.relative_to(PROJECT_ROOT)
            print(f"[生成] {rel_path}")

    # 生成分析脚本
    analysis_script = generate_analysis_script()
    analysis_path = RUNS_DIR / "analyze_results.bat"
    with open(analysis_path, "w", encoding="utf-8") as f:
        f.write(analysis_script)
    print(f"[生成] {analysis_path.relative_to(PROJECT_ROOT)}")

    # 生成结果汇总模板
    summary_path = RUNS_DIR / "results_summary.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(generate_summary_markdown())
    print(f"[生成] {summary_path.relative_to(PROJECT_ROOT)}")

    print()
    print(f"=== 完成: 生成了 {total_scripts} 个运行脚本 ===")
    print(f"运行目录: {RUNS_DIR}")
    print()
    print("运行命令:")
    print(f"  cd {RUNS_DIR}")
    print(f"  for /r %f in (*.bat) do call \"%f\"")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
