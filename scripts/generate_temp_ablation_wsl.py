#!/usr/bin/env python3
"""
I100-2: 温度参数 Ablation Study - WSL/Podman 批量运行脚本生成器

Usage:
    python scripts/generate_temp_ablation_wsl.py --dry-run
    python scripts/generate_temp_ablation_wsl.py --output run_all.sh --parallel 4
"""

import argparse
import os
import sys
from pathlib import Path

# 项目根目录 (WSL 路径)
PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_ROOT_WSL = "/mnt/d/myProject/fractal-curve-tokenizer"

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


def generate_wsl_run_script(config_name, config, seed, dataset):
    """生成 WSL/Podman 单个运行脚本"""
    T_start = config["T_start"]
    T_end = config["T_end"]
    warmup = config["warmup"]
    schedule = config["schedule"]
    learnable = "true" if config["learnable"] else "false"
    desc = config["desc"]

    script = '''#!/bin/bash
# I100-2 Ablation: 配置 ''' + config_name + ''' - ''' + desc + ''', seed=''' + str(seed) + '''
set -e
echo "[I100-2] 配置 ''' + config_name + ''' seed=''' + str(seed) + '''"
OUTPUT_DIR="experiments/temp_ablation/''' + config_name + '''/seed_''' + str(seed) + '''"
mkdir -p "$OUTPUT_DIR"
uv run python src/training/train_fractal_vit.py \\
    --dataset ''' + dataset + ''' \\
    --epochs 100 \\
    --seed ''' + str(seed) + ''' \\
    --splitter-temp-start ''' + str(T_start) + ''' \\
    --splitter-temp-end ''' + str(T_end) + ''' \\
    --splitter-temp-warmup ''' + str(warmup) + ''' \\
    --temp-schedule ''' + schedule + ''' \\
    --learnable-temperature ''' + learnable + ''' \\
    --save-dir "$OUTPUT_DIR" \\
    --batch-size 32 \\
    --dim 256 \\
    --depth 8 \\
    --heads 8 \\
    --use-amp \\
    --gradient-checkpoint
echo "[完成] 配置 ''' + config_name + ''' seed=''' + str(seed) + '''"
'''
    return script


def generate_wsl_analysis_script():
    """生成 WSL 结果分析脚本"""
    script = '''#!/bin/bash
# I100-2 Ablation 结果分析
RESULTS_DIR="experiments/temp_ablation"
echo "=== I100-2 温度参数 Ablation 结果 ==="
python3 -c "
import os
import statistics
configs = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
results = {}
for cfg in configs:
    results[cfg] = []
    cfg_dir = os.path.join(r'RESULTS_DIR', cfg)
    if os.path.exists(cfg_dir):
        for subdir in os.listdir(cfg_dir):
            log_file = os.path.join(cfg_dir, subdir, 'training.log')
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        if 'test_acc' in last_line:
                            try:
                                acc = float(last_line.split('test_acc:')[1].split()[0])
                                results[cfg].append(acc)
                            except:
                                pass
print('配置    Test Acc (mean+/-std)')
print('-' * 40)
for cfg in configs:
    accs = results[cfg]
    if accs:
        mean_acc = sum(accs) / len(accs)
        std_acc = statistics.stdev(accs) if len(accs) > 1 else 0
        print(cfg + '       ' + format(mean_acc, '.2%') + ' +/- ' + format(std_acc, '.2%'))
    else:
        print(cfg + '       N/A')
"
'''
    return script


def generate_parallel_runner(num_parallel):
    """生成并行批量运行脚本"""
    script = '''#!/bin/bash
# I100-2 Parallel Batch Runner (parallel=''' + str(num_parallel) + ''')
set -e

SCRIPT_DIR="scripts"
LOG_DIR="experiments/temp_ablation/logs"
MAX_PARALLEL=''' + str(num_parallel) + '''
mkdir -p "$LOG_DIR"

echo "=== I100-2 Parallel Run (parallel=''' + str(num_parallel) + ''') ==="

CONFIGS=(A B C D E F G H I)
SEEDS=(1 2 3)

# Generate all task pairs
TASKS=()
for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        TASKS+=("$cfg $seed")
    done
done

# Run tasks in parallel
total=${#TASKS[@]}
running=0
idx=0

next_task() {
    if (( idx >= total )); then
        return 1
    fi
    task="${TASKS[$idx]}"
    cfg=$(echo "$task" | cut -d' ' -f1)
    seed=$(echo "$task" | cut -d' ' -f2)
    ((idx++))

    local log="$LOG_DIR/${cfg}_seed_${seed}.log"
    echo "[START] $cfg seed=$seed"
    bash "$SCRIPT_DIR/run_ablation_${cfg}_seed_${seed}.sh" >> "$log" 2>&1 &
    return 0
}

# Start initial tasks
while (( running < MAX_PARALLEL )) && next_task; do
    ((running++))
done

# Wait for tasks to complete and start new ones
while (( running > 0 )); do
    wait -n
    ((running--))
    next_task && ((running++)) || true
done

echo "=== All Experiments Complete ==="
bash "$SCRIPT_DIR/analyze_results_wsl.sh"
'''
    return script


def generate_quick_commands():
    """生成快速运行命令"""
    return '''#!/bin/bash
# I100-2 Quick Run Commands

# 1. GNU parallel (recommended, requires parallel package)
cd /mnt/d/myProject/fractal-curve-tokenizer
parallel -j4 ::: scripts/run_ablation_*.sh

# 2. Simple background loop
cd /mnt/d/myProject/fractal-curve-tokenizer
for f in scripts/run_ablation_*.sh; do
    echo "Running: $f"
    bash "$f" &
done
wait

# 3. Check results
bash scripts/analyze_results_wsl.sh
'''


def main():
    parser = argparse.ArgumentParser(
        description="I100-2 温度参数 Ablation Study - WSL/Podman 批量运行脚本"
    )
    parser.add_argument("--seeds", type=int, default=3, help="种子数量 (默认: 3)")
    parser.add_argument("--dataset", type=str, default="tiny-imagenet", help="数据集")
    parser.add_argument("--dry-run", action="store_true", help="仅预览")
    parser.add_argument("--output", type=str, help="输出批量运行脚本")
    parser.add_argument("--parallel", type=int, default=4, help="并行数 (默认: 4)")
    parser.add_argument("--quick", action="store_true", help="生成快速运行命令")

    args = parser.parse_args()

    print("=== I100-2 WSL/Podman Ablation ===")
    print(f"配置: {len(ABLATION_CONFIGS)}, 种子: {args.seeds}, 总实验: {len(ABLATION_CONFIGS) * args.seeds}")
    print()

    if args.dry_run:
        for name, cfg in ABLATION_CONFIGS.items():
            print(f"  {name}: T={cfg['T_start']}->{cfg['T_end']}, warmup={cfg['warmup']}, schedule={cfg['schedule']}")
        return 0

    scripts_dir = PROJECT_ROOT / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    # 生成单个运行脚本
    for config_name, config in ABLATION_CONFIGS.items():
        for seed in range(1, args.seeds + 1):
            script = generate_wsl_run_script(config_name, config, seed, args.dataset)
            path = scripts_dir / f"run_ablation_{config_name}_seed_{seed}.sh"
            with open(path, "w") as f:
                f.write(script)
            os.chmod(path, 0o755)
            print(f"[生成] {path.relative_to(PROJECT_ROOT)}")

    # 生成分析脚本
    analysis = generate_wsl_analysis_script()
    with open(scripts_dir / "analyze_results_wsl.sh", "w") as f:
        f.write(analysis)
    os.chmod(scripts_dir / "analyze_results_wsl.sh", 0o755)
    print(f"[生成] {scripts_dir.relative_to(PROJECT_ROOT)}/analyze_results_wsl.sh")

    # 生成批量脚本
    if args.output:
        runner = generate_parallel_runner(args.parallel)
        output_path = PROJECT_ROOT / args.output
        with open(output_path, "w") as f:
            f.write(runner)
        os.chmod(output_path, 0o755)
        print(f"[生成] {output_path.relative_to(PROJECT_ROOT)}")
    elif args.quick:
        quick = generate_quick_commands()
        with open(scripts_dir / "quick_run_wsl.sh", "w") as f:
            f.write(quick)
        os.chmod(scripts_dir / "quick_run_wsl.sh", 0o755)
        print(f"[生成] {scripts_dir.relative_to(PROJECT_ROOT)}/quick_run_wsl.sh")

    print()
    print("=== WSL 批量运行命令 ===")
    print()
    print("# 在 WSL 中执行:")
    print()
    print("# 方式1: 并行运行 (推荐)")
    print("cd /mnt/d/myProject/fractal-curve-tokenizer")
    print("python scripts/generate_temp_ablation_wsl.py --output scripts/run_all.sh --parallel 4")
    print("bash scripts/run_all.sh")
    print()
    print("# 方式2: GNU parallel 并行")
    print("cd /mnt/d/myProject/fractal-curve-tokenizer")
    print("parallel -j4 ::: scripts/run_ablation_*.sh")
    print()
    print("# 方式3: 直接后台运行")
    print("cd /mnt/d/myProject/fractal-curve-tokenizer")
    print("for f in scripts/run_ablation_*.sh; do")
    print('    echo "运行: $f"')
    print('    bash "$f" &')
    print("done")
    print("wait")
    print()
    print("# 查看结果")
    print("bash scripts/analyze_results_wsl.sh")

    return 0


if __name__ == "__main__":
    sys.exit(main())
