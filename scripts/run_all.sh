#!/bin/bash
# I100-2 批量运行 (并行数=4)
set -e
SCRIPT_DIR="scripts"
LOG_DIR="experiments/temp_ablation/logs"
mkdir -p "$LOG_DIR"
echo "=== I100-2 批量运行 (并行=4) ==="
CONFIGS=(A B C D E F G H I)
SEEDS=(1 2 3)
current=0
PIDS=()

run_exp() {
    local cfg=$1 seed=$2
    local log="$LOG_DIR/${cfg}_seed_${seed}.log"
    echo "[启动] $cfg seed=$seed"
    bash "$SCRIPT_DIR/run_ablation_${cfg}_seed_${seed}.sh" >> "$log" 2>&1 &
    PIDS+=($!)
}

for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        while (( current >= 4 )); do
            wait -n
            ((current--))
        done
        run_exp "$cfg" "$seed"
        ((current++))
    done
done
wait
echo "=== 所有实验完成 ==="
bash "$SCRIPT_DIR/analyze_results_wsl.sh"
