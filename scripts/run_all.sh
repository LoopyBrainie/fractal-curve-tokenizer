#!/bin/bash
# I100-2 Parallel Batch Runner (parallel=4)
set -e

SCRIPT_DIR="scripts"
LOG_DIR="experiments/temp_ablation/logs"
MAX_PARALLEL=4
mkdir -p "$LOG_DIR"

echo "=== I100-2 Parallel Run (parallel=4) ==="

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
