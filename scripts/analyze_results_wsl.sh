#!/bin/bash
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
