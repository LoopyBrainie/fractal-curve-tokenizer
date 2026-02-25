# -*- coding: utf-8 -*-
"""__main__.py - 允许使用 python -m src.training.train_fractal_vit 运行

I170-FIX: 修复模块导入问题 - 直接执行脚本而非导入
"""

import subprocess
import sys
from pathlib import Path

# 获取脚本路径
_script_path = Path(__file__).parent / "train_fractal_vit.py"

# 直接运行脚本，传递所有命令行参数
sys.exit(subprocess.run([sys.executable, str(_script_path)] + sys.argv[1:]).returncode)
