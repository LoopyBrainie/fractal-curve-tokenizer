# -*- coding: utf-8 -*-
"""__main__.py - 允许使用 python -m src.training.train_fractal_vit 运行

I170-FIX: 添加 __main__.py 支持模块模式运行，修复相对导入问题
"""

import sys
from pathlib import Path

# 设置项目路径，确保 vit_pytorch 等模块可以正确导入
# 必须在导入主模块之前执行
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_PATH = _PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 延迟导入主模块，确保路径已设置
from src.training.train_fractal_vit import main

if __name__ == "__main__":
    main()
