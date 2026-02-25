# -*- coding: utf-8 -*-
"""__main__.py - 允许使用 python -m src.training.train_fractal_vit 运行

I170-FIX: 添加 __main__.py 支持模块模式运行，修复相对导入问题
"""

from src.training.train_fractal_vit import main

if __name__ == "__main__":
    main()
