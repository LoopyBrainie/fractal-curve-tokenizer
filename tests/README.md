# 测试目录说明

该目录使用模块化结构管理自动化测试，示例脚本被移动到 `examples/`。

## 结构

```text
tests/
├── unit/                     # 轻量单元测试
│   └── test_fractal_tokenizer_smoke.py
├── integration/              # 预留的集成测试目录
└── README.md
```

## 运行测试

```bash
uv run pytest tests -v
```

> 当前仓库仅包含一个基础冒烟测试，建议在此目录下补充更多覆盖模型和tokenizer逻辑的测试用例。
