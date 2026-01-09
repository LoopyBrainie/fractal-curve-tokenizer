"""
Configuration System for Fractal Training
==========================================

支持功能:
1. YAML 配置加载和验证
2. 配置继承 (base → dataset → experiment)
3. 命令行覆盖
4. 类型安全的 dataclass 配置

设计原则:
- 配置即文档 - 所有参数有默认值和说明
- 类型安全 - 使用 dataclass 进行验证
- 可扩展 - 支持自定义配置节
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union, Type, TypeVar
from pathlib import Path
import yaml
import json
import copy


T = TypeVar("T")


# ============================================================================
# 基础配置数据类
# ============================================================================

@dataclass
class DataConfig:
    """数据加载配置"""
    dataset: str = "tiny-imagenet"
    data_dir: str = "./data"
    
    # 批次设置
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    
    # 数据增强
    image_size: int = 64
    random_crop: bool = True
    horizontal_flip: bool = True
    color_jitter: float = 0.4
    auto_augment: Optional[str] = None  # "rand-m9-mstd0.5"
    
    # 采样器
    sampler_type: str = "default"  # "default", "class_balanced", "progressive"
    sampler_beta: float = 0.9  # ClassBalancedSampler beta
    sampler_beta_min: float = 0.5  # ProgressiveSampler beta_min


@dataclass
class ModelConfig:
    """模型配置"""
    name: str = "fractal_vit"
    
    # 核心参数
    image_size: int = 64
    patch_size: int = 8
    embed_dim: int = 384
    depth: int = 10
    num_heads: int = 8
    num_classes: int = 200
    
    # MLP
    mlp_ratio: float = 4.0
    
    # Dropout
    dropout: float = 0.1
    attention_dropout: float = 0.1
    
    # Hilbert Curve 特定
    max_depth: int = 4
    splitter_hidden_dim: int = 128
    splitter_dropout: float = 0.1


@dataclass
class LossConfig:
    """损失函数配置
    
    支持的损失类型:
    - "cross_entropy": 标准 CE
    - "focal": Focal Loss (Lin et al., 2017)
    - "class_balanced": Class-Balanced CE (Cui et al., 2019)
    - "focal_cb": Focal + Class-Balanced 组合
    """
    type: str = "cross_entropy"
    
    # Focal Loss 参数
    focal_gamma: float = 2.0
    focal_alpha: Optional[List[float]] = None  # None = 自动计算
    
    # Class-Balanced 参数
    cb_beta: float = 0.9999
    
    # Label Smoothing
    label_smoothing: float = 0.0


@dataclass
class BudgetConfig:
    """资源预算配置
    
    数学形式:
    - FLOPS Budget: L_budget = λ · ReLU(FLOPS/Budget - 1)²
    - Token Budget: L_token = λ · ReLU(N/N_max - 1)²
    """
    enabled: bool = True
    
    # 预算类型
    type: str = "flops"  # "flops", "tokens", "depth_weighted"
    
    # FLOPS 预算
    flops_budget: float = 5.0e9  # 5 GFLOPS
    flops_lambda: float = 0.1
    
    # Token 预算
    token_budget: int = 256
    token_lambda: float = 0.1
    
    # 深度加权
    depth_alpha: float = 0.5  # w(d) = e^(alpha * d)
    
    # 预算调度
    schedule: str = "cosine"  # "constant", "linear", "cosine"
    budget_max_ratio: float = 1.5
    budget_min_ratio: float = 0.5


@dataclass
class OptimizerConfig:
    """优化器配置"""
    type: str = "adamw"  # "sgd", "adam", "adamw"
    
    # 基本参数
    lr: float = 1e-4
    weight_decay: float = 0.05
    
    # SGD 特定
    momentum: float = 0.9
    nesterov: bool = True
    
    # Adam 特定
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class SchedulerConfig:
    """学习率调度器配置"""
    type: str = "cosine"  # "step", "cosine", "linear", "warmup_cosine"
    
    # 通用参数
    min_lr: float = 1e-6
    
    # Warmup
    warmup_epochs: int = 5
    warmup_lr: float = 1e-6
    
    # StepLR 参数
    step_size: int = 30
    gamma: float = 0.1
    
    # MultiStepLR 参数
    milestones: List[int] = field(default_factory=lambda: [60, 80, 90])


@dataclass
class TrainingConfig:
    """训练配置"""
    num_epochs: int = 100
    
    # 梯度
    gradient_clip_norm: float = 1.0
    accumulation_steps: int = 1
    
    # 混合精度
    use_amp: bool = True
    
    # 验证
    validate_interval: int = 1
    
    # 检查点
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    monitor_metric: str = "val_accuracy"
    
    # 早停
    early_stopping: bool = True
    patience: int = 20
    
    # 日志
    log_interval: int = 50
    
    # 调试
    debug: bool = False
    max_batches_per_epoch: Optional[int] = None


@dataclass
class WandBConfig:
    """Weights & Biases 配置"""
    enabled: bool = False
    project: str = "fractal-vit"
    entity: Optional[str] = None
    name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # 元信息
    name: str = "default_experiment"
    description: str = ""
    seed: int = 42
    
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)


# ============================================================================
# 配置加载和合并工具
# ============================================================================

def _deep_update(base: Dict, update: Dict) -> Dict:
    """深度合并字典"""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _dict_to_dataclass(data: Dict[str, Any], cls: Type[T]) -> T:
    """将字典转换为 dataclass"""
    if not hasattr(cls, "__dataclass_fields__"):
        return data
    
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    kwargs = {}
    
    for field_name, field_type in field_types.items():
        if field_name in data:
            value = data[field_name]
            # 检查是否是嵌套 dataclass
            origin = getattr(field_type, "__origin__", None)
            if hasattr(field_type, "__dataclass_fields__") and isinstance(value, dict):
                kwargs[field_name] = _dict_to_dataclass(value, field_type)
            else:
                # 处理科学记数法字符串转换为 float
                if field_type == float and isinstance(value, str):
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                kwargs[field_name] = value
    
    return cls(**kwargs)


class ConfigLoader:
    """配置加载器
    
    支持:
    1. YAML 文件加载
    2. 配置继承 (通过 base 字段)
    3. 命令行覆盖
    
    用法:
    ```python
    loader = ConfigLoader()
    config = loader.load("configs/experiment.yaml", overrides={"training.lr": 1e-3})
    ```
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir) if config_dir else Path(".")
    
    def load_yaml(self, path: Union[str, Path]) -> Dict[str, Any]:
        """加载单个 YAML 文件"""
        path = Path(path)
        if not path.is_absolute():
            path = self.config_dir / path
        
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    
    def _resolve_inheritance(self, config: Dict[str, Any], seen: set) -> Dict[str, Any]:
        """解析配置继承"""
        if "base" not in config:
            return config
        
        base_path = config.pop("base")
        if base_path in seen:
            raise ValueError(f"Circular inheritance detected: {base_path}")
        seen.add(base_path)
        
        base_config = self.load_yaml(base_path)
        base_config = self._resolve_inheritance(base_config, seen)
        
        return _deep_update(base_config, config)
    
    def _apply_overrides(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """应用命令行覆盖
        
        支持点号路径, 如 "training.lr" -> {"training": {"lr": ...}}
        """
        result = copy.deepcopy(config)
        
        for key, value in overrides.items():
            parts = key.split(".")
            current = result
            
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            
            current[parts[-1]] = value
        
        return result
    
    def load(
        self,
        path: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> ExperimentConfig:
        """加载完整配置
        
        Args:
            path: YAML 配置文件路径
            overrides: 命令行覆盖参数
        
        Returns:
            ExperimentConfig 实例
        """
        config = self.load_yaml(path)
        config = self._resolve_inheritance(config, set())
        
        if overrides:
            config = self._apply_overrides(config, overrides)
        
        return self._to_experiment_config(config)
    
    def _to_experiment_config(self, config: Dict[str, Any]) -> ExperimentConfig:
        """将字典转换为 ExperimentConfig"""
        # 手动处理每个子配置
        return ExperimentConfig(
            name=config.get("name", "default_experiment"),
            description=config.get("description", ""),
            seed=config.get("seed", 42),
            data=_dict_to_dataclass(config.get("data", {}), DataConfig),
            model=_dict_to_dataclass(config.get("model", {}), ModelConfig),
            loss=_dict_to_dataclass(config.get("loss", {}), LossConfig),
            budget=_dict_to_dataclass(config.get("budget", {}), BudgetConfig),
            optimizer=_dict_to_dataclass(config.get("optimizer", {}), OptimizerConfig),
            scheduler=_dict_to_dataclass(config.get("scheduler", {}), SchedulerConfig),
            training=_dict_to_dataclass(config.get("training", {}), TrainingConfig),
            wandb=_dict_to_dataclass(config.get("wandb", {}), WandBConfig),
        )
    
    def from_dict(self, config: Dict[str, Any]) -> ExperimentConfig:
        """从字典创建配置"""
        return self._to_experiment_config(config)


def save_config(config: ExperimentConfig, path: Union[str, Path]) -> None:
    """保存配置到 YAML 文件"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # 转换为字典
    config_dict = asdict(config)
    
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)


def create_default_configs(output_dir: Union[str, Path] = "configs") -> None:
    """创建默认配置文件模板"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 基础配置
    base_config = ExperimentConfig()
    save_config(base_config, output_dir / "base.yaml")
    
    # Tiny-ImageNet 类别平衡配置
    balanced_config = ExperimentConfig(
        name="tiny_imagenet_balanced",
        description="Tiny-ImageNet with class-balanced training",
        data=DataConfig(
            dataset="tiny-imagenet",
            sampler_type="class_balanced",
            sampler_beta=0.9,
        ),
        loss=LossConfig(
            type="focal_cb",
            focal_gamma=2.0,
            cb_beta=0.9999,
        ),
        budget=BudgetConfig(
            enabled=True,
            type="flops",
            flops_budget=5.0e9,
        ),
    )
    save_config(balanced_config, output_dir / "tiny_imagenet_balanced.yaml")
    
    print(f"Default configs created in {output_dir}/")


# ============================================================================
# 导出
# ============================================================================

__all__ = [
    # 配置类
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "BudgetConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "WandBConfig",
    "ExperimentConfig",
    # 工具
    "ConfigLoader",
    "save_config",
    "create_default_configs",
]
