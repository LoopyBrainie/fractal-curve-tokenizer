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
import logging

logger = logging.getLogger(__name__)


T = TypeVar("T")


# ============================================================================
# I36: 统一模型架构配置 (2026-01-20)
# ============================================================================

@dataclass
class ModelArchitectureConfig:
    """统一模型架构配置

    设计原则:
        - 所有模型架构参数集中定义，避免重复
        - 训练配置 (CUB200TrainingConfig) 只包含训练策略参数
        - 与 ModelConfig 保持一致，但更专注于 Fractal ViT 特有参数

    数学参数:
        - dim: 嵌入维度
        - depth: Transformer 层数
        - heads: 注意力头数
        - mlp_dim: FFN 维度 = dim × mlp_ratio
        - dim_head: 每头维度 = dim / heads

    Tokenizer 参数:
        - num_scales: 四叉树深度级别数
        - min_patch_size: 最小 patch 大小
        - K_min/K_max: Token 数量范围 (I33: 相对预算)
        - max_depth: 最大分割深度
    """
    # 核心架构参数
    num_classes: int = 200
    dim: int = 384
    depth: int = 8
    heads: int = 6
    mlp_dim: int = 1536  # dim * mlp_ratio (default 4.0)
    dim_head: int = 64   # dim / heads

    # 输入配置
    image_size: int = 224  # None 表示动态分辨率 (I78)
    patch_size: int = 8
    channels: int = 3

    # I78: 动态分辨率支持
    # image_size=None 时支持任意分辨率输入
    # 固定分辨率时用于初始化，兼容旧 API

    # Tokenizer 参数 (I33: 相对预算设计)
    # P2 Fix: num_scales 已废弃，使用 min_patch_size 动态计算 max_depth
    num_scales: int = field(default=4, metadata={
        "deprecated": True,
        "deprecated_since": "v99",
        "replacement": "使用 min_patch_size 和 max_depth_hard_limit 动态计算",
        "description": "四叉树深度级别数，已废弃"
    })
    min_patch_size: int = 4
    # I33: 相对预算参数 (替代绝对 K_min/K_max)
    token_coverage_min: float = 0.01   # α = 1% 最小覆盖率
    token_coverage_max: float = 0.05   # β = 5% 最大覆盖率
    K_min_abs: int = 8                 # 绝对下界保护
    max_depth_hard_limit: int = 8      # 最大分割深度硬上限

    # FFN 类型
    ffn_type: str = "swiglu_level"  # "swiglu", "swiglu_level"

    # 性能优化选项
    use_checkpoint: bool = False
    use_channels_last: bool = False  # I78: channels-last 内存格式
    use_compile: bool = False        # I78: torch.compile 优化
    compile_mode: str = "default"    # torch.compile 模式

    # I31: 形状-尺度编码配置
    use_area_encoding: bool = False
    use_affine_modulation: bool = True  # A17: 默认为 True
    fourier_levels: int = 4             # 傅里叶特征级别数

    # I27: 子模块 Dropout 配置
    splitter_dropout: Optional[float] = None  # None = 自动 = min(dropout, 0.15)
    pos_dropout: Optional[float] = None       # None = 自动 = dropout * 0.5

    # I24-2: 可学习配额控制
    # None = 使用常量默认值, True/False = 显式覆盖
    quota_learnable: Optional[bool] = None

    def __post_init__(self):
        """参数验证 - 数学约束"""
        assert self.dim > 0, f"dim={self.dim} 必须 > 0"
        assert self.depth >= 1, f"depth={self.depth} 必须 >= 1"
        assert self.heads >= 1, f"heads={self.heads} 必须 >= 1"
        assert self.num_classes >= 1, f"num_classes={self.num_classes} 必须 >= 1"

        # P2 Fix: 废弃参数警告
        import warnings
        if hasattr(self, 'num_scales'):
            field_info = self.__dataclass_fields__.get('num_scales')
            if field_info and field_info.metadata.get('deprecated', False):
                warnings.warn(
                    "num_scales 参数已废弃 (v99)，请使用 min_patch_size 和 max_depth_hard_limit 替代。",
                    DeprecationWarning,
                    stacklevel=2
                )

        # 验证 dim_head 一致性
        expected_dim_head = self.dim // self.heads
        if self.dim_head != expected_dim_head:
            logger.warning(
                f"dim_head={self.dim_head} != dim/heads={expected_dim_head}, "
                f"将使用 dim_head={expected_dim_head}"
            )
            self.dim_head = expected_dim_head

        # 验证 mlp_ratio
        expected_mlp_dim = int(self.dim * (self.mlp_dim / self.dim)) if self.mlp_dim != self.dim else int(self.dim * 4.0)
        # mlp_dim 已经是绝对值，不需要重新计算

    @property
    def mlp_ratio(self) -> float:
        """返回 mlp_ratio (用于向后兼容)"""
        return self.mlp_dim / self.dim if self.dim > 0 else 4.0


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


# I97-5: ModelConfig 已弃用，请使用 ModelArchitectureConfig
# 保留别名以保持向后兼容性
ModelConfig = ModelArchitectureConfig


@dataclass
class LossConfig:
    """损失函数配置
    
    支持的损失类型:
    - "cross_entropy": 标准 CE
    - "focal": Focal Loss (Lin et al., 2017)
    - "class_balanced": Class-Balanced CE (Cui et al., 2019)
    - "focal_cb": Focal + Class-Balanced 组合
    
    I25-1 Focal Loss 数学指南
    =========================
    
    标准 Cross-Entropy:
        L_CE = -log(p_t)
        
    Focal Loss:
        L_FL = -α_t (1 - p_t)^γ log(p_t)
        
    梯度对比 (γ=2):
        | p_t | CE梯度 | FL梯度 | 比例 |
        |-----|--------|--------|------|
        | 0.9 | 0.11   | 0.0011 | 100x |
        | 0.5 | 1.0    | 0.25   | 4x   |
        | 0.1 | 10.0   | 8.1    | 1.2x |
        
    效果: 易分类样本梯度降低 100x，难分类样本梯度基本保持
    
    推荐配置:
        - 类别不均衡严重 (accuracy std > 15%): 启用 Focal Loss
        - γ = 2.0: 平衡配置
        - γ = 2.5: 推荐值 (I28-1 优化，难/易比 243x)
        - γ = 3.0-5.0: 极端不均衡时使用
        - 同时启用 class_balanced 以获得 α_t 权重
        
    I28-1 γ 参数优化形式化分析
    ==========================
    
    问题数据 (实验 20260114):
        - Worst class 准确率: 8-10% (p_t ≈ 0.10)
        - Best class 准确率: 82-92% (p_t ≈ 0.90)
        - accuracy_std: 17.81%
        
    梯度权重公式:
        w(p_t) = (1 - p_t)^γ
        
    难/易梯度比:
        R(γ) = w(0.10) / w(0.90) = 0.90^γ / 0.10^γ = 9^γ
        
        | γ   | 难/易比 |
        |-----|---------|
        | 2.0 | 81x     |
        | 2.5 | 243x    | ← 推荐
        | 3.0 | 729x    |
        
    稳定性约束:
        中等样本梯度保留率 = 0.5^γ
        γ = 2.5 时保留 71%，γ = 3.0 时仅保留 50%
        
    最优解推导:
        max R(γ) s.t. 0.5^γ ≥ 0.15
        → γ ≤ ln(0.15)/ln(0.5) = 2.74
        → 推荐 γ = 2.5
    """
    type: str = "cross_entropy"
    
    # Focal Loss 参数
    # γ (gamma): 聚焦参数，控制对易分类样本的抑制强度
    # γ = 0: 退化为标准 CE
    # γ = 2.5: 推荐值 (I28-1 优化，243x 抑制易分类样本梯度)
    # γ = 5: 极端聚焦 (仅对非常难的样本有梯度)
    focal_gamma: float = 2.5  # I28-1: 从 2.0 提升到 2.5
    focal_alpha: Optional[List[float]] = None  # None = 自动计算

    # Class-Balanced 参数
    # β (beta): 有效样本数衰减因子
    # β → 1: 权重更平滑
    # β → 0: 权重差异更大
    cb_beta: float = 0.9999
    
    # Label Smoothing
    # ε: 平滑因子，y'_c = (1-ε)y_c + ε/C
    # 推荐: 0.1 (轻微正则化), 0.2 (强正则化)
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
class EvaluationConfig:
    """分层评估配置
    
    控制评估器的行为和资源使用。
    
    数学形式化
    ==========
    评估采样策略:
        N_eval = min(N_dataset, max_samples)
    
    分层控制:
        enabled_layers ⊆ {L1, L2, L3, L4, L5, L6}
    
    ECE 计算:
        ECE = Σ_{m=1}^{n_bins} (|B_m|/N) |acc(B_m) - conf(B_m)|
    
    示例
    ----
    >>> config = EvaluationConfig(
    ...     enabled_layers=["L1", "L2", "L5"],
    ...     max_samples=5000,
    ...     eval_interval=5,
    ... )
    """
    
    # 总体开关
    enabled: bool = True
    
    # 启用的评估层
    enabled_layers: List[str] = field(
        default_factory=lambda: ["L1", "L2", "L3", "L4", "L5", "L6"]
    )
    
    # 评估频率 (每 N 个 epoch 执行一次分层评估)
    eval_interval: int = 10  # 默认每 10 个 epoch
    eval_on_best: bool = True  # 当模型达到新的 best 时评估
    
    # 采样限制
    max_samples: int = 10000  # 最大评估样本数
    batch_size: int = 64
    
    # L1 分类配置
    l1_ece_n_bins: int = 15  # ECE 计算的 bin 数量
    l1_top_k: List[int] = field(default_factory=lambda: [1, 5])
    l1_confusion_top_n: int = 10  # 保留前 N 个混淆对
    
    # L2 Tokenizer 配置
    l2_max_batches: int = 50  # Tokenizer 评估的最大批次数
    l2_compute_correlation: bool = True  # 是否计算内容-token 相关性
    
    # L3 注意力配置
    l3_max_batches: int = 20
    l3_entropy_threshold_low: float = 0.1  # 低熵阈值 (dead head)
    l3_entropy_threshold_high: float = 5.0  # 高熵阈值
    l3_capture_attention: bool = True  # 是否捕获注意力权重
    
    # L4 表示配置
    l4_max_samples: int = 2000  # Fisher 比计算的最大样本数
    
    # L5 效率配置
    l5_warmup_runs: int = 10
    l5_timing_runs: int = 50
    l5_measure_components: bool = True  # 是否分解组件延迟
    
    # L6 稳定性配置
    l6_check_splitter: bool = True  # 是否检查分割器健康
    
    # 输出配置
    save_report: bool = True
    report_format: str = "json"  # "json" or "yaml"
    report_dir: Optional[str] = None  # 默认保存到 checkpoint 目录
    
    # 可视化
    generate_visualizations: bool = False  # 是否生成可视化图表


@dataclass
class ExperimentConfig:
    """完整实验配置"""
    # 元信息
    name: str = "default_experiment"
    description: str = ""
    seed: int = 42
    
    # 子配置
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandBConfig = field(default_factory=WandBConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


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
            model=_dict_to_dataclass(config.get("model", {}), ModelArchitectureConfig),
            loss=_dict_to_dataclass(config.get("loss", {}), LossConfig),
            budget=_dict_to_dataclass(config.get("budget", {}), BudgetConfig),
            optimizer=_dict_to_dataclass(config.get("optimizer", {}), OptimizerConfig),
            scheduler=_dict_to_dataclass(config.get("scheduler", {}), SchedulerConfig),
            training=_dict_to_dataclass(config.get("training", {}), TrainingConfig),
            wandb=_dict_to_dataclass(config.get("wandb", {}), WandBConfig),
            evaluation=_dict_to_dataclass(config.get("evaluation", {}), EvaluationConfig),
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
    # I36: 统一模型架构配置
    "ModelArchitectureConfig",
    # 配置类
    "DataConfig",
    "ModelConfig",
    "LossConfig",
    "BudgetConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "TrainingConfig",
    "WandBConfig",
    "EvaluationConfig",
    "ExperimentConfig",
    # 工具
    "ConfigLoader",
    "save_config",
    "create_default_configs",
]
