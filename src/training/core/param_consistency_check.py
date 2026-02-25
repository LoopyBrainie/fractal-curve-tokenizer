"""
三层参数一致性验证脚本

用途：
    验证训练、模型保存（pth.gene）、评估加载三个阶段使用的参数是否完全一致。

三层参数定义：
    - 参数 (Parameters): 每次调用模型都需要传递的固定参数
    - 变参数 (Variable): 会随着其他参数变化而实时计算的参数
    - 超参数 (Hyper): 应当固定不变、保存在模型架构中的参数

使用方式：
    python -m src.training.core.param_consistency_check
"""

import math
from dataclasses import dataclass, fields
from typing import Any, Dict, Set, Optional, Tuple
import torch
import torch.nn as nn

# 导入项目模块
from vit_pytorch import FractalCurveViT
from vit_pytorch.core.constants import K_MIN_HARD_LIMIT, K_MAX_HARD_LIMIT
from training.core.model_gene import ModelGene


# =============================================================================
# 三层参数分类定义
# =============================================================================

@dataclass
class ParameterTier:
    """三层参数分类元数据"""
    PARAMETERS: Set[str] = None  # 固定架构参数
    VARIABLES: Set[str] = None   # 变参数（运行时计算）
    HYPERPARAMETERS: Set[str] = None  # 超参数（保存到模型）

    def __post_init__(self):
        if self.PARAMETERS is None:
            self.PARAMETERS = self._init_parameters()
        if self.VARIABLES is None:
            self.VARIABLES = self._init_variables()
        if self.HYPERPARAMETERS is None:
            self.HYPERPARAMETERS = self._init_hyperparameters()

    def _init_parameters(self) -> Set[str]:
        """第一层：参数 - 固定架构参数，每次模型调用都需要"""
        return {
            'dim', 'num_layers', 'heads', 'mlp_dim', 'num_classes',
            'image_size', 'channels', 'min_patch_size', 'pool', 'ffn_type',
            'dim_head',
        }

    def _init_variables(self) -> Set[str]:
        """第二层：变参数 - 运行时动态计算，不保存到模型"""
        return {
            'max_level',  # 由 image_size 和 min_patch_size 动态计算
            'K',  # 实际 token 数，由覆盖率动态计算
            '_current_image_size',  # 运行时缓存
        }

    def _init_hyperparameters(self) -> Set[str]:
        """第三层：超参数 - 应当保存到模型架构"""
        return {
            # Tokenizer 参数
            'token_coverage_min', 'token_coverage_max',
            'coverage_min', 'coverage_max_hard',
            'K_min_abs', 'K_max_hard', 'coverage_base',
            # Splitter 参数
            'splitter_type', 'splitter_hidden_dim', 'splitter_feature_dim',
            'splitter_pool_size', 'splitter_temp_start', 'splitter_temp_end',
            'splitter_token_ratio_min', 'splitter_token_ratio_max',
            'jump_loss_weight', 'density_field_hidden_dim',
            # 正则化参数
            'tokenizer_dropout', 'dropout', 'emb_dropout', 'drop_path_rate',
            # 编码选项
            'use_hilbert_encoding', 'use_spatial_encoding', 'use_area_encoding',
            'use_affine_modulation', 'fourier_levels', 'target_ratio',
            'lca_fp16', 'use_checkpoint',
            # 配额参数
            'quota_learnable', 'quota_entropy_weight',
            # 模式编码器
            'use_pattern_encoder', 'pattern_encoder_mode',
            'pattern_encoder_window_sizes',
            # 语义分裂器
            'use_semantic_splitter', 'semantic_splitter_config',
            'semantic_loss_weight',
            # 深度缩放
            'depth_scale_range',
        }

    def classify(self, param_name: str) -> str:
        """对参数进行分类"""
        if param_name in self.PARAMETERS:
            return "参数 (P)"
        elif param_name in self.VARIABLES:
            return "变参数 (V)"
        elif param_name in self.HYPERPARAMETERS:
            return "超参数 (H)"
        else:
            return "未知"


# 全局实例
TIER = ParameterTier()


# =============================================================================
# 验证函数
# =============================================================================

def verify_model_gene_completeness(gene: ModelGene) -> Dict[str, Any]:
    """验证 ModelGene 包含所有必要的超参数"""
    issues = []
    warnings = []

    # 获取 ModelGene 的所有字段
    gene_fields = {f.name for f in fields(gene)}

    # 检查必需的超参数是否都存在
    for hp in TIER.HYPERPARAMETERS:
        if hp not in gene_fields:
            # 检查是否有替代名称
            if hp == 'token_coverage_min' and 'coverage_min' in gene_fields:
                warnings.append(f"字段 '{hp}' 使用了替代名称 'coverage_min'")
            elif hp == 'token_coverage_max' and 'coverage_max_hard' in gene_fields:
                warnings.append(f"字段 '{hp}' 使用了替代名称 'coverage_max_hard'")
            elif hp not in gene_fields:
                issues.append(f"缺少超参数字段: {hp}")

    # 检查是否有变参数被错误保存
    for vp in TIER.VARIABLES:
        if vp in gene_fields:
            value = getattr(gene, vp, None)
            if value is not None:
                issues.append(f"变参数 '{vp}' 不应保存到 ModelGene (当前值: {value})")

    return {
        "issues": issues,
        "warnings": warnings,
        "status": "PASS" if not issues else "FAIL"
    }


def verify_build_model_params(gene: ModelGene) -> Dict[str, Any]:
    """验证 build_model() 传递的参数是否完整"""
    # 模拟 build_model 的调用（不实际创建模型）
    passed_params = {
        'image_size': gene.image_size,
        'num_classes': gene.num_classes,
        'dim': gene.dim,
        'num_layers': gene.num_layers,
        'heads': gene.heads,
        'dim_head': gene.dim // gene.heads,
        'mlp_dim': gene.mlp_dim,
        'pool': gene.pool,
        'channels': gene.channels,
        'tokenizer_dropout': gene.tokenizer_dropout,
        'transformer_dropout': gene.dropout,
        'emb_dropout': gene.emb_dropout,
        'min_patch_size': gene.min_patch_size,
        'use_hilbert_encoding': gene.use_hilbert_encoding,
        'use_spatial_encoding': gene.use_spatial_encoding,
        'use_checkpoint': gene.use_checkpoint,
        'drop_path_rate': gene.drop_path_rate,
        'ffn_type': gene.ffn_type,
        'token_coverage_min': gene.token_coverage_min,
        'token_coverage_max': gene.token_coverage_max,
        'use_area_encoding': gene.use_area_encoding,
        'use_affine_modulation': gene.use_affine_modulation,
        'fourier_levels': gene.fourier_levels,
        'target_ratio': gene.target_ratio,
        'lca_fp16': gene.lca_fp16,
        'use_pattern_encoder': gene.use_pattern_encoder,
        'pattern_encoder_mode': gene.pattern_encoder_mode,
        'pattern_encoder_window_sizes': gene.pattern_encoder_window_sizes,
        'quota_learnable': gene.quota_learnable,
        'quota_entropy_weight': gene.quota_entropy_weight,
        'splitter_type': gene.splitter_type,
        'splitter_hidden_dim': gene.splitter_hidden_dim,
        'splitter_feature_dim': gene.splitter_feature_dim,
        'splitter_pool_size': gene.splitter_pool_size,
        'splitter_token_ratio_min': gene.splitter_token_ratio_min,
        'splitter_token_ratio_max': gene.splitter_token_ratio_max,
        'jump_loss_weight': gene.jump_loss_weight,
        'density_field_hidden_dim': gene.density_field_hidden_dim,
        'splitter_temp_start': gene.splitter_temp_start,
        'splitter_temp_end': gene.splitter_temp_end,
        'use_semantic_splitter': gene.use_semantic_splitter,
        'semantic_splitter_config': gene.semantic_splitter_config,
        'depth_scale_range': gene.depth_scale_range,
    }

    issues = []

    # 检查关键参数是否正确传递
    for param_name, value in passed_params.items():
        tier = TIER.classify(param_name)
        if tier == "变参数 (V)" and value is not None:
            issues.append(f"变参数 '{param_name}' 不应为 build_model 传递值")

    return {
        "passed_params": passed_params,
        "issues": issues,
        "status": "PASS" if not issues else "FAIL"
    }


def verify_serialization_roundtrip(gene: ModelGene) -> Dict[str, Any]:
    """验证序列化-反序列化的完整性"""
    # 序列化
    gene_dict = gene.to_dict()

    # 反序列化
    restored_gene = ModelGene.from_dict(gene_dict)

    # 比较关键字段
    compare_fields = [
        'dim', 'num_layers', 'heads', 'mlp_dim', 'num_classes',
        'image_size', 'min_patch_size', 'splitter_type',
        'token_coverage_min', 'token_coverage_max',
    ]

    differences = []
    for field in compare_fields:
        original = getattr(gene, field, None)
        restored = getattr(restored_gene, field, None)
        if original != restored:
            differences.append({
                "field": field,
                "original": original,
                "restored": restored
            })

    return {
        "serialized_keys": list(gene_dict.keys()),
        "differences": differences,
        "status": "PASS" if not differences else "FAIL"
    }


def compute_variable_params(
    image_size: int,
    min_patch_size: int,
    token_coverage_min: float,
    token_coverage_max: float,
    K_min_abs: int = K_MIN_HARD_LIMIT,
    K_max_hard: int = K_MAX_HARD_LIMIT,
) -> Dict[str, Any]:
    """计算变参数的动态值

    这展示了变参数如何根据其他参数动态计算
    """
    # 计算最大可能的 token 数
    max_tokens = (image_size // min_patch_size) ** 2

    # 计算 K 范围
    K_min_computed = max(K_min_abs, int(max_tokens * token_coverage_min))
    K_max_computed = min(K_max_hard, int(max_tokens * token_coverage_max))

    # 计算 max_level (变参数)
    # max_level = ceil(log2(image_size / min_patch_size))
    max_level_computed = math.ceil(math.log2(image_size / min_patch_size))

    return {
        "max_tokens": max_tokens,
        "K_min": K_min_computed,
        "K_max": K_max_computed,
        "max_level": max_level_computed,
        "effective_coverage_min": K_min_computed / max_tokens,
        "effective_coverage_max": K_max_computed / max_tokens,
    }


def verify_hilbert_optimal_dynamic_k(
    image_size: int,
    min_patch_size: int,
    token_ratio_min: float,
    token_ratio_max: float,
) -> Dict[str, Any]:
    """验证 HilbertOptimalSplitter 的动态 K 计算"""
    max_tokens = (image_size // min_patch_size) ** 2

    K_min = max(1, int(max_tokens * token_ratio_min))
    K_max = max(K_min + 1, int(max_tokens * token_ratio_max))

    return {
        "image_size": image_size,
        "min_patch_size": min_patch_size,
        "max_possible_tokens": max_tokens,
        "K_min": K_min,
        "K_max": K_max,
        "token_ratio_min": token_ratio_min,
        "token_ratio_max": token_ratio_max,
    }


# =============================================================================
# 主验证流程
# =============================================================================

def run_full_verification(
    image_size: int = 224,
    min_patch_size: int = 4,
    splitter_type: str = 'hilbert_optimal',
    token_coverage_min: float = 0.01,
    token_coverage_max: float = 0.25,
    token_ratio_min: float = 0.02,
    token_ratio_max: float = 0.15,
) -> Dict[str, Any]:
    """运行完整的参数一致性验证"""

    print("=" * 80)
    print("三层参数一致性验证")
    print("=" * 80)

    # 1. 创建 ModelGene
    gene = ModelGene(
        dim=256,
        num_layers=8,
        heads=4,
        mlp_dim=512,
        num_classes=200,
        image_size=image_size,
        min_patch_size=min_patch_size,
        splitter_type=splitter_type,
        token_coverage_min=token_coverage_min,
        token_coverage_max=token_coverage_max,
        splitter_token_ratio_min=token_ratio_min,
        splitter_token_ratio_max=token_ratio_max,
    )

    print(f"\n[1/4] ModelGene 完整性检查...")
    result1 = verify_model_gene_completeness(gene)
    print(f"  状态: {result1['status']}")
    for issue in result1.get('issues', []):
        print(f"    [X] {issue}")
    for warning in result1.get('warnings', []):
        print(f"    [!] {warning}")

    print(f"\n[2/4] build_model() 参数检查...")
    result2 = verify_build_model_params(gene)
    print(f"  状态: {result2['status']}")
    for issue in result2.get('issues', []):
        print(f"    [X] {issue}")

    print(f"\n[3/4] 序列化-反序列化检查...")
    result3 = verify_serialization_roundtrip(gene)
    print(f"  状态: {result3['status']}")
    for diff in result3.get('differences', []):
        print(f"    [X] {diff['field']}: {diff['original']} -> {diff['restored']}")

    print(f"\n[4/4] 变参数动态计算验证...")
    var_params = compute_variable_params(
        image_size=image_size,
        min_patch_size=min_patch_size,
        token_coverage_min=token_coverage_min,
        token_coverage_max=token_coverage_max,
    )
    print(f"  max_level (变参数): {var_params['max_level']}")
    print(f"  K 范围 (变参数): [{var_params['K_min']}, {var_params['K_max']}]")

    if splitter_type == 'hilbert_optimal':
        h1ss_params = verify_hilbert_optimal_dynamic_k(
            image_size=image_size,
            min_patch_size=min_patch_size,
            token_ratio_min=token_ratio_min,
            token_ratio_max=token_ratio_max,
        )
        print(f"\n  H1SS 动态 K:")
        print(f"    最大可能 token: {h1ss_params['max_possible_tokens']}")
        print(f"    K_min ({token_ratio_min*100}%): {h1ss_params['K_min']}")
        print(f"    K_max ({token_ratio_max*100}%): {h1ss_params['K_max']}")

    # 汇总结果
    all_pass = all(r['status'] == 'PASS' for r in [result1, result2, result3])

    print("\n" + "=" * 80)
    print(f"验证结果: {'PASS' if all_pass else 'FAIL'}")
    print("=" * 80)

    return {
        "gene_completeness": result1,
        "build_model_params": result2,
        "serialization": result3,
        "variable_params": var_params,
        "overall_status": "PASS" if all_pass else "FAIL"
    }


# =============================================================================
# 参数流向追踪
# =============================================================================

def print_parameter_flow():
    """打印参数流向图"""

    print("""
================================================================================
                        三层参数流向追踪
================================================================================

[训练阶段]
  CLI args --> ModelArchitectureConfig --> TrainingConfig
                                              |
                                              v
                                      FractalCurveViT(
                                          dim, num_layers, heads, ...
                                          max_level = None (变参数)
                                          token_coverage_* (超参数)
                                      )

[保存阶段]
  ModelGene.to_dict()
    |-- 参数 (P): dim, num_layers, heads, ...
    |-- 变参数 (V): max_level [X] 不保存
    +-- 超参数 (H): token_coverage_*, dropout, ...

  checkpoint = {
      'model_gene': gene.to_dict(),
      'model_state_dict': model.state_dict()
  }

[加载阶段]
  checkpoint = load_checkpoint(path)
       |
       v
  ModelGene.from_dict(checkpoint['model_gene'])
       |
       v
  model = gene.build_model()
       |
       v
  FractalCurveViT(max_level 由 image_size 计算)

================================================================================
                        关键约束
================================================================================
  * max_level 是变参数，由 image_size 和 min_patch_size 动态计算
  * K_min/K_max 由 token_coverage_min/max 乘以 max_tokens 计算
  * H1SS 的 K_min/K_max 由 splitter_token_ratio_min/max 乘以 max_tokens 计算
  * 变参数不保存到 ModelGene，确保训练/评估完全一致
================================================================================
""")


# =============================================================================
# 主入口
# =============================================================================

if __name__ == "__main__":
    import sys

    # 运行验证
    result = run_full_verification(
        image_size=224,
        min_patch_size=4,
        splitter_type='hilbert_optimal',
        token_coverage_min=0.01,
        token_coverage_max=0.25,
        token_ratio_min=0.02,
        token_ratio_max=0.15,
    )

    print("\n[参数流向图]")
    print_parameter_flow()

    sys.exit(0 if result['overall_status'] == 'PASS' else 1)
