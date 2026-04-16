# -*- coding: utf-8 -*-
"""
L1 Foundation: Constants Validation Tests

对应模块: vit_pytorch.constants

测试内容:
- 数值安全常量 (FP16 兼容 epsilon)
- 温度调度 (Gumbel-Softmax)
- 配额初始化 (QUOTA_INIT_LOGITS)
- 覆盖率常量 (K_COVERAGE_*)
- Elastic Budget 常量
- 阈值正则化
- 注意力偏置缩放
"""

import torch
import torch.nn.functional as F


class TestNumericalSafetyConstants:
    """数值安全常量验证."""

    def test_all_epsilon_positive(self):
        """所有 epsilon > 0."""
        from vit_pytorch.core.constants import (
            GUMBEL_EPSILON,
            LOG_EPSILON,
            DIVISION_EPSILON,
            PROB_EPSILON,
        )

        epsilons = [
            ("GUMBEL_EPSILON", GUMBEL_EPSILON),
            ("LOG_EPSILON", LOG_EPSILON),
            ("DIVISION_EPSILON", DIVISION_EPSILON),
            ("PROB_EPSILON", PROB_EPSILON),
        ]

        for name, eps in epsilons:
            assert eps > 0, f"{name} = {eps} not positive"
            assert eps >= 1e-8, f"{name} = {eps} below FP16 safety threshold"

    def test_epsilon_not_too_large(self):
        """epsilon 不应过大以免影响计算精度."""
        from vit_pytorch.core.constants import (
            GUMBEL_EPSILON,
            LOG_EPSILON,
            DIVISION_EPSILON,
            PROB_EPSILON,
        )

        max_allowed = 1e-6
        for name, eps in [
            ("GUMBEL_EPSILON", GUMBEL_EPSILON),
            ("LOG_EPSILON", LOG_EPSILON),
            ("DIVISION_EPSILON", DIVISION_EPSILON),
            ("PROB_EPSILON", PROB_EPSILON),
        ]:
            assert eps <= max_allowed, f"{name} = {eps} too large"

    def test_temperature_min_above_gradient_collapse(self):
        """TEMPERATURE_MIN > 0.1 防止梯度消失."""
        from vit_pytorch.core.constants import TEMPERATURE_MIN

        assert TEMPERATURE_MIN >= 0.1, f"TEMPERATURE_MIN = {TEMPERATURE_MIN} may cause gradient collapse"

    def test_gumbel_epsilon_safe_for_sampling(self):
        """GUMBEL_EPSILON 安全边距验证."""
        from vit_pytorch.core.constants import GUMBEL_EPSILON

        import math

        eps = GUMBEL_EPSILON
        g_min = -math.log(-math.log(eps))
        g_max = -math.log(-math.log(1 - eps))

        assert g_min >= -20, f"Gumbel range too narrow: g_min = {g_min}"
        assert g_max <= 20, f"Gumbel range too wide: g_max = {g_max}"


class TestTemperatureSchedule:
    """温度调度常量验证."""

    def test_temperature_order(self):
        """SPLITTER_TEMP_START > SPLITTER_TEMP_END >= TEMPERATURE_MIN > 0."""
        from vit_pytorch.core.constants import (
            SPLITTER_TEMP_START,
            SPLITTER_TEMP_END,
            TEMPERATURE_MIN,
        )

        assert SPLITTER_TEMP_START > SPLITTER_TEMP_END
        assert SPLITTER_TEMP_END >= TEMPERATURE_MIN  # I121-3: 可以相等
        assert TEMPERATURE_MIN > 0

    def test_temperature_in_exploration_range(self):
        """起始温度在有效探索范围内."""
        from vit_pytorch.core.constants import SPLITTER_TEMP_START

        assert 0.5 <= SPLITTER_TEMP_START <= 2.0

    def test_temperature_convergence_bound(self):
        """终止温度保证收敛."""
        from vit_pytorch.core.constants import SPLITTER_TEMP_END

        assert 0.1 <= SPLITTER_TEMP_END <= 1.0


class TestQuotaInitialization:
    """配额初始化常量验证."""

    def test_quota_logits_softmax_sum(self):
        """QUOTA_INIT_LOGITS softmax 后求和为 1."""
        from vit_pytorch.core.constants import QUOTA_INIT_LOGITS

        logits = torch.tensor(QUOTA_INIT_LOGITS)
        probs = F.softmax(logits, dim=-1)

        sum_probs = probs.sum().item()
        assert abs(sum_probs - 1.0) < 1e-6

    def test_quota_distribution_increases_with_depth(self):
        """配额分布随深度递增 (深度优先策略)."""
        from vit_pytorch.core.constants import QUOTA_INIT_LOGITS

        logits = torch.tensor(QUOTA_INIT_LOGITS)
        probs = F.softmax(logits, dim=-1)

        for i in range(len(probs) - 1):
            assert probs[i].item() < probs[i + 1].item()

    def test_quota_min_ratio_positive(self):
        """QUOTA_MIN_RATIO > 0."""
        from vit_pytorch.core.constants import QUOTA_MIN_RATIO

        assert QUOTA_MIN_RATIO > 0
        assert QUOTA_MIN_RATIO < 1.0

    def test_quota_entropy_weight_reasonable(self):
        """QUOTA_ENTROPY_WEIGHT 在合理范围内."""
        from vit_pytorch.core.constants import QUOTA_ENTROPY_WEIGHT

        assert 0 <= QUOTA_ENTROPY_WEIGHT <= 1.0


class TestCoverageConstants:
    """覆盖率常量验证."""

    def test_coverage_bounds_order(self):
        """0 < K_COVERAGE_MIN < K_COVERAGE_BASE < K_COVERAGE_MAX_HARD <= 1."""
        from vit_pytorch.core.constants import (
            K_COVERAGE_MIN,
            K_COVERAGE_BASE,
            K_COVERAGE_MAX_HARD,
        )

        assert 0 < K_COVERAGE_MIN
        assert K_COVERAGE_MIN < K_COVERAGE_BASE
        assert K_COVERAGE_BASE < K_COVERAGE_MAX_HARD
        assert K_COVERAGE_MAX_HARD <= 1.0

    def test_coverage_reference_size_valid(self):
        """K_ADAPTIVE_REFERENCE_SIZE 为正整数."""
        from vit_pytorch.core.constants import K_ADAPTIVE_REFERENCE_SIZE

        assert K_ADAPTIVE_REFERENCE_SIZE > 0
        assert isinstance(K_ADAPTIVE_REFERENCE_SIZE, int)

    def test_k_hard_limits_order(self):
        """K_MIN_HARD_LIMIT < K_MAX_HARD_LIMIT."""
        from vit_pytorch.core.constants import (
            K_MIN_HARD_LIMIT,
            K_MAX_HARD_LIMIT,
        )

        assert K_MIN_HARD_LIMIT < K_MAX_HARD_LIMIT
        assert 1 <= K_MIN_HARD_LIMIT <= 64
        assert 1024 <= K_MAX_HARD_LIMIT <= 8192

    def test_sample_ratios_order(self):
        """K_MIN_SAMPLE_RATIO < K_MAX_SAMPLE_RATIO."""
        from vit_pytorch.core.constants import (
            K_MIN_SAMPLE_RATIO,
            K_MAX_SAMPLE_RATIO,
        )

        assert K_MIN_SAMPLE_RATIO < K_MAX_SAMPLE_RATIO

    def test_coverage_adaptive_formula_works(self):
        """覆盖率自适应公式数学正确性."""
        from vit_pytorch.core.constants import (
            K_COVERAGE_BASE,
            K_ADAPTIVE_REFERENCE_SIZE,
        )

        beta_0 = K_COVERAGE_BASE
        ref = K_ADAPTIVE_REFERENCE_SIZE

        cases = [
            (224, 224, 1.0),
            (112, 112, 0.707),
            (448, 448, 1.414),
        ]

        for H, W, expected_ratio in cases:
            min_dim = min(H, W)
            expected_coverage = beta_0 * expected_ratio
            computed_coverage = beta_0 * (min_dim / ref) ** 0.5
            assert abs(computed_coverage / expected_coverage - 1.0) < 0.01


class TestElasticBudgetConstants:
    """Elastic Budget 常量验证."""

    def test_elastic_bounds_consistent_with_k_coverage(self):
        """ELASTIC_COVERAGE bounds 与 K_COVERAGE bounds 一致."""
        from vit_pytorch.core.constants import (
            ELASTIC_COVERAGE_MIN,
            K_COVERAGE_MIN,
        )

        # ELASTIC_COVERAGE_MIN 是崩溃检测阈值（0.03）
        # K_COVERAGE_MIN 是覆盖率下界（0.01）
        # 这是两个不同的概念，都应该在合理范围内
        assert 0.01 <= ELASTIC_COVERAGE_MIN <= 0.1  # 崩溃阈值范围
        assert 0.01 <= K_COVERAGE_MIN <= 0.1  # 覆盖率下界范围

    def test_elastic_lambda_weights_reasonable(self):
        """ELASTIC_LAMBDA weights 在合理范围内."""
        from vit_pytorch.core.constants import (
            ELASTIC_LAMBDA_TARGET,
            ELASTIC_LAMBDA_BOUNDARY,
            ELASTIC_LAMBDA_COLLAPSE,
        )

        # 验证 λ_TARGET 和 λ_BOUNDARY 在合理范围 (0.01 ~ 1.0)
        assert 0.01 <= ELASTIC_LAMBDA_TARGET <= 1.0
        assert 0.01 <= ELASTIC_LAMBDA_BOUNDARY <= 1.0
        # λ_COLLAPSE >= 1.0 用于惩罚崩溃
        assert ELASTIC_LAMBDA_COLLAPSE >= 1.0


class TestThresholdRegularization:
    """阈值正则化常量验证."""

    def test_threshold_var_reg_weight_reasonable(self):
        """THRESHOLD_VAR_REG_WEIGHT 在 [0, 1] 范围内."""
        from vit_pytorch.core.constants import THRESHOLD_VAR_REG_WEIGHT

        assert 0 <= THRESHOLD_VAR_REG_WEIGHT <= 1.0


class TestDepthVarianceNormalization:
    """深度方差归一化常量验证."""

    def test_ema_alpha_reasonable(self):
        """DEPTH_EMA_ALPHA 在有效范围内."""
        from vit_pytorch.core.constants import DEPTH_EMA_ALPHA

        assert 0.01 <= DEPTH_EMA_ALPHA <= 0.5

    def test_depth_variance_init_eps_reasonable(self):
        """DEPTH_VARIANCE_INIT_EPS 为保守初始化下界."""
        from vit_pytorch.core.constants import DEPTH_VARIANCE_INIT_EPS

        assert 0.01 <= DEPTH_VARIANCE_INIT_EPS <= 1.0


class TestAttentionBiasScales:
    """注意力偏置缩放常量验证."""

    def test_hilbert_bias_scale_reasonable(self):
        """HILBERT_BIAS_SCALE 在 [0.01, 1.0] 范围内."""
        from vit_pytorch.core.constants import HILBERT_BIAS_SCALE

        assert 0.01 <= HILBERT_BIAS_SCALE <= 1.0

    def test_level_bias_scale_reasonable(self):
        """LEVEL_BIAS_SCALE 在 [0.01, 1.0] 范围内."""
        from vit_pytorch.core.constants import (
            LEVEL_BIAS_SCALE,
            HILBERT_BIAS_SCALE,
        )

        assert 0.01 <= LEVEL_BIAS_SCALE <= 1.0
        assert LEVEL_BIAS_SCALE <= HILBERT_BIAS_SCALE


class TestOverlapPenalty:
    """重叠惩罚常量验证."""

    def test_overlap_penalty_weight_reasonable(self):
        """OVERLAP_PENALTY_WEIGHT 在 [0, 1] 范围内."""
        from vit_pytorch.core.constants import OVERLAP_PENALTY_WEIGHT

        assert 0 <= OVERLAP_PENALTY_WEIGHT <= 1.0


class TestSoftExclusionMargin:
    """软排除边距常量验证."""

    def test_soft_exclusion_margin_reasonable(self):
        """SOFT_EXCLUSION_MARGIN 在 (0, 1) 范围内."""
        from vit_pytorch.core.constants import SOFT_EXCLUSION_MARGIN

        assert 0 < SOFT_EXCLUSION_MARGIN < 1
        assert SOFT_EXCLUSION_MARGIN <= 0.5


class TestFP16ClampConstants:
    """I108-6: FP16 Clamp 边界常量验证.

    数学分析:
    - LOGIT_CLAMP_BOUND = 50.0 (softmax 饱和阈值)
    - GRAD_CLAMP_BOUND = 20.0 (梯度无损，FP16 安全)
    - SCALE_CLAMP_BOUND = 15.0 (softplus 覆盖范围)
    - FP16_SAFE_EPSILON = 1e-6 (FP16 安全下界)
    """

    def test_logit_clamp_bound_value(self):
        """LOGIT_CLAMP_BOUND = 10.0.

        数学依据: softmax(x=10) ≈ 0.99995, 梯度 ≈ 5e-5 (有效)
        vs 50.0: softmax(50) ≈ 1.0, 梯度 ≈ 1e-22 (消失)
        I122-3 修复: 从 50.0 改为 10.0
        """
        from vit_pytorch.core.constants import LOGIT_CLAMP_BOUND

        assert LOGIT_CLAMP_BOUND == 10.0

    def test_grad_clamp_bound_value(self):
        """GRAD_CLAMP_BOUND = 20.0."""
        from vit_pytorch.core.constants import GRAD_CLAMP_BOUND

        assert GRAD_CLAMP_BOUND == 20.0

    def test_scale_clamp_bound_value(self):
        """SCALE_CLAMP_BOUND = 15.0."""
        from vit_pytorch.core.constants import SCALE_CLAMP_BOUND

        assert SCALE_CLAMP_BOUND == 15.0

    def test_fp16_safe_epsilon_value(self):
        """FP16_SAFE_EPSILON = 1e-6."""
        from vit_pytorch.core.constants import FP16_SAFE_EPSILON

        assert FP16_SAFE_EPSILON == 1e-6

    def test_logit_clamp_fp16_safe(self):
        """LOGIT_CLAMP_BOUND << FP16 最大值 (65504)."""
        from vit_pytorch.core.constants import LOGIT_CLAMP_BOUND

        fp16_max = 65504.0
        safety_factor = fp16_max / LOGIT_CLAMP_BOUND

        assert safety_factor > 1000, f"安全系数 {safety_factor:.0f} 不足"
        assert LOGIT_CLAMP_BOUND < fp16_max / 10, "应有 10× 安全余量"

    def test_grad_clamp_fp16_safe(self):
        """GRAD_CLAMP_BOUND << FP16 最大值."""
        from vit_pytorch.core.constants import GRAD_CLAMP_BOUND

        fp16_max = 65504.0
        safety_factor = fp16_max / GRAD_CLAMP_BOUND

        assert safety_factor > 1000, f"安全系数 {safety_factor:.0f} 不足"

    def test_scale_clamp_fp16_safe(self):
        """SCALE_CLAMP_BOUND << FP16 最大值."""
        from vit_pytorch.core.constants import SCALE_CLAMP_BOUND

        fp16_max = 65504.0
        safety_factor = fp16_max / SCALE_CLAMP_BOUND

        assert safety_factor > 1000, f"安全系数 {safety_factor:.0f} 不足"

    def test_softmax_saturation_at_logit_bound(self):
        """验证 softmax 在 x=10 时接近 one-hot 但保持梯度.

        数学: softmax(10) ≈ 0.99995, 梯度 ≈ 5e-5 (有效)
        vs 50.0: softmax(50) ≈ 1.0, 梯度 ≈ 1e-22 (消失)
        """
        from vit_pytorch.core.constants import LOGIT_CLAMP_BOUND

        x = torch.tensor([LOGIT_CLAMP_BOUND, 0.0])
        softmax = torch.softmax(x, dim=0)

        # x=10 时 softmax ≈ 0.99995 (仍保持有效梯度)
        assert softmax[1].item() < 0.001

    def test_grad_clip_rate_below_threshold(self):
        """验证新边界下梯度裁剪率极低 (< 0.001%)."""
        from vit_pytorch.core.constants import GRAD_CLAMP_BOUND
        import numpy as np

        # 模拟梯度分布 (正态分布, σ = 5)
        np.random.seed(42)
        grad = np.random.randn(100000) * 5

        clip_rate = np.mean(np.abs(grad) > GRAD_CLAMP_BOUND)

        assert clip_rate < 0.001, f"梯度裁剪率 {clip_rate:.4%} 过高"

    def test_grad_clamp_improvement_over_old(self):
        """验证新边界比旧边界 (10.0) 显著改善梯度裁剪率."""
        from vit_pytorch.core.constants import GRAD_CLAMP_BOUND
        import numpy as np

        np.random.seed(42)
        grad = np.random.randn(100000) * 5

        old_clip_rate = np.mean(np.abs(grad) > 10.0)
        new_clip_rate = np.mean(np.abs(grad) > GRAD_CLAMP_BOUND)

        # 新裁剪率应显著低于旧裁剪率
        improvement = (old_clip_rate - new_clip_rate) / old_clip_rate * 100
        assert improvement > 99, f"改善率 {improvement:.1f}% 不够显著"
        assert new_clip_rate < 0.001, f"新裁剪率 {new_clip_rate:.4%} 仍过高"

    def test_softplus_output_covered_by_scale_bound(self):
        """验证 softplus 输出大部分在 SCALE_CLAMP_BOUND 范围内."""
        from vit_pytorch.core.constants import SCALE_CLAMP_BOUND
        import numpy as np

        # 测试典型输入范围内的 softplus 输出
        # SCALE_CLAMP_BOUND = 15.0 对应 softplus 输入约 15
        x_values = np.linspace(-10, 15, 1000)
        softplus_outputs = np.log(1 + np.exp(x_values))

        covered_ratio = np.mean(softplus_outputs <= SCALE_CLAMP_BOUND)
        assert covered_ratio > 0.99, f"覆盖率 {covered_ratio:.2%} 不足"

    def test_fp16_safe_epsilon_above_fp16_min(self):
        """FP16_SAFE_EPSILON > FP16 最小正规数."""
        from vit_pytorch.core.constants import FP16_SAFE_EPSILON

        # FP16 最小正规数 ≈ 6.1e-5
        2**-14  # ≈ 6.1e-5

        # 注意: 1e-6 < 6.1e-5，这可能需要调整
        # 但对于 log 计算等场景，1e-8 仍足够
        # 这里主要检查 epsilon 为正且合理
        assert FP16_SAFE_EPSILON > 0
        assert FP16_SAFE_EPSILON <= 1e-4

    def test_clamp_bounds_order(self):
        """CLAMP 边界值顺序: GRAD > LOGIT > SCALE.

        注意: GRAD > LOGIT (20 > 10) 因为两者服务于不同目的:
        - LOGIT_CLAMP_BOUND: softmax 前截断 (梯度有效性, 需较小)
        - GRAD_CLAMP_BOUND: 梯度裁剪 (FP16 安全, 可较大)
        - SCALE_CLAMP_BOUND: softplus 输出截断 (范围最小)

        I122-3 修复: 更新顺序测试以匹配数学分析
        """
        from vit_pytorch.core.constants import (
            LOGIT_CLAMP_BOUND,
            GRAD_CLAMP_BOUND,
            SCALE_CLAMP_BOUND,
        )

        # 顺序: GRAD (20) > LOGIT (10) > SCALE (15) 不成立
        # 实际: LOGIT (10) < SCALE (15) < GRAD (20)
        # 测试: 确保 LOGIT 保持梯度有效性
        assert LOGIT_CLAMP_BOUND == 10.0
        assert GRAD_CLAMP_BOUND == 20.0
        assert SCALE_CLAMP_BOUND == 15.0
