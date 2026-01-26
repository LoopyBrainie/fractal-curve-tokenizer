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

import pytest
import torch
import torch.nn.functional as F


class TestNumericalSafetyConstants:
    """数值安全常量验证."""

    def test_all_epsilon_positive(self):
        """所有 epsilon > 0."""
        from vit_pytorch.constants import (
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
        from vit_pytorch.constants import (
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
        from vit_pytorch.constants import TEMPERATURE_MIN

        assert TEMPERATURE_MIN >= 0.1, f"TEMPERATURE_MIN = {TEMPERATURE_MIN} may cause gradient collapse"

    def test_gumbel_epsilon_safe_for_sampling(self):
        """GUMBEL_EPSILON 安全边距验证."""
        from vit_pytorch.constants import GUMBEL_EPSILON

        import math

        eps = GUMBEL_EPSILON
        g_min = -math.log(-math.log(eps))
        g_max = -math.log(-math.log(1 - eps))

        assert g_min >= -20, f"Gumbel range too narrow: g_min = {g_min}"
        assert g_max <= 20, f"Gumbel range too wide: g_max = {g_max}"


class TestTemperatureSchedule:
    """温度调度常量验证."""

    def test_temperature_order(self):
        """SPLITTER_TEMP_START > SPLITTER_TEMP_END > TEMPERATURE_MIN > 0."""
        from vit_pytorch.constants import (
            SPLITTER_TEMP_START,
            SPLITTER_TEMP_END,
            TEMPERATURE_MIN,
        )

        assert SPLITTER_TEMP_START > SPLITTER_TEMP_END
        assert SPLITTER_TEMP_END > TEMPERATURE_MIN
        assert TEMPERATURE_MIN > 0

    def test_temperature_in_exploration_range(self):
        """起始温度在有效探索范围内."""
        from vit_pytorch.constants import SPLITTER_TEMP_START

        assert 0.5 <= SPLITTER_TEMP_START <= 2.0

    def test_temperature_convergence_bound(self):
        """终止温度保证收敛."""
        from vit_pytorch.constants import SPLITTER_TEMP_END

        assert 0.1 <= SPLITTER_TEMP_END <= 1.0


class TestQuotaInitialization:
    """配额初始化常量验证."""

    def test_quota_logits_softmax_sum(self):
        """QUOTA_INIT_LOGITS softmax 后求和为 1."""
        from vit_pytorch.constants import QUOTA_INIT_LOGITS

        logits = torch.tensor(QUOTA_INIT_LOGITS)
        probs = F.softmax(logits, dim=-1)

        sum_probs = probs.sum().item()
        assert abs(sum_probs - 1.0) < 1e-6

    def test_quota_distribution_increases_with_depth(self):
        """配额分布随深度递增 (深度优先策略)."""
        from vit_pytorch.constants import QUOTA_INIT_LOGITS

        logits = torch.tensor(QUOTA_INIT_LOGITS)
        probs = F.softmax(logits, dim=-1)

        for i in range(len(probs) - 1):
            assert probs[i].item() < probs[i + 1].item()

    def test_quota_min_ratio_positive(self):
        """QUOTA_MIN_RATIO > 0."""
        from vit_pytorch.constants import QUOTA_MIN_RATIO

        assert QUOTA_MIN_RATIO > 0
        assert QUOTA_MIN_RATIO < 1.0

    def test_quota_entropy_weight_reasonable(self):
        """QUOTA_ENTROPY_WEIGHT 在合理范围内."""
        from vit_pytorch.constants import QUOTA_ENTROPY_WEIGHT

        assert 0 <= QUOTA_ENTROPY_WEIGHT <= 1.0


class TestCoverageConstants:
    """覆盖率常量验证."""

    def test_coverage_bounds_order(self):
        """0 < K_COVERAGE_MIN < K_COVERAGE_BASE < K_COVERAGE_MAX_HARD <= 1."""
        from vit_pytorch.constants import (
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
        from vit_pytorch.constants import K_ADAPTIVE_REFERENCE_SIZE

        assert K_ADAPTIVE_REFERENCE_SIZE > 0
        assert isinstance(K_ADAPTIVE_REFERENCE_SIZE, int)

    def test_k_hard_limits_order(self):
        """K_MIN_HARD_LIMIT < K_MAX_HARD_LIMIT."""
        from vit_pytorch.constants import (
            K_MIN_HARD_LIMIT,
            K_MAX_HARD_LIMIT,
        )

        assert K_MIN_HARD_LIMIT < K_MAX_HARD_LIMIT
        assert 1 <= K_MIN_HARD_LIMIT <= 64
        assert 1024 <= K_MAX_HARD_LIMIT <= 8192

    def test_sample_ratios_order(self):
        """K_MIN_SAMPLE_RATIO < K_MAX_SAMPLE_RATIO."""
        from vit_pytorch.constants import (
            K_MIN_SAMPLE_RATIO,
            K_MAX_SAMPLE_RATIO,
        )

        assert K_MIN_SAMPLE_RATIO < K_MAX_SAMPLE_RATIO

    def test_coverage_adaptive_formula_works(self):
        """覆盖率自适应公式数学正确性."""
        from vit_pytorch.constants import (
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
        from vit_pytorch.constants import (
            ELASTIC_COVERAGE_MAX,
            ELASTIC_COVERAGE_MIN,
            K_COVERAGE_MAX_HARD,
        )

        assert ELASTIC_COVERAGE_MAX == K_COVERAGE_MAX_HARD
        assert ELASTIC_COVERAGE_MIN >= 0.01

    def test_elastic_lambda_weights_reasonable(self):
        """ELASTIC_LAMBDA weights 在合理范围内."""
        from vit_pytorch.constants import (
            ELASTIC_LAMBDA_OVER,
            ELASTIC_LAMBDA_COLLAPSE,
        )

        assert 0.01 <= ELASTIC_LAMBDA_OVER <= 1.0
        assert ELASTIC_LAMBDA_COLLAPSE >= 1.0


class TestThresholdRegularization:
    """阈值正则化常量验证."""

    def test_threshold_var_reg_weight_reasonable(self):
        """THRESHOLD_VAR_REG_WEIGHT 在 [0, 1] 范围内."""
        from vit_pytorch.constants import THRESHOLD_VAR_REG_WEIGHT

        assert 0 <= THRESHOLD_VAR_REG_WEIGHT <= 1.0


class TestDepthVarianceNormalization:
    """深度方差归一化常量验证."""

    def test_ema_alpha_reasonable(self):
        """DEPTH_EMA_ALPHA 在有效范围内."""
        from vit_pytorch.constants import DEPTH_EMA_ALPHA

        assert 0.01 <= DEPTH_EMA_ALPHA <= 0.5

    def test_depth_variance_init_eps_reasonable(self):
        """DEPTH_VARIANCE_INIT_EPS 为保守初始化下界."""
        from vit_pytorch.constants import DEPTH_VARIANCE_INIT_EPS

        assert 0.01 <= DEPTH_VARIANCE_INIT_EPS <= 1.0


class TestAttentionBiasScales:
    """注意力偏置缩放常量验证."""

    def test_hilbert_bias_scale_reasonable(self):
        """HILBERT_BIAS_SCALE 在 [0.01, 1.0] 范围内."""
        from vit_pytorch.constants import HILBERT_BIAS_SCALE

        assert 0.01 <= HILBERT_BIAS_SCALE <= 1.0

    def test_level_bias_scale_reasonable(self):
        """LEVEL_BIAS_SCALE 在 [0.01, 1.0] 范围内."""
        from vit_pytorch.constants import (
            LEVEL_BIAS_SCALE,
            HILBERT_BIAS_SCALE,
        )

        assert 0.01 <= LEVEL_BIAS_SCALE <= 1.0
        assert LEVEL_BIAS_SCALE <= HILBERT_BIAS_SCALE


class TestOverlapPenalty:
    """重叠惩罚常量验证."""

    def test_overlap_penalty_weight_reasonable(self):
        """OVERLAP_PENALTY_WEIGHT 在 [0, 1] 范围内."""
        from vit_pytorch.constants import OVERLAP_PENALTY_WEIGHT

        assert 0 <= OVERLAP_PENALTY_WEIGHT <= 1.0


class TestSoftExclusionMargin:
    """软排除边距常量验证."""

    def test_soft_exclusion_margin_reasonable(self):
        """SOFT_EXCLUSION_MARGIN 在 (0, 1) 范围内."""
        from vit_pytorch.constants import SOFT_EXCLUSION_MARGIN

        assert 0 < SOFT_EXCLUSION_MARGIN < 1
        assert SOFT_EXCLUSION_MARGIN <= 0.5
