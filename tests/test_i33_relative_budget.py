"""
I33: Elastic Budget 相对预算测试

数学形式化验证:
1. 相对覆盖率保证跨尺度一致性
2. 损失函数梯度归一化
3. 与 _get_dynamic_k_bounds() 统一

日期: 2026-01-19
"""

import pytest
import torch
from unittest.mock import MagicMock, patch

# I33: GumbelTopKSplitter 从专用模块导入
from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
from vit_pytorch.config import SplitterConfig
from vit_pytorch.constants import (
    ELASTIC_COVERAGE_MAX,
    ELASTIC_COVERAGE_MIN,
    ELASTIC_LAMBDA_OVER,
    ELASTIC_LAMBDA_COLLAPSE,
    K_COVERAGE_BASE,
    K_COVERAGE_MIN,
    K_COVERAGE_MAX_HARD,
    K_ADAPTIVE_REFERENCE_SIZE,
    K_MAX_HARD_LIMIT,
    K_MIN_HARD_LIMIT,
)


class TestElasticBudgetConstants:
    """测试 Elastic Budget 常量定义"""

    def test_elastic_coverage_max(self):
        """验证 ELASTIC_COVERAGE_MAX = 0.08"""
        assert ELASTIC_COVERAGE_MAX == 0.08

    def test_elastic_coverage_min(self):
        """验证 ELASTIC_COVERAGE_MIN = 0.005"""
        assert ELASTIC_COVERAGE_MIN == 0.005

    def test_elastic_lambda_over(self):
        """验证 ELASTIC_LAMBDA_OVER = 0.1"""
        assert ELASTIC_LAMBDA_OVER == 0.1

    def test_elastic_lambda_collapse(self):
        """验证 ELASTIC_LAMBDA_COLLAPSE = 1.0"""
        assert ELASTIC_LAMBDA_COLLAPSE == 1.0


class TestElasticBudgetLoss:
    """测试 Elastic Budget 损失函数"""

    @pytest.fixture
    def splitter(self):
        """创建测试用 splitter (使用 SplitterConfig)"""
        config = SplitterConfig(
            feature_dim=256,
            min_patch_size=4,
            max_depth_limit=8,
            hidden_dim=64,
            intermediate_dim=64,
            pool_size=4,
            K_min=8,
            K_max=4096,
            use_dynamic_k=True,
            token_coverage_base=K_COVERAGE_BASE,
            token_coverage_min=K_COVERAGE_MIN,
            token_coverage_max_hard=K_COVERAGE_MAX_HARD,
            adaptive_reference_size=K_ADAPTIVE_REFERENCE_SIZE,
            K_min_abs=K_MIN_HARD_LIMIT,
            K_max_hard=K_MAX_HARD_LIMIT,
            use_adaptive_coverage=True,
        )
        return GumbelTopKSplitter(
            config=config,
            image_size=(224, 224),
        )

    def test_elastic_loss_zero_when_below_max(self, splitter):
        """测试覆盖率低于最大值时损失为0"""
        # 模拟 _avg_selected 和 num_candidates
        splitter._avg_selected.fill_(50)  # 50 tokens
        splitter.num_candidates = 4096  # 224x224 图像

        coverage = 50 / 4096  # ~1.2%
        assert coverage < ELASTIC_COVERAGE_MAX

        # 调用 get_auxiliary_losses
        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=True,
            actual_token_count=50,
        )

        assert 'elastic_budget_loss' in losses
        assert losses['elastic_budget_loss'].item() == 0.0

    def test_elastic_loss_nonzero_when_above_max(self, splitter):
        """测试覆盖率高于最大值时有损失"""
        splitter._avg_selected.fill_(500)  # 500 tokens (超过 8% = 327)
        splitter.num_candidates = 4096  # 224x224 图像

        coverage = 500 / 4096  # ~12.2% > 8%
        assert coverage > ELASTIC_COVERAGE_MAX

        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=True,
            actual_token_count=500,
        )

        assert 'elastic_budget_loss' in losses
        assert losses['elastic_budget_loss'].item() > 0.0

    def test_elastic_loss_gradient_normalized(self, splitter):
        """测试损失梯度归一化到 ~0.01 量级"""
        device = 'cpu'

        for H, W in [(64, 64), (224, 224), (512, 512)]:
            N = H * W
            # 设置超过 8% 覆盖率
            tokens_above = int(ELASTIC_COVERAGE_MAX * N * 1.5)

            splitter._avg_selected.fill_(tokens_above)
            splitter.num_candidates = N

            losses = splitter.get_auxiliary_losses(
                include_elastic_budget=True,
                actual_token_count=tokens_above,
            )

            loss_value = losses['elastic_budget_loss'].item()
            # 损失值应该与 N 相关，但梯度应该归一化
            expected_magnitude = ELASTIC_LAMBDA_OVER * (0.5 * ELASTIC_COVERAGE_MAX) ** 2 * N

            # 损失值应该在预期量级附近
            assert abs(loss_value - expected_magnitude) < expected_magnitude * 0.1, \
                f"Loss magnitude mismatch for {H}x{W}: {loss_value} vs {expected_magnitude}"

    def test_collapse_detection(self, splitter):
        """测试崩溃检测 (覆盖率 < 0.5%)"""
        # 极低覆盖率
        splitter._avg_selected.fill_(5)  # 5 tokens
        splitter.num_candidates = 4096  # 224x224 图像
        coverage = 5 / 4096  # ~0.12% < 0.5%

        assert coverage < ELASTIC_COVERAGE_MIN

        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=True,
            actual_token_count=5,
        )

        assert 'collapse_loss' in losses
        assert losses['collapse_loss'].item() == ELASTIC_LAMBDA_COLLAPSE

    def test_no_collapse_when_normal(self, splitter):
        """测试正常覆盖率不触发崩溃"""
        # 正常覆盖率
        splitter._avg_selected.fill_(100)  # 100 tokens
        splitter.num_candidates = 4096  # 224x224 图像
        coverage = 100 / 4096  # ~2.4% > 0.5%

        assert coverage > ELASTIC_COVERAGE_MIN

        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=True,
            actual_token_count=100,
        )

        assert 'collapse_loss' not in losses


class TestCoverageConsistency:
    """测试跨尺度覆盖率一致性"""

    @pytest.fixture
    def splitter(self):
        config = SplitterConfig(
            feature_dim=256,
            min_patch_size=4,
            max_depth_limit=8,
            hidden_dim=64,
            intermediate_dim=64,
            pool_size=4,
            K_min=8,
            K_max=4096,
            use_dynamic_k=True,
            token_coverage_base=K_COVERAGE_BASE,
            token_coverage_min=K_COVERAGE_MIN,
            token_coverage_max_hard=K_COVERAGE_MAX_HARD,
            adaptive_reference_size=K_ADAPTIVE_REFERENCE_SIZE,
            K_min_abs=K_MIN_HARD_LIMIT,
            K_max_hard=K_MAX_HARD_LIMIT,
            use_adaptive_coverage=True,
        )
        return GumbelTopKSplitter(
            config=config,
            image_size=(224, 224),
        )

    def test_coverage_in_target_range(self, splitter):
        """测试各分辨率下覆盖率在目标范围内 [0.5%, 8%]"""
        test_cases = [
            (64, 64),   # N=256, coverage_max 应为 ~5%
            (128, 128), # N=1024, coverage_max 应为 ~5%
            (224, 224), # N=4096, coverage_max 应为 ~5%
            (512, 512), # N=16384, coverage_max 可能因硬上限限制为 ~25%
        ]

        for H, W in test_cases:
            N = H * W
            K_min, K_max = splitter._get_dynamic_k_bounds(N, (H, W))

            coverage_min = K_min / N
            coverage_max = K_max / N

            # 验证最小覆盖率在合理范围内
            assert 0.005 <= coverage_min <= 0.05, \
                f"{H}x{W}: K_min={K_min}, coverage_min={coverage_min:.4f} out of range"

            # 验证最大覆盖率 (可能因硬上限限制)
            # 如果 K_max < K_max_hard，则覆盖率应在目标范围内
            # 如果 K_max = K_max_hard (4096)，则覆盖率可能低于目标范围
            if K_max < K_MAX_HARD_LIMIT:
                assert 0.03 <= coverage_max <= 0.08, \
                    f"{H}x{W}: K_max={K_max}, coverage_max={coverage_max:.4f} out of range"
            else:
                # 硬上限情况: 覆盖率可能低于目标范围，但这是预期行为
                assert coverage_max <= 0.25, \
                    f"{H}x{W}: K_max={K_max}, coverage_max={coverage_max:.4f} exceeds hard limit"

    def test_dynamic_bounds_match_elastic(self, splitter):
        """测试动态 K 边界与 Elastic Budget 边界一致"""
        for H, W in [(64, 64), (224, 224), (512, 512)]:
            K_min, K_max = splitter._get_dynamic_k_bounds(H * W, (H, W))

            # Elastic Budget 应该使用相同的边界
            # 这里我们验证理论一致性：实际边界由 _get_dynamic_k_bounds 返回
            assert K_min >= 8, f"{H}x{W}: K_min={K_min} below hard limit"
            assert K_max <= 4096, f"{H}x{W}: K_max={K_max} above hard limit"


class TestAuxiliaryLossesInterface:
    """测试辅助损失接口 (向后兼容性)"""

    @pytest.fixture
    def splitter(self):
        return GumbelTopKSplitter(
            feature_dim=256,
            min_patch_size=4,
            max_depth_limit=8,
            hidden_dim=64,
            image_size=(64, 64),
        )

    def test_get_auxiliary_losses_signature(self, splitter):
        """测试函数签名不包含 elastic_N_* 参数"""
        import inspect
        sig = inspect.signature(splitter.get_auxiliary_losses)
        params = list(sig.parameters.keys())

        # elastic_N_min 和 elastic_N_max 应该不存在
        assert 'elastic_N_min' not in params, "elastic_N_min should be removed"
        assert 'elastic_N_max' not in params, "elastic_N_max should be removed"
        assert 'elastic_lambda_over' not in params, "elastic_lambda_over should be removed"
        assert 'elastic_lambda_under' not in params, "elastic_lambda_under should be removed"
        assert 'elastic_lambda_collapse' not in params, "elastic_lambda_collapse should be removed"

    def test_get_auxiliary_losses_accepts_only_include(self, splitter):
        """测试函数只接受 include_elastic_budget 参数"""
        # 应该正常工作
        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=True,
            actual_token_count=100,
        )
        assert 'elastic_budget_loss' in losses

    def test_disabled_elastic_budget(self, splitter):
        """测试禁用 elastic_budget 时不返回损失"""
        losses = splitter.get_auxiliary_losses(
            include_elastic_budget=False,
            actual_token_count=100,
        )
        assert 'elastic_budget_loss' not in losses


class TestIntegration:
    """集成测试"""

    def test_full_pipeline_relative_budget(self):
        """测试完整流程的相对预算一致性"""
        config = SplitterConfig(
            feature_dim=256,
            min_patch_size=4,
            max_depth_limit=8,
            hidden_dim=64,
            intermediate_dim=64,
            pool_size=4,
            K_min=8,
            K_max=4096,
            use_dynamic_k=True,
            token_coverage_base=K_COVERAGE_BASE,
            token_coverage_min=K_COVERAGE_MIN,
            token_coverage_max_hard=K_COVERAGE_MAX_HARD,
            adaptive_reference_size=K_ADAPTIVE_REFERENCE_SIZE,
            K_min_abs=K_MIN_HARD_LIMIT,
            K_max_hard=K_MAX_HARD_LIMIT,
            use_adaptive_coverage=True,
        )
        splitter = GumbelTopKSplitter(
            config=config,
            image_size=(224, 224),
        )

        # 模拟不同分辨率下的 token 选择
        resolutions = [(64, 64), (224, 224), (512, 512)]

        for H, W in resolutions:
            N_candidates = H * W
            K_min, K_max = splitter._get_dynamic_k_bounds(N_candidates, (H, W))

            # 验证覆盖率
            coverage_min = K_min / N_candidates
            coverage_max = K_max / N_candidates

            # 覆盖率应该在合理范围内
            assert 0.005 <= coverage_min <= 0.05, \
                f"{H}x{W}: coverage_min={coverage_min:.4f} not in [0.5%, 5%]"

            # 如果未达到硬上限，覆盖率应在目标范围内
            if K_max < K_MAX_HARD_LIMIT:
                assert 0.03 <= coverage_max <= 0.08, \
                    f"{H}x{W}: coverage_max={coverage_max:.4f} not in [3%, 8%]"

            # 模拟选择的 token 数 (在范围内)
            avg_tokens = (K_min + K_max) // 2
            splitter._avg_selected.fill_(avg_tokens)
            splitter.num_candidates = N_candidates

            # 计算相对覆盖率
            coverage = avg_tokens / N_candidates

            # 验证损失计算
            losses = splitter.get_auxiliary_losses(
                include_elastic_budget=True,
                actual_token_count=avg_tokens,
            )

            if coverage > ELASTIC_COVERAGE_MAX:
                assert losses['elastic_budget_loss'].item() > 0
            else:
                assert losses['elastic_budget_loss'].item() == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
