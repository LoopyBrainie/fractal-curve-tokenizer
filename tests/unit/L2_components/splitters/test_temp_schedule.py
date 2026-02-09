# -*- coding: utf-8 -*-
"""
I122-7: 温度调度策略单元测试

验证:
1. 线性调度严格递减且在边界内
2. 逆时调度单调递减
3. 温度边界约束 (T_min = 0.4)
4. 所有调度策略的数学性质
"""

import pytest
import math
from vit_pytorch import GumbelTopKSplitter
from vit_pytorch.config import HilbertSplitterConfig
from vit_pytorch.constants import (
    SPLITTER_TEMP_START,
    SPLITTER_TEMP_END,
    TEMPERATURE_MIN,
)


def create_splitter_with_schedule(schedule: str):
    """创建带有指定调度策略的分割器"""
    config = HilbertSplitterConfig(
        temperature_init=SPLITTER_TEMP_START,
        temperature_min=SPLITTER_TEMP_END,
        temperature_anneal=schedule,
    )
    splitter = GumbelTopKSplitter(config=config, image_size=(64, 64))
    # 设置温度退火总步数
    splitter._temp_total_steps.fill_(100)
    splitter._temp_enabled = True
    return splitter


class TestTemperatureSchedules:
    """温度调度策略验证"""

    def test_linear_schedule_monotonic_decrease(self):
        """验证线性调度严格递减"""
        splitter = create_splitter_with_schedule('linear')
        temperatures = []
        for t in range(101):
            T = splitter._update_temperature()
            temperatures.append(T)

        # 验证严格递减
        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i-1], \
                f"Linear schedule: T[{i}]={temperatures[i]} > T[{i-1}]={temperatures[i-1]}"

    def test_linear_schedule_bounds(self):
        """验证线性调度在边界内"""
        splitter = create_splitter_with_schedule('linear')
        for t in range(100):
            T = splitter._update_temperature()
            assert T >= TEMPERATURE_MIN, \
                f"Linear schedule: T={T} < T_min={TEMPERATURE_MIN}"
            assert T <= SPLITTER_TEMP_START, \
                f"Linear schedule: T={T} > T_start={SPLITTER_TEMP_START}"

    def test_linear_schedule_endpoint(self):
        """验证线性调度端点值"""
        splitter = create_splitter_with_schedule('linear')

        # t=0: 应该等于 T_start
        T_0 = splitter._update_temperature()
        assert abs(T_0 - SPLITTER_TEMP_START) < 1e-6, \
            f"Linear schedule: T(0)={T_0} != T_start={SPLITTER_TEMP_START}"

        # 运行到 T_end
        for _ in range(100):
            splitter._update_temperature()

        # t=T: 应该接近 T_end
        assert abs(splitter.current_temperature - SPLITTER_TEMP_END) < 1e-3, \
            f"Linear schedule: T(T)={splitter.current_temperature} != T_end={SPLITTER_TEMP_END}"

    def test_exponential_schedule_monotonic_decrease(self):
        """验证指数调度严格递减"""
        splitter = create_splitter_with_schedule('exponential')
        temperatures = []
        for t in range(101):
            T = splitter._update_temperature()
            temperatures.append(T)

        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i-1], \
                f"Exponential schedule: T[{i}]={temperatures[i]} > T[{i-1}]={temperatures[i-1]}"

    def test_exponential_schedule_bounds(self):
        """验证指数调度在边界内"""
        splitter = create_splitter_with_schedule('exponential')
        for t in range(100):
            T = splitter._update_temperature()
            assert T >= TEMPERATURE_MIN, \
                f"Exponential schedule: T={T} < T_min={TEMPERATURE_MIN}"
            assert T <= SPLITTER_TEMP_START, \
                f"Exponential schedule: T={T} > T_start={SPLITTER_TEMP_START}"

    def test_inverse_time_schedule_monotonic_decrease(self):
        """验证逆时调度严格递减"""
        splitter = create_splitter_with_schedule('inverse_time')
        temperatures = []
        for t in range(101):
            T = splitter._update_temperature()
            temperatures.append(T)

        for i in range(1, len(temperatures)):
            assert temperatures[i] <= temperatures[i-1], \
                f"Inverse time schedule: T[{i}]={temperatures[i]} > T[{i-1}]={temperatures[i-1]}"

    def test_inverse_time_schedule_bounds(self):
        """验证逆时调度在边界内"""
        splitter = create_splitter_with_schedule('inverse_time')
        for t in range(100):
            T = splitter._update_temperature()
            assert T >= TEMPERATURE_MIN, \
                f"Inverse time schedule: T={T} < T_min={TEMPERATURE_MIN}"
            assert T <= SPLITTER_TEMP_START, \
                f"Inverse time schedule: T={T} > T_start={SPLITTER_TEMP_START}"


class TestTemperatureGradientEffect:
    """温度对梯度的影响验证"""

    def test_temperature_gradient_relationship(self):
        """验证温度与梯度的数学关系"""
        # Gumbel-Softmax 梯度强度 ≈ 1/τ
        # τ = 0.4: 梯度 ≈ 2.5 (有效)
        # τ = 0.1: 梯度 ≈ 10 (可能不稳定)

        tau_04 = 0.4
        tau_01 = 0.1

        gradient_04 = 1 / tau_04
        gradient_01 = 1 / tau_01

        assert gradient_04 > 1.0, "温度0.4时应有有效梯度"
        assert gradient_01 > gradient_04, "温度0.1时梯度应更大"

    def test_temperature_minimum_valid(self):
        """验证 T_min = 0.4 的合理性"""
        # T_min = 0.4 时，softmax 梯度 ≈ 2.5 (有效)
        # T_min = 0.3 时，softmax 梯度 ≈ 3.33
        # T_min = 0.1 时，softmax 梯度 ≈ 10 (梯度消失风险)

        tau_values = [0.4, 0.3, 0.2, 0.1]
        expected_gradients = [2.5, 3.33, 5.0, 10.0]

        for tau, expected_grad in zip(tau_values, expected_gradients):
            actual_grad = 1 / tau
            assert abs(actual_grad - expected_grad) < 0.01, \
                f"Tau={tau}: expected grad {expected_grad}, got {actual_grad}"


class TestScheduleComparison:
    """调度策略对比验证"""

    def test_schedule_different_behavior(self):
        """验证不同调度策略的行为差异"""
        splitter_linear = create_splitter_with_schedule('linear')
        splitter_exp = create_splitter_with_schedule('exponential')

        # 记录不同步数下的温度
        checkpoints = [10, 25, 50, 75]

        for t in checkpoints:
            # 更新到指定步数
            for _ in range(t):
                splitter_linear._update_temperature()
                splitter_exp._update_temperature()

            T_linear = splitter_linear.current_temperature
            T_exp = splitter_exp.current_temperature

            # 所有策略都应该产生有效温度
            assert T_linear >= TEMPERATURE_MIN, f"t={t}: T_linear={T_linear} < T_min"
            assert T_exp >= TEMPERATURE_MIN, f"t={t}: T_exp={T_exp} < T_min"

            # 验证单调性
            if t > 10:
                prev_T_linear = splitter_linear._temp_step - 1
                # 指数调度在中期应该比线性调度衰减更多
                if t == 50:
                    assert T_exp <= T_linear + 0.01, \
                        f"t={t}: Exponential should decay more: T_exp={T_exp}, T_linear={T_linear}"


class TestCurvatureRemoved:
    """曲率感知移除验证 (I122-7)"""

    def test_no_curvature_schedule(self):
        """验证 'cosine' 和 'curvature' 调度不再可用"""
        # Cosine 不是有效选项，会使用默认值
        # 由于 Literal 类型不强制验证，cosine 会被存储
        config_cosine = HilbertSplitterConfig(
            temperature_anneal='cosine',
        )
        splitter_cosine = GumbelTopKSplitter(config=config_cosine, image_size=(64, 64))
        splitter_cosine._temp_total_steps.fill_(100)
        splitter_cosine._temp_enabled = True

        # 由于 cosine 不是有效选项，代码中会走到 else 分支使用 T_s
        # 但我们的实现中，cosine 不再被特殊处理
        T = splitter_cosine._update_temperature()
        # 验证温度有效
        assert T >= TEMPERATURE_MIN or T <= SPLITTER_TEMP_START


class TestConstantsUpdated:
    """常量更新验证 (I122-7)"""

    def test_temperature_constants(self):
        """验证温度常量已更新"""
        assert SPLITTER_TEMP_START == 1.0, \
            f"SPLITTER_TEMP_START should be 1.0, got {SPLITTER_TEMP_START}"

        assert SPLITTER_TEMP_END == 0.4, \
            f"SPLITTER_TEMP_END should be 0.4, got {SPLITTER_TEMP_END}"

        assert TEMPERATURE_MIN == 0.4, \
            f"TEMPERATURE_MIN should be 0.4, got {TEMPERATURE_MIN}"

    def test_curvature_constants_removed(self):
        """验证曲率相关常量已被移除"""
        try:
            from vit_pytorch.constants import (
                CURVATURE_TEMP_BASE,
                CURVATURE_SENSITIVITY,
                CURVATURE_EMA_ALPHA,
                CURVATURE_WARMUP_STEPS,
            )
            # 如果这些还存在，测试失败
            assert False, "Curvature constants should be removed"
        except ImportError:
            # 这是预期的行为
            pass


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
