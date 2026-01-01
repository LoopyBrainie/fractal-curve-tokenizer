"""
P10-14: Splitter 健康监控机制单元测试

验证 check_splitter_health() 函数正确检测各种异常状态
"""

import pytest
import sys
from pathlib import Path

# 添加 examples/training 到路径
TRAINING_PATH = Path(__file__).parent.parent.parent / "examples" / "training"
sys.path.insert(0, str(TRAINING_PATH))

from train_fractal_vit import (
    SplitterHealthConfig,
    SplitterHealthStatus,
    check_splitter_health,
)


class TestSplitterHealthCheck:
    """分割器健康检查测试"""
    
    def test_healthy_splitter(self):
        """测试正常分割器被正确识别"""
        status = check_splitter_health(
            avg_tokens=10.0,
            entropy=0.8,
            max_entropy=1.609,
            epoch=5,
        )
        
        assert status.is_healthy is True
        assert status.severity == 'ok'
        assert status.collapse_detected is False
        assert status.monotone_detected is False
        assert status.saturate_detected is False
        assert status.health_score > 0.9
    
    def test_collapse_detection(self):
        """测试坍缩检测: avg_tokens < 2"""
        status = check_splitter_health(
            avg_tokens=1.5,
            entropy=0.5,
            max_entropy=1.609,
            epoch=5,
        )
        
        assert status.is_healthy is False
        assert status.collapse_detected is True
        assert status.severity == 'critical'
        assert 'COLLAPSE' in status.message
    
    def test_monotone_detection(self):
        """测试单调检测: entropy_ratio < 10%"""
        status = check_splitter_health(
            avg_tokens=10.0,
            entropy=0.05,  # ratio = 0.05/1.609 ≈ 3%
            max_entropy=1.609,
            epoch=5,
        )
        
        assert status.is_healthy is False
        assert status.monotone_detected is True
        assert status.severity == 'warning'
        assert 'Monotone' in status.message
    
    def test_saturation_detection(self):
        """测试饱和检测: avg_tokens > 90% of max"""
        status = check_splitter_health(
            avg_tokens=60.0,  # > 0.9 * 64
            entropy=1.0,
            max_entropy=1.609,
            epoch=5,
        )
        
        assert status.is_healthy is False
        assert status.saturate_detected is True
        assert status.severity == 'warning'
        assert 'Saturation' in status.message
    
    def test_warmup_period_no_warning(self):
        """测试宽限期内不报警"""
        # 即使 tokens 很低，宽限期内也不应报警
        status = check_splitter_health(
            avg_tokens=1.0,  # 非常低
            entropy=0.05,    # 非常低
            max_entropy=1.609,
            epoch=2,  # 在宽限期内 (默认 3 epochs)
        )
        
        assert status.is_healthy is True
        assert status.severity == 'ok'
        assert 'warmup' in status.message.lower()
    
    def test_no_data_returns_healthy(self):
        """测试无数据时返回健康状态"""
        status = check_splitter_health(
            avg_tokens=None,
            entropy=None,
            max_entropy=None,
            epoch=5,
        )
        
        assert status.is_healthy is True
        assert status.health_score == 1.0
    
    def test_health_score_calculation(self):
        """测试健康评分在 [0, 1] 范围内"""
        # 完全健康
        status1 = check_splitter_health(10.0, 0.8, 1.609, epoch=5)
        assert 0.9 <= status1.health_score <= 1.0
        
        # 部分健康 (接近坍缩)
        status2 = check_splitter_health(2.5, 0.8, 1.609, epoch=5)
        assert 0.5 <= status2.health_score <= 1.0
        
        # 不健康 (坍缩)
        status3 = check_splitter_health(1.0, 0.8, 1.609, epoch=5)
        assert 0.0 <= status3.health_score < 0.9
    
    def test_custom_config(self):
        """测试自定义配置"""
        config = SplitterHealthConfig(
            min_tokens_threshold=5.0,  # 更严格
            min_entropy_ratio=0.2,
            warmup_epochs=1,
        )
        
        # 使用默认配置是健康的
        status1 = check_splitter_health(3.0, 0.3, 1.609, epoch=3)
        assert status1.is_healthy is True
        
        # 使用严格配置是不健康的
        status2 = check_splitter_health(3.0, 0.3, 1.609, epoch=3, config=config)
        assert status2.is_healthy is False
        assert status2.collapse_detected is True  # 3 < 5


class TestHealthStatusDataclass:
    """测试健康状态数据类"""
    
    def test_dataclass_fields(self):
        """测试数据类包含所有必要字段"""
        status = SplitterHealthStatus(
            is_healthy=True,
            collapse_detected=False,
            monotone_detected=False,
            saturate_detected=False,
            health_score=1.0,
            severity='ok',
            message='test'
        )
        
        assert hasattr(status, 'is_healthy')
        assert hasattr(status, 'collapse_detected')
        assert hasattr(status, 'monotone_detected')
        assert hasattr(status, 'saturate_detected')
        assert hasattr(status, 'health_score')
        assert hasattr(status, 'severity')
        assert hasattr(status, 'message')


class TestHealthConfigDataclass:
    """测试健康配置数据类"""
    
    def test_default_values(self):
        """测试默认值合理"""
        config = SplitterHealthConfig()
        
        assert config.min_tokens_threshold == 2.0
        assert config.min_entropy_ratio == 0.1
        assert config.saturation_ratio == 0.9
        assert config.warmup_epochs == 3
        assert config.expected_tokens_max == 64.0
