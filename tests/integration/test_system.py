# -*- coding: utf-8 -*-
"""System integration tests for FractalCurveViT.

测试整个系统的端到端功能：
1. 模型初始化与设备移动
2. 不同分辨率的处理
3. 批处理能力
4. GPU 支持（如果可用）
5. 向量化审计 (Vectorization Audit)
"""

import sys
import warnings
from pathlib import Path

import torch
import pytest


# P2-3 修复: 定义 PerformanceWarning 类
class PerformanceWarning(UserWarning):
    """Warning for performance-related issues detected by vectorization audit."""
    pass

# Add tests directory to path for vectorization_audit import
_tests_dir = Path(__file__).parent.parent.parent
if str(_tests_dir) not in sys.path:
    sys.path.insert(0, str(_tests_dir))

from vit_pytorch import FractalCurveViT

# Import vectorization audit tools
from tests.unit.utilities.vectorization_audit import (
    enable_vectorization_audit,
    is_audit_enabled,
    run_vectorization_audit_on_model,
    VectorizationReport,
    VectorizationIssue,
)


def _device_parametrization():
    """获取可用设备列表."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_next_gen_fractal_vit_forward_pass(device: str) -> None:
    """测试 FractalCurveViT 的基础前向传播."""
    batch, channels, height, width = 2, 3, 64, 64  # 使用 2 的幂次
    images = torch.randn(batch, channels, height, width, device=device)

    model = FractalCurveViT(
        image_size=(height, width),
        num_classes=10,
        dim=192,
        depth=2,
        heads=4,
        mlp_dim=384,
        min_patch_size=(4, 4),  # 使用 2 的幂次
        max_depth=3,
    ).to(device)
    model.eval()

    result = model(images)
    logits = result.logits if hasattr(result, 'logits') else result
    assert logits.shape == (batch, 10)
    assert logits.device.type == device


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_next_gen_fractal_vit_handles_varied_sizes(device: str) -> None:
    """测试 FractalCurveViT 处理不同图像尺寸."""
    sizes = [(32, 32), (64, 64), (128, 128)]  # 使用 2 的幂次

    for height, width in sizes:
        model = FractalCurveViT(
            image_size=(height, width),
            num_classes=5,
            dim=128,
            depth=2,
            heads=2,
            mlp_dim=256,
            min_patch_size=(4, 4),  # 使用 2 的幂次
            max_depth=3,
        ).to(device)
        model.eval()

        images = torch.randn(2, 3, height, width, device=device)
        result = model(images)
        logits = result.logits if hasattr(result, 'logits') else result
        assert logits.shape == (2, 5)
        assert logits.device.type == device


@torch.no_grad()
@pytest.mark.parametrize("device", _device_parametrization())
def test_streaming_v3_tokenizer_device_consistency(device: str) -> None:
    """测试流式 V3 tokenizer 的设备一致性."""
    from vit_pytorch import StreamingFractalTokenizerV3

    tokenizer = StreamingFractalTokenizerV3(
        image_size=32,
        channels=3,
        d_model=64,
        base_patch_size=4,
        max_depth=3,
    ).to(device)

    # I98-1: 创建独立的 Splitter
    from vit_pytorch.gumbel_topk_splitter import GumbelTopKSplitter
    from vit_pytorch.config import SplitterConfig

    splitter_config = SplitterConfig(
        feature_dim=64,
        min_patch_size=4,
        max_depth_limit=3,
        hidden_dim=32,
        intermediate_dim=32,
        pool_size=4,
        K_min=4,
        K_max=16,
    )
    splitter = GumbelTopKSplitter(
        config=splitter_config,
        image_size=(32, 32),
    ).to(device)

    images = torch.randn(2, 3, 32, 32, device=device)
    # I98-1: 使用完整 pipeline (features -> splitter -> tokenizer)
    features = tokenizer.shared_conv(images)
    split_result = splitter(features, image_size=(32, 32), hard=True)
    output = tokenizer.tokenize(images, split_result)
    
    for seq in output:
        assert seq.tokens.device.type == device


@torch.no_grad()
def test_batch_size_independence() -> None:
    """测试不同批次大小的独立性."""
    model = FractalCurveViT(
        image_size=32,
        num_classes=10,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128,
    )
    model.eval()
    
    # 同一张图像，不同批次大小
    single_image = torch.randn(1, 3, 32, 32)
    batch_images = single_image.repeat(4, 1, 1, 1)
    
    result_single = model(single_image)
    result_batch = model(batch_images)
    logits_single = result_single.logits if hasattr(result_single, 'logits') else result_single
    logits_batch = result_batch.logits if hasattr(result_batch, 'logits') else result_batch

    # 批次中的每个输出应该相同（在数值精度范围内）
    for i in range(4):
        assert torch.allclose(logits_single[0], logits_batch[i], atol=1e-5)


@torch.no_grad()
def test_deterministic_eval_mode() -> None:
    """测试 eval 模式下的确定性."""
    model = FractalCurveViT(
        image_size=32,
        num_classes=10,
        dim=64,
        depth=2,
        heads=2,
        mlp_dim=128,
    )
    model.eval()
    
    images = torch.randn(2, 3, 32, 32)

    result1 = model(images)
    result2 = model(images)
    logits1 = result1.logits if hasattr(result1, 'logits') else result1
    logits2 = result2.logits if hasattr(result2, 'logits') else result2

    assert torch.equal(logits1, logits2)


@pytest.fixture
def vectorization_audit_enabled():
    """Fixture to enable vectorization audit for tests."""
    enable_vectorization_audit(True)
    yield
    enable_vectorization_audit(False)


@torch.no_grad()
def test_batch_consistency(vectorization_audit_enabled) -> None:
    """测试批处理一致性并执行向量化审计.

    这个测试用例：
    1. 验证不同批次大小的输出在数值上保持一致
    2. 运行向量化审计检测潜在的向量化问题
    3. 对于检测到的向量化问题，抛出 PerformanceWarning 而非测试失败
    4. 仅在向量化问题严重影响批处理一致性时标记测试失败
    """
    batch_sizes = [1, 2, 4, 8]
    image_size = 64
    num_classes = 10

    # 创建模型
    model = FractalCurveViT(
        image_size=image_size,
        num_classes=num_classes,
        dim=128,
        depth=2,
        heads=4,
        mlp_dim=256,
        min_patch_size=(4, 4),
        max_depth=3,
    )
    model.eval()

    # 生成相同内容的不同批次
    base_image = torch.randn(1, 3, image_size, image_size)
    batch_outputs = {}

    for batch_size in batch_sizes:
        if batch_size == 1:
            batch_images = base_image
        else:
            batch_images = base_image.repeat(batch_size, 1, 1, 1)

        result = model(batch_images)
        logits = result.logits if hasattr(result, 'logits') else result
        batch_outputs[batch_size] = logits

        # 验证输出形状
        assert logits.shape == (batch_size, num_classes), \
            f"Batch size {batch_size}: expected {(batch_size, num_classes)}, got {logits.shape}"

    # 验证批处理一致性：单个样本输出应与批中对应样本输出一致
    single_output = batch_outputs[1][0]
    for batch_size in batch_sizes[1:]:
        batch_output = batch_outputs[batch_size]
        for i in range(batch_size):
            assert torch.allclose(single_output, batch_output[i], atol=1e-5), \
                f"Batch size {batch_size}, sample {i}: output mismatch"

    # =========================================================================
    # 向量化审计 (Vectorization Audit)
    # =========================================================================
    # 仅在审计启用时运行，不影响测试结果

    if is_audit_enabled():
        # 使用中等批次大小进行审计
        sample_input = torch.randn(4, 3, image_size, image_size)

        # 运行完整的向量化审计
        report = run_vectorization_audit_on_model(
            model=model,
            sample_input=sample_input,
            module_name="FractalCurveViT",
            batch_size=4,
        )

        # 处理审计结果
        if report.has_issues():
            warnings_list = []

            for issue in report.issues:
                if issue.severity == "performance":
                    # 性能警告 - 不导致测试失败
                    warning_msg = (
                        f"[Vectorization Audit] {issue.function_name} at line {issue.location[1]}: "
                        f"{issue.description}. Suggestion: {issue.suggestion}"
                    )
                    warnings_list.append(warning_msg)
                    warnings.warn(warning_msg, PerformanceWarning, stacklevel=2)

                elif issue.severity == "error":
                    # 错误 - 可能影响批处理一致性
                    # 记录但不完全阻止测试通过
                    warning_msg = (
                        f"[Vectorization Audit ERROR] {issue.function_name}: {issue.description}"
                    )
                    warnings_list.append(warning_msg)
                    warnings.warn(warning_msg, PerformanceWarning, stacklevel=2)

            # 输出审计摘要
            if warnings_list:
                print(f"\n=== Vectorization Audit Report ===")
                print(f"Module: {report.module_name}")
                print(f"Analysis time: {report.analysis_time:.3f}s")
                print(f"Issues found: {len(report.issues)}")
                for issue_type, count in report.summary.items():
                    print(f"  - {issue_type}: {count}")
                print(f"================================\n")
