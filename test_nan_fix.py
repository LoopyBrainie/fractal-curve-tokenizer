# test_nan_fix.py - 验证梯度流修复
import torch
import torch.nn as nn
from vit_pytorch import FractalCurveViT

def test_gradient_flow(manifold_native=True):
    """测试梯度流是否稳定"""
    print(f"Testing gradient flow (manifold_native={manifold_native})...")
    torch.autograd.set_detect_anomaly(True)

    # 创建小测试模型
    model = FractalCurveViT(
        image_size=64,
        num_classes=200,
        dim=128,
        num_layers=2,
        heads=4,
        mlp_dim=256,
        min_patch_size=16,
        use_manifold_native=manifold_native,
        depth_scale_range=(0.5, 2.0),
    )
    model = model.cuda() if torch.cuda.is_available() else model
    model.train()

    # 随机输入
    x = torch.randn(2, 3, 64, 64)
    if torch.cuda.is_available():
        x = x.cuda()

    # 前向传播
    try:
        output = model(x)
        print(f"Forward pass OK, num_tokens={output.num_tokens}")
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 反向传播
    try:
        loss = output.logits.mean()
        loss.backward()
        print(f"Backward pass OK, loss={loss.item():.4f}")
    except RuntimeError as e:
        if "nan" in str(e).lower():
            print(f"NaN detected in backward: {e}")
            return False
        raise
    except Exception as e:
        print(f"Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 检查梯度
    has_nan = False
    nan_layers = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN gradient in {name}")
                nan_layers.append(name)
                has_nan = True

    if not has_nan:
        print("All gradients are finite")

    return not has_nan

if __name__ == "__main__":
    # 测试两种配置
    success1 = test_gradient_flow(manifold_native=False)
    print(f"\n{'='*50}")
    print(f"Test (manifold_native=False): {'PASSED' if success1 else 'FAILED'}")

    success2 = test_gradient_flow(manifold_native=True)
    print(f"\n{'='*50}")
    print(f"Test (manifold_native=True): {'PASSED' if success2 else 'FAILED'}")

    print(f"\n{'='*50}")
    print(f"Overall: {'ALL PASSED' if (success1 and success2) else 'SOME FAILED'}")
