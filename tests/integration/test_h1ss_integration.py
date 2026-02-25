"""
H1SS 集成测试脚本
"""
import torch
from vit_pytorch import FractalCurveViT

def test_h1ss_integration():
    print("=" * 60)
    print("H1SS (Hilbert-Optimal Splitter) 集成测试")
    print("=" * 60)

    # 创建模型
    print("\n[1] 创建 FractalCurveViT 模型 (splitter_type='hilbert_optimal')...")
    model = FractalCurveViT(
        image_size=224,
        num_classes=200,
        splitter_type='hilbert_optimal',
        dim=256,
        mlp_dim=512,
        num_layers=4,
        heads=4,
    )
    print(f"    模型创建成功!")

    # 验证 splitter 类型
    splitter_type = type(model.splitter).__name__
    print(f"    Splitter 类型: {splitter_type}")

    # 测试前向传播
    print("\n[2] 测试前向传播...")
    x = torch.randn(2, 3, 224, 224)

    # 训练模式
    model.train()
    output_train = model(x)
    print(f"    训练模式输出类型: {type(output_train).__name__}")
    print(f"    logits shape: {output_train.logits.shape}")
    print(f"    num_tokens: {output_train.num_tokens}")
    print(f"    depth_distribution: {output_train.depth_distribution}")

    # 评估模式
    model.eval()
    with torch.no_grad():
        output_eval = model(x)
    print(f"    评估模式输出类型: {type(output_eval).__name__}")
    print(f"    logits shape: {output_eval.logits.shape}")

    # 测试梯度流
    print("\n[3] 测试梯度流...")
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    model.train()
    output = model(x)
    loss = output.logits.sum()
    loss.backward()
    print(f"    梯度流成功! input.grad is not None: {x.grad is not None}")

    # 测试诊断信息
    print("\n[4] 测试诊断信息...")
    if hasattr(model.splitter, 'get_diagnostics'):
        diagnostics = model.splitter.get_diagnostics()
        print(f"    诊断信息: {diagnostics}")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)

if __name__ == "__main__":
    test_h1ss_integration()
