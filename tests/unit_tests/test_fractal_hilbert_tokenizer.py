import torch
from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer

def test_fractal_hilbert_tokenizer():
    print("=== 测试 FractalHilbertTokenizer ===")
    
    # 创建tokenizer
    tokenizer = FractalHilbertTokenizer(
        min_patch_size=(16, 9),
        max_level=3
    )
    
    # 测试1: 小图像 (32x18)
    print("\n测试1: 32x18 图像")
    image_small = torch.randn(1, 3, 32, 18)
    tokens_list, levels_list = tokenizer(image_small)
    tokens, levels = tokens_list[0], levels_list[0]
    print(f"输入图像形状: {image_small.shape}")
    print(f"输出tokens形状: {tokens.shape}")
    print(f"输出levels形状: {levels.shape}")
    if levels.shape[0] > 0:
        print(f"第一个token的层级向量: {levels[0]}")
    
    # 测试2: 中等图像 (64x36)
    print("\n测试2: 64x36 图像")
    image_medium = torch.randn(1, 3, 64, 36)
    tokens_list, levels_list = tokenizer(image_medium)
    tokens, levels = tokens_list[0], levels_list[0]
    print(f"输入图像形状: {image_medium.shape}")
    print(f"输出tokens形状: {tokens.shape}")
    print(f"输出levels形状: {levels.shape}")
    print(f"前5个token的层级向量:")
    for i in range(min(5, levels.shape[0])):
        print(f"  Token {i}: {levels[i]}")
    
    # 测试3: 大图像 (128x72)
    print("\n测试3: 128x72 图像")
    image_large = torch.randn(1, 3, 128, 72)
    tokens_list, levels_list = tokenizer(image_large)
    tokens, levels = tokens_list[0], levels_list[0]
    print(f"输入图像形状: {image_large.shape}")
    print(f"输出tokens形状: {tokens.shape}")
    print(f"输出levels形状: {levels.shape}")
    
    # 测试4: Batch处理
    print("\n测试4: Batch处理 (2张32x18图像)")
    image_batch = torch.randn(2, 3, 32, 18)
    tokens_list, levels_list = tokenizer(image_batch)
    print(f"输入图像形状: {image_batch.shape}")
    print(f"输出: 第一张图 tokens 形状: {tokens_list[0].shape}, 第二张图 tokens 形状: {tokens_list[1].shape}")
    print(f"输出levels形状: 第一张: {levels_list[0].shape}, 第二张: {levels_list[1].shape}")
    
    # 测试5: 检查自适应Hilbert遍历
    print("\n测试5: 检查自适应Hilbert遍历")
    
    # 测试不同层级和尺寸的遍历顺序
    test_cases = [
        (0, 16, 16, "层级0,方形"),
        (1, 8, 12, "层级1,长方形"),
        (2, 6, 4, "层级2,高长方形"),
    ]
    
    for level, h, w, desc in test_cases:
        order = tokenizer.get_hilbert_order_for_level(level, h, w)
        adaptive_order = tokenizer.adaptive_hilbert_order(h, w, level)
        print(f"  {desc}: 基础顺序={order}, 自适应顺序={adaptive_order}")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_fractal_hilbert_tokenizer()
