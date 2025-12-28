"""Test scale distribution logging with LearnableSplitter temperature."""
import sys
import torch
sys.path.insert(0, 'src')
from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3

torch.manual_seed(42)
img = torch.randn(8, 3, 64, 64)
img[:, :, :16, :16] = 0.5  # 平坦区域

print('=== Testing Temperature Effect on Scale Distribution ===')
print('(Lower temperature → harder decisions → more consistent splits)')

# Test with different temperatures
for temp in [1.0, 0.5, 0.3, 0.1, 0.05]:
    tokenizer = StreamingFractalTokenizerV3(
        image_size=64, channels=3, d_model=192,
        base_patch_size=4, max_depth=2,
        gamma=0.85,
        learnable_temperature=temp,
    )
    dist = tokenizer.compute_scale_distribution(img)
    ratios = dist['scale_ratios']
    ratio_str = ', '.join([f'p{ps}:{r*100:.0f}%' for ps, r in sorted(ratios.items(), reverse=True)])
    entropy = dist["entropy"]
    print(f'T={temp:.2f}: {ratio_str}  entropy={entropy:.2f}')
