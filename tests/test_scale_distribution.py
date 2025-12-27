"""Test scale distribution logging."""
import sys
import torch
sys.path.insert(0, 'src')
from vit_pytorch.tokenizer_streaming import StreamingFractalTokenizerV3

torch.manual_seed(42)
img = torch.randn(8, 3, 64, 64)
img[:, :, :16, :16] = 0.5  # 平坦区域

print('=== Testing Higher tau_0 Values ===')
for tau_0 in [0.15, 0.30, 0.50, 0.70, 0.90, 0.95, 0.99]:
    tokenizer = StreamingFractalTokenizerV3(
        image_size=64, channels=3, d_model=192,
        base_patch_size=4, max_depth=2,
        tau_0=tau_0, gamma=0.85,
    )
    dist = tokenizer.compute_scale_distribution(img)
    ratios = dist['scale_ratios']
    ratio_str = ', '.join([f'p{ps}:{r*100:.0f}%' for ps, r in sorted(ratios.items(), reverse=True)])
    entropy = dist["entropy"]
    print(f'tau_0={tau_0}: {ratio_str}  entropy={entropy:.2f}')
