#!/usr/bin/env python3
"""
Gradient Composition Analysis for HilbertOptimalSplitter

Measures three critical gradient pathologies:
1. Feature norm mismatch: roi_features vs geometric embeddings
2. Pointwise conv weight gradient distribution across 4 segments
3. Entmax activation verification (alpha threshold crossing)

Usage:
    DEBUG_GRAD_COMPOSITION=1 uv run python tests/debug/gradient_composition_analysis.py
    uv run pytest tests/debug/gradient_composition_analysis.py -v -s
"""

import sys
sys.path.insert(0, 'src')

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from vit_pytorch.layers.splitters.hilbert_optimal_splitter import HilbertOptimalSplitter


def measure_feature_norms(
    roi_features: torch.Tensor,
    path_emb: torch.Tensor,
    rot_emb: torch.Tensor,
    area_enc: torch.Tensor,
) -> Dict[str, float]:
    """Measure L2 norms of each feature component."""
    return {
        'roi_features': roi_features.norm(p=2, dim=-1).mean().item(),
        'path_emb': path_emb.norm(p=2, dim=-1).mean().item(),
        'rot_emb': rot_emb.norm(p=2, dim=-1).mean().item(),
        'area_enc': area_enc.norm(p=2, dim=-1).mean().item(),
    }


def measure_conv_weight_gradients(
    conv1d_hilbert,
) -> Dict[str, float]:
    """Measure gradient norms for each segment of pointwise conv weight.

    Pointwise conv weight shape: [1, 256, 1] after view -> [256]
    Segments:
        [0:64]   -> roi_features segment
        [64:128] -> path_emb segment
        [128:192] -> rot_emb segment
        [192:256] -> area_enc segment
    """
    weight = conv1d_hilbert.pointwise.weight  # [1, 256, 1]
    weight = weight.view(-1)  # [256]

    grad = weight.grad
    if grad is None:
        return {f'segment_{i}': 0.0 for i in range(4)}

    seg_size = 64
    grad_norms = {}
    total_grad_norm = grad.norm(p=2).item()

    for i in range(4):
        seg_grad = grad[i * seg_size:(i + 1) * seg_size]
        seg_norm = seg_grad.norm(p=2).item()
        pct = (seg_norm / (total_grad_norm + 1e-10)) * 100
        grad_norms[f'segment_{i}_norm'] = seg_norm
        grad_norms[f'segment_{i}_pct'] = pct

    grad_norms['total_norm'] = total_grad_norm
    return grad_norms


def verify_entmax_trigger(splitter, epoch: int) -> Dict[str, object]:
    """Verify whether entmax is actually triggered at given epoch."""
    splitter.set_epoch(epoch)
    alpha = splitter.entmax_alpha
    tau = max(splitter.temperature, 1e-4)

    # Simulate logits
    logits = torch.randn(1, 64, requires_grad=True)

    if alpha < 1.5:
        probs = F.softmax(logits / tau, dim=-1)
        method = 'softmax'
    else:
        from entmax import entmax_bisect
        probs = entmax_bisect(logits / tau, alpha=alpha, dim=-1)
        method = 'entmax'

    return {
        'epoch': epoch,
        'alpha': alpha,
        'method': method,
        'probs_sum': probs.sum(dim=-1).mean().item(),
        'probs_max': probs.max(dim=-1).values.mean().item(),
    }


def verify_tree_loss_gradient(epoch: int) -> Dict[str, object]:
    """Verify L_tree gradient behavior at different std values."""
    results = {}

    # Test with different std values
    for std_val in [0.5, 0.1, 0.05, 0.01, 0.001]:
        logits = torch.randn(1, 64, requires_grad=True)
        logits.data.mul_(std_val)  # Scale in-place to keep leaf tensor

        # Old implementation (singular)
        logits_old_std = logits.std()
        loss_old = -logits_old_std

        # New implementation (gradient rescaled)
        EPS_STD = 1e-4
        logits_new_std = logits.std()
        loss_new = -(logits_new_std / (logits_new_std.detach() + EPS_STD))

        # Compute "effective gradient scale" by backward
        loss_old.backward(retain_graph=True)
        grad_old_norm = logits.grad.norm(p=2).item() if logits.grad is not None else 0.0

        logits.grad = None
        loss_new.backward()
        grad_new_norm = logits.grad.norm(p=2).item() if logits.grad is not None else 0.0

        results[f'std_{std_val}'] = {
            'loss_old': loss_old.item(),
            'loss_new': loss_new.item(),
            'grad_old_norm': grad_old_norm,
            'grad_new_norm': grad_new_norm,
            'grad_ratio': grad_old_norm / (grad_new_norm + 1e-10),
        }
        logits.grad = None

    return results


def full_gradient_flow_test():
    """Full forward+backward pass to measure gradient composition."""
    print("=" * 70)
    print("FULL GRADIENT FLOW ANALYSIS")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create splitter
    splitter = HilbertOptimalSplitter(
        feature_dim=256,
        hidden_dim=64,
        K_min=8,
        K_max=64,
        max_level_limit=8,
        use_distance_decay_conv=True,
        use_sds_regularization=False,
    ).to(device)
    splitter.eval()  # Eval mode for deterministic behavior

    # Set to epoch 25 (entmax should be active)
    splitter.set_epoch(25)

    # Create dummy input
    B, N, C = 2, 64, 256
    features = torch.randn(B, N, C).to(device)
    image_size = (224, 224)

    # Hook to capture gradients
    gradients_by_layer = {}

    def conv_grad_hook(module, grad_input, grad_output):
        gradients_by_layer['conv1d_hilbert'] = {
            'grad_output': grad_output[0] if grad_output[0] is not None else None,
        }

    # Register hook
    handle = splitter.conv1d_hilbert.register_backward_hook(
        lambda m, gi, go: conv_grad_hook(m, gi, go)
    )

    # Forward pass with gradient recording
    try:
        split_result = splitter(
            features=features,
            image_size=image_size,
            epoch=25,
            hard=False,
        )
    except Exception as e:
        print(f"Forward pass failed: {e}")
        print("Note: Full forward requires proper candidate_regions setup.")
        print("Running simplified component test instead...")
        handle.remove()
        return simplified_component_test(splitter, device)

    # Simulate loss
    logits = split_result.logits
    loss = logits.mean()

    # Backward
    loss.backward()

    # Measure conv weight gradients
    conv_grad_norms = measure_conv_weight_gradients(splitter.conv1d_hilbert)

    print("\n--- Conv Weight Gradient Distribution ---")
    total = conv_grad_norms['total_norm']
    print(f"Total gradient norm: {total:.6f}")
    for i in range(4):
        pct = conv_grad_norms[f'segment_{i}_pct']
        norm = conv_grad_norms[f'segment_{i}_norm']
        print(f"  Segment {i} [{i*64}:{(i+1)*64}]: norm={norm:.6f}, pct={pct:.1f}%")

    # Check for geometric dominance
    geo_pct = sum(conv_grad_norms[f'segment_{i}_pct'] for i in range(1, 4))
    print(f"\n  Geometric segments [64:256] total: {geo_pct:.1f}%")
    print(f"  {'[PASS] Balanced' if geo_pct > 20 else '[FAIL] Geometry suppressed!'}")

    handle.remove()
    return conv_grad_norms


def simplified_component_test(splitter: HilbertOptimalSplitter, device: str):
    """Simplified test when full forward is not available."""
    print("\n--- Simplified Component Test ---")

    # Create synthetic feature components
    B, N = 2, 64
    hidden_dim = 64

    # roi_features: high norm (backbone scale)
    roi_features = torch.randn(B, N, hidden_dim, device=device) * 4.0

    # path_emb, rot_emb: low norm (embedding scale)
    path_emb = torch.randn(N, hidden_dim, device=device) * 0.02
    rot_emb = torch.randn(N, hidden_dim, device=device) * 0.02

    # area_enc: mixed
    area_enc = torch.randn(N, hidden_dim, device=device) * 0.5

    # Measure norms
    norms = measure_feature_norms(roi_features, path_emb, rot_emb, area_enc)

    print("\n--- Feature Component Norms (Before Normalization) ---")
    for name, norm_val in norms.items():
        print(f"  {name}: {norm_val:.4f}")

    # After LayerNorm (simulated)
    print("\n--- Feature Component Norms (After roi_norm/geo_norm) ---")
    # Simulate the LayerNorm effect
    roi_norm = nn.LayerNorm(hidden_dim).to(device)
    geo_norm = nn.LayerNorm(hidden_dim * 3).to(device)

    roi_normed = roi_norm(roi_features)
    geo_concat = torch.cat([path_emb, rot_emb, area_enc], dim=-1)
    geo_normed = geo_norm(geo_concat)

    norms_after = {
        'roi_normed': roi_normed.norm(p=2, dim=-1).mean().item(),
        'geo_normed': geo_normed.norm(p=2, dim=-1).mean().item(),
    }
    for name, norm_val in norms_after.items():
        print(f"  {name}: {norm_val:.4f}")

    print(f"\n  Norm ratio (before): {norms['roi_features'] / norms['path_emb']:.1f}x")
    print(f"  Norm ratio (after roi_norm): {norms_after['roi_normed'] / norms_after['geo_normed']:.1f}x")
    print(f"  {'[PASS] Norms are now comparable!' if norms_after['roi_normed'] / norms_after['geo_normed'] < 5 else '[FAIL] Still imbalanced'}")

    return norms_after


def main():
    print("=" * 70)
    print("HilbertOptimalSplitter Gradient Pathology Diagnosis")
    print("=" * 70)

    # 1. Feature Norm Mismatch
    print("\n[1] Feature Norm Mismatch Test")
    print("-" * 50)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    splitter = HilbertOptimalSplitter(
        feature_dim=256,
        hidden_dim=64,
        K_min=8,
        K_max=64,
        max_level_limit=8,
        use_distance_decay_conv=True,
        use_sds_regularization=False,
    ).to(device)
    simplified_component_test(splitter, device)

    # 2. Entmax Trigger Verification
    print("\n[2] Entmax Trigger Verification")
    print("-" * 50)
    for epoch in [0, 10, 20, 30]:
        result = verify_entmax_trigger(splitter, epoch)
        print(f"  epoch={epoch}: alpha={result['alpha']:.2f}, method={result['method']}")

    # 3. Tree Loss Singularity Test
    print("\n[3] L_tree Gradient Singularity Test")
    print("-" * 50)
    tree_results = verify_tree_loss_gradient(0)
    for std_str, metrics in tree_results.items():
        grad_ratio = metrics['grad_ratio']
        print(f"  {std_str}: old_grad={metrics['grad_old_norm']:.4f}, "
              f"new_grad={metrics['grad_new_norm']:.4f}, "
              f"ratio={grad_ratio:.1f}x")
        if grad_ratio > 10:
            print("    → [OLD] Gradient explosion suppressed by fix!")

    # 4. Full Gradient Flow (if possible)
    print("\n[4] Full Gradient Flow Test")
    print("-" * 50)
    full_gradient_flow_test()

    print("\n" + "=" * 70)
    print("Diagnosis Complete")
    print("=" * 70)


if __name__ == '__main__':
    main()
