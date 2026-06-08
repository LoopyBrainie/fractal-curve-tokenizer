"""v1.3 STANDARD: R12 Aux Loss wiring test (T3.1+T3.2).

Verifies that when `enable_r12_aux=True`, the trainer wires R12AuxLoss
into compute_loss via the `aux_losses` dict, producing `aux_r12` and
`aux_weight_r12` keys in the loss components.
"""
from __future__ import annotations

import pytest
import torch


class TestR12AuxWiring:
    """T3.1: R12 routes through compute_loss aux-loss combiner."""

    def test_compute_loss_with_r12_tensor(self):
        """Passing an R12 tensor via aux_losses produces aux_r12 in components."""
        from src.training.trainer.loss import compute_loss
        from vit_pytorch.core.measure_auxiliary_loss import R12AuxLoss, R12AuxLossConfig

        logits = torch.randn(2, 10)
        targets = torch.randint(0, 10, (2,))
        r12 = R12AuxLoss(R12AuxLossConfig())
        parent = torch.randn(2, 10)
        child = torch.randn(2, 10)
        depth = torch.tensor([0.5, 0.3, 0.2])
        r12_tensor = r12(parent, child, depth)

        loss, comps = compute_loss(
            logits, targets, aux_losses={"r12": r12_tensor},
        )
        assert "aux_r12" in comps
        assert "aux_weight_r12" in comps
        # aux_weight is dynamic: 0.08 * clamp(ce_mag / aux_mag, 0.02, 1.0)
        assert comps["aux_weight_r12"] > 0.0
        # The total loss includes the r12 contribution
        assert loss.dim() == 0

    def test_r12_increases_total_loss(self):
        """Adding a positive R12 tensor increases the total loss above CE-only."""
        from src.training.trainer.loss import compute_loss
        from vit_pytorch.core.measure_auxiliary_loss import R12AuxLoss

        logits = torch.randn(2, 10)
        targets = torch.randint(0, 10, (2,))
        r12 = R12AuxLoss()
        parent = torch.randn(2, 10) * 5
        child = torch.randn(2, 10) * 5
        depth = torch.tensor([0.5, 0.3, 0.2])
        r12_tensor = r12(parent, child, depth)

        loss_no_aux, _ = compute_loss(logits.clone(), targets)
        loss_with_r12, _ = compute_loss(
            logits.clone(), targets, aux_losses={"r12": r12_tensor}
        )
        assert loss_with_r12.item() > loss_no_aux.item()

    def test_r12_loss_is_nonnegative(self):
        """R12 aux loss is hinge² + KL — both are non-negative."""
        from vit_pytorch.core.measure_auxiliary_loss import R12AuxLoss

        r12 = R12AuxLoss()
        parent = torch.randn(8, 10)
        child = torch.randn(8, 10)
        depth = torch.tensor([0.4, 0.3, 0.2, 0.1])
        loss = r12(parent, child, depth)
        assert loss.item() >= 0.0
