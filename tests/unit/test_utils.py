"""Unit tests for utility functions in utils.py."""

import pytest
import torch

from vit_pytorch.utils import (
    create_attention_mask,
    extract_depths,
    normalize_levels_info,
    pair,
    exists,
    default,
)


class TestExtractDepths:
    """Tests for extract_depths function."""

    def test_extract_depths_2d_tensor(self) -> None:
        """Test depth extraction from 2D tensor (Seq, Info)."""
        levels_info = torch.tensor([
            [0, 1, 2, 3],  # depth=0
            [1, 2, 3, 4],  # depth=1
            [2, 3, 4, 5],  # depth=2
        ])
        depths = extract_depths(levels_info, max_level=5)
        
        assert depths.shape == (3,)
        assert depths.tolist() == [0, 1, 2]

    def test_extract_depths_3d_tensor(self) -> None:
        """Test depth extraction from 3D tensor (Batch, Seq, Info)."""
        levels_info = torch.tensor([
            [[0, 1, 2], [1, 2, 3]],  # batch 0
            [[2, 3, 4], [3, 4, 5]],  # batch 1
        ])
        depths = extract_depths(levels_info, max_level=5)
        
        assert depths.shape == (2, 2)
        assert depths.tolist() == [[0, 1], [2, 3]]

    def test_extract_depths_clamps_to_max_level(self) -> None:
        """Test that depths are clamped to max_level."""
        levels_info = torch.tensor([
            [10, 1, 2],  # depth=10, should clamp to 3
            [0, 2, 3],   # depth=0
        ])
        depths = extract_depths(levels_info, max_level=3)
        
        assert depths.tolist() == [3, 0]

    def test_extract_depths_clamps_negative_to_zero(self) -> None:
        """Test that negative depths are clamped to 0."""
        levels_info = torch.tensor([
            [-1, 1, 2],  # depth=-1, should clamp to 0
            [2, 2, 3],
        ])
        depths = extract_depths(levels_info, max_level=5)
        
        assert depths.tolist() == [0, 2]

    def test_extract_depths_returns_long_tensor(self) -> None:
        """Test that returned tensor has dtype long."""
        levels_info = torch.tensor([[1.5, 2.0, 3.0]])  # float input
        depths = extract_depths(levels_info, max_level=5)
        
        assert depths.dtype == torch.long


class TestNormalizeLevelsInfo:
    """Tests for normalize_levels_info function."""

    def test_normalize_2d_to_3d(self) -> None:
        """Test that 2D tensor is expanded to 3D."""
        levels_info = torch.tensor([
            [0, 1, 2],
            [1, 2, 3],
        ])  # (2, 3)
        
        normalized = normalize_levels_info(levels_info)
        
        assert normalized.dim() == 3
        assert normalized.shape == (1, 2, 3)
        assert torch.equal(normalized[0], levels_info)

    def test_normalize_3d_unchanged(self) -> None:
        """Test that 3D tensor remains unchanged."""
        levels_info = torch.tensor([
            [[0, 1, 2], [1, 2, 3]],
            [[2, 3, 4], [3, 4, 5]],
        ])  # (2, 2, 3)
        
        normalized = normalize_levels_info(levels_info)
        
        assert normalized.dim() == 3
        assert normalized.shape == (2, 2, 3)
        assert torch.equal(normalized, levels_info)


class TestHelperFunctions:
    """Tests for helper functions pair, exists, default."""

    def test_pair_with_int(self) -> None:
        assert pair(5) == (5, 5)
        assert pair(1) == (1, 1)

    def test_pair_with_tuple(self) -> None:
        assert pair((3, 4)) == (3, 4)
        assert pair((1, 2)) == (1, 2)

    def test_exists_with_none(self) -> None:
        assert exists(None) is False

    def test_exists_with_value(self) -> None:
        assert exists(0) is True
        assert exists("") is True
        assert exists([]) is True
        assert exists(42) is True

    def test_default_with_none(self) -> None:
        assert default(None, 10) == 10
        assert default(None, "fallback") == "fallback"

    def test_default_with_value(self) -> None:
        assert default(5, 10) == 5
        assert default(0, 10) == 0  # 0 is a valid value
        assert default("", "fallback") == ""  # empty string is a valid value


class TestCreateAttentionMask:
    """Tests for create_attention_mask function."""

    def test_empty_input(self) -> None:
        """Test with empty input list."""
        mask = create_attention_mask([], torch.device("cpu"))
        assert mask.shape == (0, 0, 0)

    def test_single_sample(self) -> None:
        """Test with single sample."""
        levels_info = [torch.tensor([[0, 1], [0, 2], [1, 3]])]
        mask = create_attention_mask(levels_info, torch.device("cpu"))
        
        assert mask.shape == (1, 3, 3)
        # Same level (0-0, 1-1) should have value 1.2
        assert mask[0, 0, 1] == 1.2  # both depth 0
        # Adjacent level should have 1.1
        assert mask[0, 0, 2] == 1.1  # depth 0 vs 1

    def test_batch_with_different_lengths(self) -> None:
        """Test with batch having different sequence lengths."""
        levels_info = [
            torch.tensor([[0, 1], [1, 2]]),  # length 2
            torch.tensor([[0, 1], [0, 2], [1, 3]]),  # length 3
        ]
        mask = create_attention_mask(levels_info, torch.device("cpu"))
        
        assert mask.shape == (2, 3, 3)  # max_len = 3
