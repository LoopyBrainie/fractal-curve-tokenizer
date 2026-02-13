import torch
import torch.nn as nn
from torch.optim import AdamW
import pytest

from vit_pytorch.models.fractal_vit import FractalCurveViT
from vit_pytorch.modules.base_tokenizer import BaseTokenProcessor, BaseTokenizer, TokenSequence, TokenizerOutput


class DummyTokenizer(BaseTokenizer):
    def __init__(self, token_dim: int, tokens_per_image: int = 3) -> None:
        super().__init__()
        self.token_dim = token_dim
        self.tokens_per_image = tokens_per_image
        self.max_level = 3  # I145: 添加 max_level 属性
        self.called = False

    def tokenize(self, images: torch.Tensor) -> TokenizerOutput:
        self.called = True
        batch: list[TokenSequence] = []
        for _ in range(images.shape[0]):
            tokens = torch.full((self.tokens_per_image, self.token_dim), 0.5, device=images.device)
            # Create dummy levels with enough columns to satisfy max_info_len check if needed
            # But FractalCurveViT handles variable lengths.
            # Let's give it 2 columns (depth, path)
            levels = torch.zeros(self.tokens_per_image, 2, dtype=torch.long, device=images.device)
            batch.append(TokenSequence(tokens=tokens, metadata={"levels": levels}))
        return TokenizerOutput(batch)


class DummyProcessor(BaseTokenProcessor):
    def __init__(self) -> None:
        super().__init__()
        self.called = False

    def process(self, batch: TokenizerOutput) -> TokenizerOutput:
        self.called = True
        return batch


class DummyPositional(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.called = False

    def forward(
        self,
        levels_info,
        sequence_positions: torch.Tensor | None = None,
        *,
        regions: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        self.called = True
        # I98-4: Support both LevelsInfo and raw tensor
        if hasattr(levels_info, 'data'):
            data = levels_info.data
        else:
            data = levels_info

        if data.numel() == 0:
            return torch.zeros(0, self.dim, device=data.device)

        # levels_info shape is (Batch, Seq, Info) from FractalCurveViT
        # We need to return (Batch, Seq, Dim) matching the expected shape
        if data.dim() == 3:
            # Shape: (B, Seq, Info) -> return (B, Seq, Dim)
            return torch.zeros(data.shape[0], data.shape[1], self.dim, device=data.device)
        else:
            # Shape: (Batch*Seq, Info) -> return (Batch*Seq, Dim)
            return torch.zeros(data.shape[0], self.dim, device=data.device)


def test_vit_uses_custom_components() -> None:
    """测试 FractalCurveViT 使用自定义组件."""
    dim = 16
    batch_size = 2

    tokenizer = DummyTokenizer(token_dim=dim)
    positional = DummyPositional(dim=dim)

    model = FractalCurveViT(
        image_size=32,
        num_classes=4,
        dim=dim,
        num_layers=1,
        heads=2,
        mlp_dim=32,
        min_patch_size=4,
        tokenizer=tokenizer,
        position_embedding=positional,
    )

    images = torch.randn(batch_size, 3, 32, 32)
    result = model(images)
    outputs = result.logits if hasattr(result, 'logits') else result

    assert outputs.shape == (batch_size, 4)
    assert tokenizer.called
    assert positional.called


def test_next_gen_vit_single_training_step_updates_parameters() -> None:
    """测试 FractalCurveViT 单步训练更新参数."""
    torch.manual_seed(42)
    model = FractalCurveViT(
        image_size=32,
        num_classes=5,
        dim=64,
        num_layers=2,
        heads=4,
        mlp_dim=128,
        min_patch_size=4,
    )

    model.train()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    inputs = torch.randn(3, 3, 32, 32)
    labels = torch.randint(0, 5, (3,))

    params_before = [p.detach().clone() for p in model.parameters() if p.requires_grad]

    optimizer.zero_grad()
    result = model(inputs)
    logits = result.logits if hasattr(result, 'logits') else result
    loss = criterion(logits, labels)
    
    # Handle auxiliary loss
    if hasattr(model, "get_tokenizer_loss"):
        aux_loss = model.get_tokenizer_loss()
    else:
        aux_loss = torch.tensor(0.0, device=inputs.device)

    total_loss = loss + 0.1 * aux_loss
    total_loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.any(g != 0) for g in grads)

    optimizer.step()

    params_after = [p.detach().clone() for p in model.parameters() if p.requires_grad]
    assert any(torch.sum(torch.abs(after - before)) > 0 for before, after in zip(params_before, params_after))
