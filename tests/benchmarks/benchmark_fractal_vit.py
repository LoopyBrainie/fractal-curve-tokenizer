from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import torch
from torch import nn
from torch.optim.adamw import AdamW

import sys

# Allow running the script directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vit_pytorch.fractal_curve_tokenizer import FractalHilbertTokenizer
from vit_pytorch.fractal_vit import NextGenerationFractalViT, SimpleFractalViT


@dataclass
class BenchmarkResult:
    label: str
    batch_size: int
    image_size: tuple[int, int]
    mean_ms: float
    std_ms: float
    extra: str = ""


def _format_size(image_size: tuple[int, int]) -> str:
    return f"{image_size[0]}x{image_size[1]}"


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is available.")
    return torch.device(requested)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_execution(
    fn: Callable[[], None],
    *,
    warmup: int,
    repeat: int,
    sync: Callable[[], None],
) -> tuple[float, float]:
    for _ in range(max(0, warmup)):
        fn()
        sync()

    timings: list[float] = []
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        fn()
        sync()
        timings.append((time.perf_counter() - start) * 1000.0)

    return statistics.mean(timings), statistics.pstdev(timings) if len(timings) > 1 else 0.0


def _generate_images(batch_size: int, image_size: tuple[int, int], device: torch.device) -> torch.Tensor:
    height, width = image_size
    return torch.randn(batch_size, 3, height, width, device=device)


def _token_count(output) -> int:
    return sum(seq.tokens.shape[0] for seq in output.sequences)


def benchmark_tokenizer(
    device: torch.device,
    batch_sizes: Sequence[int],
    image_sizes: Sequence[tuple[int, int]],
    *,
    warmup: int,
    repeat: int,
) -> list[BenchmarkResult]:
    tokenizer = FractalHilbertTokenizer(min_patch_size=(4, 4), max_level=3).to(device)
    tokenizer.eval()
    results: list[BenchmarkResult] = []

    with torch.no_grad():
        for batch in batch_sizes:
            for image_size in image_sizes:
                inputs = _generate_images(batch, image_size, device)
                token_total = 0

                def run() -> None:
                    nonlocal token_total
                    output = tokenizer.tokenize(inputs)
                    token_total = _token_count(output)

                mean_ms, std_ms = _time_execution(
                    run,
                    warmup=warmup,
                    repeat=repeat,
                    sync=lambda: _synchronize(device),
                )

                results.append(
                    BenchmarkResult(
                        label="tokenizer",
                        batch_size=batch,
                        image_size=image_size,
                        mean_ms=mean_ms,
                        std_ms=std_ms,
                        extra=f"tokens={token_total}",
                    )
                )

    return results


def _build_simple(image_size: tuple[int, int]) -> SimpleFractalViT:
    return SimpleFractalViT(
        image_size=image_size,
        num_classes=10,
        dim=128,
        depth=3,
        heads=4,
        mlp_dim=256,
        min_patch_size=(4, 4),
        max_level=3,
    )


def _build_next(image_size: tuple[int, int]) -> NextGenerationFractalViT:
    return NextGenerationFractalViT(
        image_size=image_size,
        num_classes=10,
        dim=160,
        depth=3,
        heads=5,
        mlp_dim=320,
        min_patch_size=(4, 4),
        max_level=3,
        use_dynamic_depth=False,
    )


def benchmark_forward(
    label: str,
    builder: Callable[[tuple[int, int]], nn.Module],
    device: torch.device,
    batch_sizes: Sequence[int],
    image_sizes: Sequence[tuple[int, int]],
    *,
    warmup: int,
    repeat: int,
) -> list[BenchmarkResult]:
    results: list[BenchmarkResult] = []

    with torch.no_grad():
        for image_size in image_sizes:
            model = builder(image_size).to(device)
            model.eval()

            for batch in batch_sizes:
                inputs = _generate_images(batch, image_size, device)

                mean_ms, std_ms = _time_execution(
                    lambda: model(inputs),
                    warmup=warmup,
                    repeat=repeat,
                    sync=lambda: _synchronize(device),
                )

                results.append(
                    BenchmarkResult(
                        label=label,
                        batch_size=batch,
                        image_size=image_size,
                        mean_ms=mean_ms,
                        std_ms=std_ms,
                    )
                )

    return results


def _auxiliary_loss(model: nn.Module, device: torch.device) -> torch.Tensor:
    if hasattr(model, "get_tokenizer_loss"):
        return model.get_tokenizer_loss()
    if hasattr(model, "enhanced_model") and hasattr(model.enhanced_model, "get_tokenizer_loss"):
        return model.enhanced_model.get_tokenizer_loss()
    return torch.tensor(0.0, device=device)


def benchmark_training_step(
    device: torch.device,
    batch_size: int,
    image_size: tuple[int, int],
    *,
    warmup: int,
    repeat: int,
) -> list[BenchmarkResult]:
    model = _build_simple(image_size).to(device)
    optimizer = AdamW(model.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()
    model.train()

    def run() -> None:
        optimizer.zero_grad(set_to_none=True)
        inputs = _generate_images(batch_size, image_size, device)
        labels = torch.randint(0, 10, (batch_size,), device=device)
        logits = model(inputs)
        loss = criterion(logits, labels) + 0.1 * _auxiliary_loss(model, device)
        loss.backward()
        optimizer.step()

    mean_ms, std_ms = _time_execution(
        run,
        warmup=warmup,
        repeat=repeat,
        sync=lambda: _synchronize(device),
    )

    return [
        BenchmarkResult(
            label="train-step",
            batch_size=batch_size,
            image_size=image_size,
            mean_ms=mean_ms,
            std_ms=std_ms,
        )
    ]


def _print_results(title: str, results: Iterable[BenchmarkResult]) -> None:
    print(f"\n{title}")
    print("label         batch size   mean(ms)  std(ms)  notes")
    for item in results:
        print(
            f"{item.label:12} {item.batch_size:5d} {_format_size(item.image_size):>8} "
            f"{item.mean_ms:8.2f} {item.std_ms:8.2f} {item.extra}"
        )


def _parse_image_sizes(raw: Sequence[str]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for item in raw:
        if "x" not in item:
            raise ValueError(f"Invalid image size '{item}', expected HxW format.")
        height_str, width_str = item.lower().split("x", maxsplit=1)
        sizes.append((int(height_str), int(width_str)))
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fractal ViT benchmarking utility")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="execution device")
    parser.add_argument("--mode", default="all", choices=["all", "tokenizer", "forward", "train"], help="subset of benchmarks to run")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2], help="batch sizes to test")
    parser.add_argument(
        "--image-sizes",
        type=str,
        nargs="+",
        default=["32x32", "48x48"],
        help="image sizes in HxW format",
    )
    parser.add_argument("--repeats", type=int, default=5, help="measurement repeats")
    parser.add_argument("--warmup", type=int, default=2, help="warmup iterations before timing")
    parser.add_argument("--train-batch", type=int, default=4, help="batch size for the training step benchmark")
    parser.add_argument("--train-image", type=str, default="40x40", help="image size for the training step benchmark")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _select_device(args.device)
    image_sizes = _parse_image_sizes(args.image_sizes)
    train_image_size = _parse_image_sizes([args.train_image])[0]

    print(f"Using device: {device}")

    if args.mode in {"all", "tokenizer"}:
        token_results = benchmark_tokenizer(
            device,
            batch_sizes=args.batch_sizes,
            image_sizes=image_sizes,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        _print_results("Tokenizer", token_results)

    if args.mode in {"all", "forward"}:
        simple_results = benchmark_forward(
            "simple-fwd",
            _build_simple,
            device,
            batch_sizes=args.batch_sizes,
            image_sizes=image_sizes,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        next_results = benchmark_forward(
            "next-fwd",
            _build_next,
            device,
            batch_sizes=args.batch_sizes,
            image_sizes=image_sizes,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        _print_results("Forward", [*simple_results, *next_results])

    if args.mode in {"all", "train"}:
        train_results = benchmark_training_step(
            device,
            batch_size=args.train_batch,
            image_size=train_image_size,
            warmup=args.warmup,
            repeat=args.repeats,
        )
        _print_results("Training Step", train_results)


if __name__ == "__main__":
    main()
