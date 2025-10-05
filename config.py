"""Configuration objects and CLI parsing for the WSI pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List
import argparse

AUG_CHOICES = ("none", "basic", "strong")
MODEL_CHOICES = ("abmil", "hiermil", "transmil", "gcn", "gat", "all")
ENCODER_CHOICES = (
    "resnet18",
    "resnet50",
    "vit_b16",
    "convnext_base",
    "clip_vitb16",
    "hipt",
)


@dataclass
class DataConfig:
    """Dataset and tiling configuration."""

    slides: Path
    annotations: Path
    cache: Path
    magnifications: List[int]
    patch_size: int
    csv_dir: Path


@dataclass
class TrainerConfig:
    """Training related hyper-parameters."""

    batch_size: int
    num_workers: int
    lr: float
    epochs: int
    k_folds: int
    device: str
    mixed_precision: bool
    oversample: bool
    class_weighting: bool


@dataclass
class ExperimentConfig:
    """Full experiment configuration container."""

    data: DataConfig
    trainer: TrainerConfig
    model: str
    encoder: str
    augmentation: str
    stain_normalization: bool
    output: Path
    verbose: str


def _default_data(root: Path) -> DataConfig:
    return DataConfig(
        slides=root / "slides",
        annotations=root / "annotations",
        cache=root / "cache",
        magnifications=[40, 10],
        patch_size=256,
        csv_dir=root / "patch_csvs",
    )


def _default_trainer(device: str) -> TrainerConfig:
    return TrainerConfig(
        batch_size=64,
        num_workers=8,
        lr=2e-4,
        epochs=40,
        k_folds=5,
        device=device,
        mixed_precision=True,
        oversample=True,
        class_weighting=True,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the cross-validation training argument parser.

    The parser advertises concrete end-to-end examples so that invoking
    ``python -m train_mil_cv --help`` provides an immediate cheatsheet for the
    complete research pipeline.
    """

    usage_examples = """\
Examples:
  # 1️⃣ Generate patch metadata from WSIs and annotations
  python -m runner \
    --slides data/wss1_v2/out/train/slides \
    --annos  data/wss1_v2/anno \
    --out    data/wss1_v2/out/train \
    --levels 0 1 2 \
    --patch-size 224 \
    --stride 224

  # 2️⃣ Train Hierarchical MIL (ViT backbone) with CV
  python -m train_mil_cv \
    --data-root data/wss1_v2/out/train \
    --output runs/hiermil_vit224 \
    --model hiermil --encoder vit_b16 \
    --aug strong --stain --epochs 20 --k-folds 5 --device cuda \
    --patch-size 224

  # 3️⃣ Compare GCN vs GAT vs TransMIL vs ABMIL vs HierMIL
  python -m train_mil_cv \
    --data-root data/wss1_v2/out/train \
    --output runs/compare_models \
    --model all --encoder vit_b16 --aug strong --epochs 20 \
    --k-folds 5 --device cuda

  # 4️⃣ Visualize attention & embeddings
  python -m explain \
    --features runs/hiermil_vit224/features.npy \
    --attn runs/hiermil_vit224/attn/ \
    --out runs/hiermil_vit224/tsne/
"""

    parser = argparse.ArgumentParser(
        "train_mil_cv",
        description=(
            "Cross-validated weakly supervised MIL / Hierarchical MIL training "
            "for whole-slide histopathology."
        ),
        epilog=usage_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/latest"))
    parser.add_argument("--model", choices=MODEL_CHOICES, default="hiermil")
    parser.add_argument("--encoder", choices=ENCODER_CHOICES, default="vit_b16")
    parser.add_argument("--aug", choices=AUG_CHOICES, default="strong")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--k-folds", type=int, default=5)
    parser.add_argument("--cv", type=int, dest="k_folds", help="Alias for --k-folds")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-oversample", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--stain",
        action="store_true",
        help="Enable Macenko stain normalization before augmentation",
    )
    parser.add_argument("--verbose", choices=("info", "debug"), default="info")
    parser.add_argument("--seed", type=int, default=17)
    return parser


def build_config(args: argparse.Namespace) -> ExperimentConfig:
    normalized_device = _normalize_device(args.device)
    data = _default_data(args.data_root)
    trainer = _default_trainer(normalized_device)
    trainer.batch_size = args.batch_size
    trainer.num_workers = args.num_workers
    trainer.lr = args.lr
    trainer.epochs = args.epochs
    trainer.k_folds = args.k_folds
    trainer.mixed_precision = not args.no_mixed_precision
    trainer.oversample = not args.no_oversample
    trainer.class_weighting = not args.no_class_weights
    trainer.device = normalized_device
    data.patch_size = args.patch_size
    output = args.output
    return ExperimentConfig(
        data=data,
        trainer=trainer,
        model=args.model,
        encoder=args.encoder,
        augmentation=args.aug,
        stain_normalization=args.stain,
        output=output,
        verbose=args.verbose,
    )


def _normalize_device(device_str: str) -> str:
    mapping = {"gpu": "cuda", "cuda": "cuda", "cpu": "cpu"}
    lowered = device_str.lower()
    if lowered in mapping:
        return mapping[lowered]
    return device_str
