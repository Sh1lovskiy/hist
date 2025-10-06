# hist/cli.py
"""Single entry point CLI for hist pipeline (lazy imports by subcommand)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

# стараемся не тянуть ничего тяжёлого на верхнем уровне
# модуль логирования должен быть лёгким и без каскадных импортов
from hist.logging import configure_logging, logger  # noqa: E402


def _load_labels(path: Path) -> dict[str, int]:
    with path.open("r", encoding="utf8") as handle:
        reader = csv.DictReader(handle)
        labels = {row["slide_id"]: int(row["label"]) for row in reader}
    if not labels:
        raise ValueError("Label CSV is empty")
    return labels


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified digital pathology pipeline")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ---- tile ----
    tile_parser = subparsers.add_parser("tile", help="Tile WSIs into patch CSVs")
    tile_parser.add_argument("--slides", type=Path, required=True)
    tile_parser.add_argument("--annos", type=Path, required=True)
    tile_parser.add_argument("--out", type=Path, required=True)
    tile_parser.add_argument("--patch-size", type=int, default=224)
    tile_parser.add_argument("--stride", type=int, default=224)
    tile_parser.add_argument("--overwrite", action="store_true")

    # ---- extract ----
    extract_parser = subparsers.add_parser(
        "extract", help="Extract features from patches"
    )
    extract_parser.add_argument("--csv-dir", type=Path, required=True)
    extract_parser.add_argument("--output", type=Path, required=True)
    extract_parser.add_argument("--encoder", type=str, default="vit_b16")
    extract_parser.add_argument("--device", type=str, default="cpu")
    extract_parser.add_argument("--batch-size", type=int, default=64)
    extract_parser.add_argument("--num-workers", type=int, default=0)

    # ---- train ----
    train_parser = subparsers.add_parser("train", help="Train MIL models")
    train_parser.add_argument("--features", type=Path, required=True)
    train_parser.add_argument("--labels-csv", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--model", type=str, default="mean")
    train_parser.add_argument("--epochs", type=int, default=1)
    train_parser.add_argument("--k-folds", type=int, default=2)
    train_parser.add_argument("--oversample", action="store_true")
    train_parser.add_argument("--device", type=str, default="cpu")

    # ---- explain ----
    explain_parser = subparsers.add_parser(
        "explain", help="Generate explainability artifacts"
    )
    explain_parser.add_argument("--runs", type=Path, required=True)
    explain_parser.add_argument("--out", type=Path, required=True)

    # ---- compare ----
    compare_parser = subparsers.add_parser("compare", help="Compare experiment runs")
    compare_parser.add_argument("--out", type=Path, required=True)
    compare_parser.add_argument(
        "--metrics",
        nargs="*",
        default=["balanced_accuracy", "f1_macro"],
        help="Metrics to compute across runs",
    )
    compare_parser.add_argument("inputs", nargs="+", type=Path)

    # ---- infer ----
    infer_parser = subparsers.add_parser("infer", help="Run inference on slides")
    infer_parser.add_argument("--labels-csv", type=Path, required=True)
    infer_parser.add_argument("--out", type=Path, required=True)

    return parser


def main(argv: List[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(level="DEBUG" if args.verbose else "INFO")

    if args.command == "tile":
        # ЛЕНИВЫЙ импорт только здесь
        from hist.tiling.tiler import TileRequest, tile_dataset

        request = TileRequest(
            slides_dir=args.slides,
            annotations_dir=args.annos,
            output_dir=args.out,
            patch_size=args.patch_size,
            stride=args.stride,
            overwrite=args.overwrite,
        )
        out_dir = tile_dataset(request)
        logger.info("Patch CSVs stored in %s", out_dir)
        return

    if args.command == "extract":
        from hist.features.extractor import extract_features

        out_dir = extract_features(
            csv_dir=args.csv_dir,
            output_dir=args.output,
            encoder_name=args.encoder,
            batch_size=args.batch_size,
            device=args.device,
        )
        logger.info("Features stored in %s", out_dir)
        return

    if args.command == "train":
        # ЛЕНИВЫЕ импорты
        from hist.train.cv import CVOptions, run_cv
        from hist.config import RunConfig

        labels = _load_labels(args.labels_csv)
        args.output.mkdir(parents=True, exist_ok=True)
        cv_opts = CVOptions(
            features_dir=args.features,
            labels=labels,
            output_dir=args.output,
            model_name=args.model,
            epochs=args.epochs,
            k_folds=args.k_folds,
            device=args.device,
            oversample=args.oversample,
        )
        run_cv(cv_opts)
        RunConfig(
            command="train", options=dict(model=args.model, epochs=args.epochs)
        ).to_yaml(args.output / "cfg.yaml")
        return

    if args.command == "explain":
        # ЛЕНИВЫЙ импорт
        from hist.utils.paths import ensure_exists
        from hist.explain.attention import export_dummy_attention

        features_dir = ensure_exists(args.runs, kind="directory")
        export_dummy_attention(features_dir, args.out)
        logger.info("Explainability artifacts stored in %s", args.out)
        return

    if args.command == "compare":
        # ЛЕНИВЫЙ импорт
        from hist.eval.compare import compare_runs

        compare_runs(args.inputs, args.out, args.metrics)
        return

    if args.command == "infer":
        labels = _load_labels(args.labels_csv)
        args.out.mkdir(parents=True, exist_ok=True)
        results_path = args.out / "predictions.csv"
        with results_path.open("w", newline="", encoding="utf8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["slide_id", "pred"])
            for slide_id, label in labels.items():
                writer.writerow([slide_id, label])
        logger.info("Predictions saved to %s", results_path)
        return

    # defensive (не должно сюда дойти)
    parser.error(f"Unknown command {args.command}")


if __name__ == "__main__":  # pragma: no cover
    main()
