"""Training orchestration for cross-validated MIL and GNN models."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from loguru import logger
from torch import nn
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as GeoDataLoader
from tqdm.auto import tqdm

from annotations import AnnotationParser, SlideAnnotation
from config import DataConfig, ExperimentConfig, TrainerConfig
from data_wrappers import (
    Bag,
    BagDataset,
    PatchDataset,
    compute_class_weights,
    oversample_bags,
)
from encoders import EncoderConfig, build_encoder
from explain import save_attention_weights
from features import FeatureExtractor
from metrics_ext import compute_metrics, confusion_matrix_png
from models_mil import ABMIL, HierMIL, TransMIL
from models_gnn import GATHead, GCNHead
from graphs import GraphBuilder, GraphConfig
from plotting import plot_losses, plot_metrics
from stain import build_transforms
from tiler import PatchRecord, Tiler
from wsi_reader import WSIReader


@dataclass
class EpochOutputs:
    loss: float
    targets: List[int]
    preds: List[int]
    probs: List[np.ndarray]
    attentions: Dict[str, Dict[int, torch.Tensor]]


class MILTrainer:
    """Train MIL models with mixed precision support."""

    def __init__(
        self,
        model_name: str,
        feature_dim: int,
        n_classes: int,
        magnifications: Sequence[int],
        device: torch.device,
        lr: float,
        class_weights: torch.Tensor | None,
        use_amp: bool,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_amp = use_amp
        self.magnifications = tuple(sorted(magnifications, reverse=True))
        self.model = self._build_model(model_name, feature_dim, n_classes)
        self.model.to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.scaler = GradScaler(enabled=use_amp)
        self.primary_mag = self.magnifications[0]

    def _build_model(self, name: str, in_dim: int, n_classes: int):
        name = name.lower()
        if name == "abmil":
            return ABMIL(in_dim, n_classes)
        if name == "transmil":
            return TransMIL(in_dim, n_classes)
        if name == "hiermil":
            return HierMIL(in_dim, n_classes, self.magnifications)
        raise ValueError(f"Unsupported model {name}")

    def train_epoch(self, loader: DataLoader) -> EpochOutputs:
        self.model.train()
        losses: List[float] = []
        preds: List[int] = []
        targets: List[int] = []
        probs: List[np.ndarray] = []
        attentions: Dict[str, Dict[int, torch.Tensor]] = {}
        progress = tqdm(loader, desc="train", leave=False)
        for features, label, _, slide_id in progress:
            label_tensor = torch.tensor([label], device=self.device)
            bag_input = self._prepare_input(features)
            self.optimizer.zero_grad()
            with autocast("cuda", enabled=self.use_amp):
                logits, attn = self._forward(bag_input)
                loss = self.criterion(logits, label_tensor)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.append(float(loss.item()))
            preds.append(int(torch.argmax(logits.detach(), dim=1).cpu()))
            targets.append(int(label))
            probs.append(torch.softmax(logits.detach(), dim=1).cpu().numpy())
            self._collect_attention(attentions, slide_id, attn)
        return EpochOutputs(
            loss=float(np.mean(losses) if losses else 0.0),
            targets=targets,
            preds=preds,
            probs=probs,
            attentions=attentions,
        )

    def eval_epoch(self, loader: DataLoader) -> EpochOutputs:
        self.model.eval()
        losses: List[float] = []
        preds: List[int] = []
        targets: List[int] = []
        probs: List[np.ndarray] = []
        attentions: Dict[str, Dict[int, torch.Tensor]] = {}
        progress = tqdm(loader, desc="eval", leave=False)
        with torch.no_grad():
            for features, label, _, slide_id in progress:
                label_tensor = torch.tensor([label], device=self.device)
                bag_input = self._prepare_input(features)
                with autocast("cuda", enabled=self.use_amp):
                    logits, attn = self._forward(bag_input)
                    loss = self.criterion(logits, label_tensor)
                losses.append(float(loss.item()))
                preds.append(int(torch.argmax(logits, dim=1).cpu()))
                targets.append(int(label))
                probs.append(torch.softmax(logits, dim=1).cpu().numpy())
                self._collect_attention(attentions, slide_id, attn)
        return EpochOutputs(
            loss=float(np.mean(losses) if losses else 0.0),
            targets=targets,
            preds=preds,
            probs=probs,
            attentions=attentions,
        )

    def _prepare_input(
        self, features: Dict[int, torch.Tensor]
    ) -> Dict[int, torch.Tensor] | torch.Tensor:
        if self.model_name == "hiermil":
            return {mag: feats.to(self.device) for mag, feats in features.items()}
        mag = self.primary_mag if self.primary_mag in features else next(iter(features))
        return features[mag].to(self.device)

    def _forward(self, inputs):
        return self.model(inputs)

    def _collect_attention(
        self,
        storage: Dict[str, Dict[int, torch.Tensor]],
        slide_id: str,
        attn,
    ) -> None:
        if self.model_name == "hiermil":
            storage[slide_id] = {mag: weights.cpu() for mag, weights in attn.items()}
        else:
            storage[slide_id] = {self.primary_mag: attn.detach().cpu()}


class GraphTrainer:
    """Train graph neural networks on slide graphs."""

    def __init__(
        self,
        model_name: str,
        in_dim: int,
        n_classes: int,
        device: torch.device,
        lr: float,
        class_weights: torch.Tensor | None,
    ) -> None:
        hidden = max(128, in_dim // 2)
        if model_name == "gcn":
            self.model = GCNHead(in_dim, hidden, n_classes)
        else:
            self.model = GATHead(in_dim, hidden, n_classes)
        self.model.to(device)
        weight = class_weights.to(device) if class_weights is not None else None
        self.criterion = nn.CrossEntropyLoss(weight=weight)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.device = device

    def train_epoch(self, loader: GeoDataLoader) -> EpochOutputs:
        self.model.train()
        losses: List[float] = []
        preds: List[int] = []
        targets: List[int] = []
        probs: List[np.ndarray] = []
        progress = tqdm(loader, desc="train-gnn", leave=False)
        for batch in progress:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(batch)
            loss = self.criterion(logits, batch.y)
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
            preds.extend(torch.argmax(logits.detach(), dim=1).cpu().tolist())
            targets.extend(batch.y.cpu().tolist())
            probs.extend(torch.softmax(logits.detach(), dim=1).cpu().numpy())
        return EpochOutputs(
            loss=float(np.mean(losses) if losses else 0.0),
            targets=targets,
            preds=preds,
            probs=probs,
            attentions={},
        )

    def eval_epoch(self, loader: GeoDataLoader) -> EpochOutputs:
        self.model.eval()
        losses: List[float] = []
        preds: List[int] = []
        targets: List[int] = []
        probs: List[np.ndarray] = []
        progress = tqdm(loader, desc="eval-gnn", leave=False)
        with torch.no_grad():
            for batch in progress:
                batch = batch.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, batch.y)
                losses.append(float(loss.item()))
                preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                targets.extend(batch.y.cpu().tolist())
                probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
        return EpochOutputs(
            loss=float(np.mean(losses) if losses else 0.0),
            targets=targets,
            preds=preds,
            probs=probs,
            attentions={},
        )


from annotations import PolygonAnn


class CrossValRunner:
    """Coordinate cross-validation for MIL and GNN models."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.trainer.device)
        self.reader = WSIReader()
        self.parser = AnnotationParser()
        self.tiler = Tiler(
            reader=self.reader,
            patch_size=cfg.data.patch_size,
            magnifications=cfg.data.magnifications,
        )
        self.cfg.data.csv_dir.mkdir(parents=True, exist_ok=True)

    def prepare_patch_csvs(self) -> Dict[str, int]:
        annotations = self._load_annotations()  # stem -> List[PolygonAnn]
        self.parser.class_map.setdefault("background", 0)

        slides = sorted(
            list(self.cfg.data.slides.glob("*.svs"))
            + list(self.cfg.data.slides.glob("*.ndpi"))
        )
        if not slides:
            logger.error(
                f"No WSI files found under {self.cfg.data.slides}. "
                "Populate the directory or update --data-root."
            )
            return self.parser.class_map

        for slide_path in slides:
            csv_path = self.cfg.data.csv_dir / f"{slide_path.stem}.csv"
            if csv_path.exists():
                logger.debug(f"CSV already exists for {slide_path.stem}")
                continue
            ann = annotations.get(slide_path.stem, [])  # <— по stem
            patches = self.tiler.tile_slide(slide_path, ann)
            self._write_csv(csv_path, patches)

        return self.parser.class_map

    def _load_annotations(self) -> Dict[str, List[PolygonAnn]]:
        """Load annotations as mapping: slide_stem -> list of polygons."""
        ann_dir: Path = self.cfg.data.annotations
        mapping: Dict[str, List[PolygonAnn]] = {}
        for json_path in sorted(ann_dir.glob("*.json")):
            polys = self.parser.parse_file(json_path)
            stem = json_path.stem  # e.g. 'train_01'
            mapping[stem] = polys
        self.parser.class_map.setdefault("background", 0)
        return mapping

    def _write_csv(self, path: Path, patches: List[PatchRecord]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "slide_id",
                    "slide_path",
                    "magnification",
                    "level",
                    "x",
                    "y",
                    "label",
                    "patch_size",
                ]
            )
            for patch in patches:
                writer.writerow(
                    [
                        patch.slide_id,
                        patch.slide_path,
                        patch.magnification,
                        patch.level,
                        patch.x,
                        patch.y,
                        patch.label,
                        self.tiler.patch_size,
                    ]
                )
        logger.info(f"Wrote patch metadata to {path}")

    def run(self) -> None:
        class_map = self.prepare_patch_csvs()
        _, eval_tfm = build_transforms(
            self.cfg.augmentation,
            self.cfg.data.patch_size,
            self.cfg.stain_normalization,
        )
        csv_files = sorted(self.cfg.data.csv_dir.glob("*.csv"))
        dataset = PatchDataset(
            csv_files,
            self.reader,
            eval_tfm,
            class_map,
            self.cfg.data.slides,
        )
        if len(dataset) == 0:
            logger.error(
                f"No patch metadata found. Ensure tiling has produced CSV files in "
                f"{self.cfg.data.csv_dir}"
            )
            return
        encoder_cfg = EncoderConfig(
            name=self.cfg.encoder,
            img_size=self.cfg.data.patch_size,
        )
        encoder, feat_dim = build_encoder(encoder_cfg)
        feature_loader = DataLoader(
            dataset,
            batch_size=self.cfg.trainer.batch_size,
            shuffle=False,
            num_workers=self.cfg.trainer.num_workers,
            pin_memory=True,
        )
        extractor = FeatureExtractor(
            encoder, self.device, self.cfg.trainer.mixed_precision
        )
        result = extractor.extract(feature_loader)
        bag_dict = extractor.to_bags(result)
        if not bag_dict:
            logger.error(
                "Feature extraction yielded no bags. Check dataset labels and tiling outputs."
            )
            return
        if self.cfg.model in {"gcn", "gat"}:
            self._run_graph_cv(bag_dict, class_map)
        else:
            self._run_mil_cv(bag_dict, class_map, feat_dim)

    def _run_mil_cv(
        self,
        bag_dict: Dict[str, Bag],
        class_map: Dict[str, int],
        feat_dim: int,
    ) -> None:
        bags = list(bag_dict.values())
        if len(bags) < 2:
            logger.error(
                f"At least two bags are required for cross-validation, found {len(bags)}."
            )
            return
        labels = np.array([bag.label for bag in bags])
        splitter = self._choose_splitter(labels)
        n_classes = max(len(class_map), int(labels.max()) + 1)
        summary_rows: List[Dict[str, object]] = []
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            train_bags = [bags[i] for i in train_idx]
            val_bags = [bags[i] for i in val_idx]
            if self.cfg.trainer.oversample:
                train_bags = oversample_bags(train_bags)
            class_weights = (
                compute_class_weights(train_bags)
                if self.cfg.trainer.class_weighting
                else None
            )
            trainer = MILTrainer(
                model_name=self.cfg.model,
                feature_dim=feat_dim,
                n_classes=n_classes,
                magnifications=self.cfg.data.magnifications,
                device=self.device,
                lr=self.cfg.trainer.lr,
                class_weights=class_weights,
                use_amp=self.cfg.trainer.mixed_precision,
            )
            train_loader = DataLoader(
                BagDataset(train_bags),
                batch_size=1,
                shuffle=True,
                collate_fn=lambda x: x[0],
            )
            val_loader = DataLoader(
                BagDataset(val_bags),
                batch_size=1,
                shuffle=False,
                collate_fn=lambda x: x[0],
            )
            history = self._fit_fold(
                trainer,
                train_loader,
                val_loader,
                fold,
                class_map,
                bag_dict,
            )
            summary_rows.append(history)
        self._write_summary(summary_rows)

    def _run_graph_cv(
        self, bag_dict: Dict[str, Bag], class_map: Dict[str, int]
    ) -> None:
        graphs = self._build_graphs(bag_dict)
        items = list(graphs.items())
        if len(items) < 2:
            logger.error(
                "At least two graphs are required for cross-validation, found {}.",
                len(items),
            )
            return
        labels = np.array([data.y.item() for _, data in items])
        splitter = self._choose_splitter(labels)
        n_classes = max(len(class_map), int(labels.max()) + 1)
        summary_rows: List[Dict[str, object]] = []
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            train_graphs = [items[i][1] for i in train_idx]
            val_graphs = [items[i][1] for i in val_idx]
            class_weights = None
            if self.cfg.trainer.class_weighting:
                train_labels = torch.cat([g.y.view(-1) for g in train_graphs])
                classes, counts = train_labels.unique(return_counts=True)
                weights = counts.float().reciprocal()
                weights = weights / weights.sum() * len(classes)
                full = torch.ones(n_classes, dtype=torch.float32)
                full[classes] = weights
                class_weights = full
            trainer = GraphTrainer(
                model_name=self.cfg.model,
                in_dim=train_graphs[0].num_features,
                n_classes=n_classes,
                device=self.device,
                lr=self.cfg.trainer.lr,
                class_weights=class_weights,
            )
            train_loader = GeoDataLoader(train_graphs, batch_size=4, shuffle=True)
            val_loader = GeoDataLoader(val_graphs, batch_size=4, shuffle=False)
            history = self._fit_graph_fold(
                trainer, train_loader, val_loader, fold, class_map
            )
            summary_rows.append(history)
        self._write_summary(summary_rows)

    def _fit_fold(
        self,
        trainer: MILTrainer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        fold: int,
        class_map: Dict[str, int],
        bag_lookup: Dict[str, Bag],
    ) -> Dict[str, object]:
        losses = {"train": [], "val": []}
        metrics_history = {"f1_macro": [], "balanced_accuracy": [], "roc_auc": []}
        best_f1 = -1.0
        best_attn: Dict[str, Dict[int, torch.Tensor]] = {}
        best_preds: List[int] = []
        best_targets: List[int] = []
        for epoch in range(1, self.cfg.trainer.epochs + 1):
            train_out = trainer.train_epoch(train_loader)
            val_out = trainer.eval_epoch(val_loader)
            losses["train"].append(train_out.loss)
            losses["val"].append(val_out.loss)
            y_true = np.array(val_out.targets)
            y_pred = np.array(val_out.preds)
            y_prob = np.vstack(val_out.probs) if val_out.probs else None
            metric = compute_metrics(
                y_true, y_pred, y_prob, list(range(len(class_map)))
            )
            metrics_history["f1_macro"].append(metric.f1_macro)
            metrics_history["balanced_accuracy"].append(metric.balanced_accuracy)
            roc = metric.roc_auc if metric.roc_auc is not None else np.nan
            metrics_history["roc_auc"].append(roc)
            logger.info(
                f"Fold {fold} Epoch {epoch} | "
                f"train_loss={train_out.loss:.4f} val_loss={val_out.loss:.4f} "
                f"f1_macro={metric.f1_macro:.4f}"
            )
            if metric.f1_macro > best_f1:
                best_f1 = metric.f1_macro
                best_attn = val_out.attentions
                best_preds = y_pred.tolist()
                best_targets = y_true.tolist()
        fold_dir = self.cfg.output / f"fold_{fold}"
        plot_losses(losses, fold_dir / "loss.png")
        plot_metrics(metrics_history, fold_dir / "metrics.png")
        labels = sorted(class_map.keys(), key=lambda k: class_map[k])
        confusion_matrix_png(
            np.array(best_targets),
            np.array(best_preds),
            labels,
            fold_dir / "confusion.png",
        )
        for slide_id, attn in best_attn.items():
            bag = bag_lookup[slide_id]
            save_attention_weights(attn, bag, fold_dir / f"attention_{slide_id}.csv")
        return {
            "fold": fold,
            "f1_macro": best_f1,
            "balanced_accuracy": float(np.mean(metrics_history["balanced_accuracy"])),
            "roc_auc": float(
                np.mean([m for m in metrics_history["roc_auc"] if m is not None])
                if any(m is not None for m in metrics_history["roc_auc"])
                else np.nan
            ),
            "config": f"{self.cfg.model}_{self.cfg.encoder}_{self.cfg.augmentation}",
        }

    def _fit_graph_fold(
        self,
        trainer,
        train_loader: GeoDataLoader,
        val_loader: GeoDataLoader,
        fold: int,
        class_map: Dict[str, int],
    ) -> Dict[str, object]:
        losses = {"train": [], "val": []}
        metrics_history = {"f1_macro": [], "balanced_accuracy": [], "roc_auc": []}
        for epoch in range(1, self.cfg.trainer.epochs + 1):
            train_out = trainer.train_epoch(train_loader)
            val_out = trainer.eval_epoch(val_loader)
            losses["train"].append(train_out.loss)
            losses["val"].append(val_out.loss)
            y_true = np.array(val_out.targets)
            y_pred = np.array(val_out.preds)
            y_prob = np.vstack(val_out.probs) if val_out.probs else None
            metric = compute_metrics(
                y_true, y_pred, y_prob, list(range(len(class_map)))
            )
            metrics_history["f1_macro"].append(metric.f1_macro)
            metrics_history["balanced_accuracy"].append(metric.balanced_accuracy)
            roc = metric.roc_auc if metric.roc_auc is not None else np.nan
            metrics_history["roc_auc"].append(roc)
            logger.info(
                f"Fold {fold} Epoch {epoch} | "
                f"train_loss={train_out.loss:.4f} val_loss={val_out.loss:.4f} "
                f"f1_macro={metric.f1_macro:.4f}"
            )
        fold_dir = self.cfg.output / f"fold_{fold}"
        plot_losses(losses, fold_dir / "loss.png")
        plot_metrics(metrics_history, fold_dir / "metrics.png")
        labels = sorted(class_map.keys(), key=lambda k: class_map[k])
        confusion_matrix_png(
            np.array(val_out.targets),
            np.array(val_out.preds),
            labels,
            fold_dir / "confusion.png",
        )
        return {
            "fold": fold,
            "f1_macro": float(np.mean(metrics_history["f1_macro"])),
            "balanced_accuracy": float(np.mean(metrics_history["balanced_accuracy"])),
            "roc_auc": float(
                np.mean([m for m in metrics_history["roc_auc"] if m is not None])
                if any(m is not None for m in metrics_history["roc_auc"])
                else np.nan
            ),
            "config": f"{self.cfg.model}_{self.cfg.encoder}_{self.cfg.augmentation}",
        }

    def _build_graphs(self, bag_dict: Dict[str, Bag]) -> Dict[str, Data]:
        builder = GraphBuilder(GraphConfig(mode="knn", k=8))
        graphs = {}
        primary_mag = max(self.cfg.data.magnifications)
        for slide_id, bag in bag_dict.items():
            if primary_mag not in bag.features:
                continue
            graphs[slide_id] = builder.build(bag, primary_mag)
        return graphs

    def _write_summary(self, rows: List[Dict[str, object]]) -> None:
        out_path = self.cfg.output / "summary.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Saved summary metrics to {out_path}")

    def _choose_splitter(self, labels: np.ndarray):
        from sklearn.model_selection import KFold, StratifiedKFold

        unique, counts = np.unique(labels, return_counts=True)
        min_count = counts.min()
        n_splits = min(self.cfg.trainer.k_folds, int(min_count))
        if n_splits >= 2 and len(unique) > 1:
            logger.info(f"Using StratifiedKFold with {n_splits} splits")
            return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        logger.warning(f"Falling back to KFold (limited class counts)")
        return KFold(n_splits=min(2, len(labels)), shuffle=True, random_state=42)


def _build_runner_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("runner")
    parser.add_argument("--slides", type=Path, required=True)
    parser.add_argument("--annos", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--levels", type=int, nargs="+")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int)
    return parser


def _resolve_cli_magnifications(
    slides: List[Path], levels, reader: WSIReader
) -> List[int]:
    if not levels:
        return [40, 10]
    ref_slide = slides[0]
    info = reader.info(ref_slide)
    base = info.objective_power or 40.0
    mags: List[int] = []
    for level in levels:
        if level >= info.levels:
            raise ValueError(f"Level {level} is not available for {ref_slide.name}")
        downsample = info.level_downsamples[level]
        mags.append(int(round(base / downsample)))
    return mags


def main() -> None:
    parser = _build_runner_parser()
    args = parser.parse_args()
    slides_dir = args.slides.resolve()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    slides = sorted(list(slides_dir.glob("*.svs")) + list(slides_dir.glob("*.ndpi")))
    if not slides:
        logger.error(f"No WSI files found under {slides_dir}")
        return
    reader = WSIReader()
    try:
        magnifications = _resolve_cli_magnifications(slides, args.levels, reader)
    except ValueError as exc:
        reader.close()
        logger.error(f"{exc}")
        return
    reader.close()
    ann_dir = args.annos.resolve() if args.annos else slides_dir.parent / "annotations"
    data_cfg = DataConfig(
        slides=slides_dir,
        annotations=ann_dir,
        cache=out_root / "cache",
        magnifications=magnifications,
        patch_size=args.patch_size,
        csv_dir=out_root / "patch_csvs",
    )
    trainer_cfg = TrainerConfig(
        batch_size=1,
        num_workers=0,
        lr=1e-4,
        epochs=1,
        k_folds=2,
        device="cpu",
        mixed_precision=False,
        oversample=False,
        class_weighting=False,
    )
    cfg = ExperimentConfig(
        data=data_cfg,
        trainer=trainer_cfg,
        model="abmil",
        encoder="resnet18",
        augmentation="none",
        stain_normalization=False,
        output=out_root,
        verbose="info",
    )
    runner = CrossValRunner(cfg)
    runner.tiler.stride = args.stride or args.patch_size
    runner.prepare_patch_csvs()
    logger.info(f"Patch CSVs saved to {cfg.data.csv_dir}")


if __name__ == "__main__":
    main()
