"""Extended classification metrics for histopathology MIL experiments.

Each metric is documented with the underlying mathematical definition to
facilitate reproducibility in research reports.

F1 score is defined as ``2·(precision·recall) / (precision + recall)``. The
balanced accuracy corresponds to the mean recall across classes. ROC-AUC denotes
"the area under the Receiver Operating Characteristic curve", while PR-AUC is
"the area under the Precision-Recall curve". The Matthews Correlation Coefficient
(MCC) is computed as ``(TP·TN − FP·FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))`` and
Cohen's κ measures inter-rater agreement beyond chance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


@dataclass
class MetricCurves:
    """Container for ROC/PR curve coordinates."""

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray
    auc: float

    def to_dict(self) -> Dict[str, np.ndarray | float]:
        return {
            "fpr": self.fpr,
            "tpr": self.tpr,
            "thresholds": self.thresholds,
            "auc": self.auc,
        }


@dataclass
class PRCurves:
    """Precision-Recall curve container."""

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    auc: float

    def to_dict(self) -> Dict[str, np.ndarray | float]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "thresholds": self.thresholds,
            "auc": self.auc,
        }


@dataclass
class ClassificationReport:
    """Full metric report for a single evaluation split/fold."""

    metrics: Dict[str, float]
    per_class: Dict[str, Dict[str, float]]
    roc: Dict[str, MetricCurves]
    pr: Dict[str, PRCurves]
    confusion: np.ndarray
    labels: Sequence[str]
    class_distribution: Dict[str, Dict[str, float]]

    def as_flat_dict(self) -> Dict[str, float]:
        """Flatten the report to a single-level mapping for CSV export."""

        flat: Dict[str, float] = {}
        for key, value in self.metrics.items():
            flat[key] = float(value)
        for label, stats in self.per_class.items():
            for metric_name, val in stats.items():
                flat[f"{label}_{metric_name}"] = float(val)
        return flat


def _compute_distribution(
    y_true: np.ndarray, class_names: Sequence[str]
) -> Dict[str, Dict[str, float]]:
    counts = np.bincount(y_true, minlength=len(class_names))
    total = counts.sum()
    distribution: Dict[str, Dict[str, float]] = {}
    for idx, name in enumerate(class_names):
        distribution[name] = {
            "count": float(counts[idx]),
            "fraction": float(counts[idx] / total) if total > 0 else 0.0,
        }
    return distribution


def _ensure_class_names(labels: Sequence[int] | Sequence[str]) -> List[str]:
    if not labels:
        raise ValueError("Labels must not be empty")
    first = labels[0]
    if isinstance(first, str):
        return list(labels)  # type: ignore[return-value]
    return [str(l) for l in labels]


def compute_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: np.ndarray | None,
    labels: Sequence[int] | Sequence[str],
) -> ClassificationReport:
    """Compute an exhaustive classification report for MIL models.

    Parameters
    ----------
    y_true:
        Ground-truth integer class labels.
    y_pred:
        Predicted integer class labels.
    y_prob:
        Optional probability estimates with shape ``[N, num_classes]``.
    labels:
        Class identifiers corresponding to the probability columns.

    Returns
    -------
    ClassificationReport
        A dataclass holding scalar metrics, per-class values, ROC/PR curves, and
        class distribution statistics.
    """

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    class_names = _ensure_class_names(labels)
    metrics: Dict[str, float] = {}
    metrics["accuracy"] = accuracy_score(y_true_arr, y_pred_arr)
    metrics["balanced_accuracy"] = balanced_accuracy_score(y_true_arr, y_pred_arr)
    metrics["f1_micro"] = f1_score(
        y_true_arr, y_pred_arr, average="micro", zero_division=0
    )
    metrics["f1_macro"] = f1_score(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    metrics["f1_weighted"] = f1_score(
        y_true_arr, y_pred_arr, average="weighted", zero_division=0
    )
    metrics["precision_macro"] = precision_score(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    metrics["precision_micro"] = precision_score(
        y_true_arr, y_pred_arr, average="micro", zero_division=0
    )
    metrics["recall_macro"] = recall_score(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )
    metrics["recall_micro"] = recall_score(
        y_true_arr, y_pred_arr, average="micro", zero_division=0
    )
    metrics["mcc"] = matthews_corrcoef(y_true_arr, y_pred_arr)
    metrics["cohen_kappa"] = cohen_kappa_score(y_true_arr, y_pred_arr)
    metrics["log_loss"] = (
        log_loss(y_true_arr, y_prob) if y_prob is not None else float("nan")
    )

    per_class: Dict[str, Dict[str, float]] = {}
    precision_pc = precision_score(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    recall_pc = recall_score(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    f1_pc = f1_score(
        y_true_arr,
        y_pred_arr,
        average=None,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    for idx, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision_pc[idx]),
            "recall": float(recall_pc[idx]),
            "f1": float(f1_pc[idx]),
        }

    roc_curves: Dict[str, MetricCurves] = {}
    pr_curves: Dict[str, PRCurves] = {}
    roc_macro = float("nan")
    pr_macro = float("nan")
    if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == len(class_names):
        y_true_bin = label_binarize(y_true_arr, classes=list(range(len(class_names))))
        # Handle binary case where label_binarize returns shape (N,1)
        if y_true_bin.shape[1] == 1:
            y_true_bin = np.hstack([1 - y_true_bin, y_true_bin])
        try:
            roc_macro = roc_auc_score(y_true_arr, y_prob, multi_class="ovr")
            metrics["roc_auc_macro"] = roc_macro
        except ValueError:
            metrics["roc_auc_macro"] = float("nan")
        try:
            pr_macro = average_precision_score(y_true_arr, y_prob, average="macro")
            metrics["pr_auc_macro"] = pr_macro
        except ValueError:
            metrics["pr_auc_macro"] = float("nan")

        for idx, name in enumerate(class_names):
            try:
                fpr, tpr, thresholds = roc_curve(y_true_bin[:, idx], y_prob[:, idx])
                auc_val = roc_auc_score(y_true_bin[:, idx], y_prob[:, idx])
                roc_curves[name] = MetricCurves(
                    fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc_val
                )
            except ValueError:
                roc_curves[name] = MetricCurves(
                    fpr=np.asarray([0.0, 1.0]),
                    tpr=np.asarray([0.0, 1.0]),
                    thresholds=np.asarray([1.0]),
                    auc=float("nan"),
                )
            try:
                precision, recall, thresholds_pr = precision_recall_curve(
                    y_true_bin[:, idx], y_prob[:, idx]
                )
                auc_pr = average_precision_score(y_true_bin[:, idx], y_prob[:, idx])
                pr_curves[name] = PRCurves(
                    precision=precision,
                    recall=recall,
                    thresholds=thresholds_pr,
                    auc=auc_pr,
                )
            except ValueError:
                pr_curves[name] = PRCurves(
                    precision=np.asarray([1.0]),
                    recall=np.asarray([0.0]),
                    thresholds=np.asarray([1.0]),
                    auc=float("nan"),
                )

    confusion = confusion_matrix(
        y_true_arr, y_pred_arr, labels=list(range(len(class_names)))
    )
    distribution = _compute_distribution(y_true_arr, class_names)
    metrics.setdefault("roc_auc_macro", float("nan"))
    metrics.setdefault("pr_auc_macro", float("nan"))
    report = ClassificationReport(
        metrics=metrics,
        per_class=per_class,
        roc=roc_curves,
        pr=pr_curves,
        confusion=confusion,
        labels=class_names,
        class_distribution=distribution,
    )
    logger.debug("Computed classification metrics: {}", report.metrics)
    return report


@dataclass
class FoldSummary:
    """Container holding per-fold metrics and aggregate statistics."""

    fold_metrics: Dict[int, ClassificationReport] = field(default_factory=dict)
    fold_metadata: Dict[int, Dict[str, object]] = field(default_factory=dict)

    def add_fold(
        self,
        fold: int,
        report: ClassificationReport,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        self.fold_metrics[fold] = report
        if metadata:
            self.fold_metadata[fold] = dict(metadata)

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for fold, report in self.fold_metrics.items():
            row = {"fold": fold}
            row.update(self.fold_metadata.get(fold, {}))
            row.update(report.as_flat_dict())
            rows.append(row)
        return pd.DataFrame(rows)

    def aggregate(self) -> pd.DataFrame:
        df = self.to_dataframe()
        metrics_cols = [c for c in df.columns if c != "fold"]
        agg = df[metrics_cols].agg(["mean", "std"])
        agg.loc["mean_std"] = [
            f"{m:.4f}±{s:.4f}" for m, s in zip(agg.loc["mean"], agg.loc["std"])
        ]
        agg.insert(0, "statistic", agg.index)
        return agg

    def export(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe()
        numeric_cols = [
            c
            for c in df.columns
            if c != "fold" and pd.api.types.is_numeric_dtype(df[c])
        ]
        means = df[numeric_cols].mean(numeric_only=True)
        stds = df[numeric_cols].std(numeric_only=True)
        combined = df.copy()
        combined.loc[len(combined)] = {"fold": "mean", **means.to_dict()}
        combined.loc[len(combined)] = {"fold": "std", **stds.to_dict()}
        combined.loc[len(combined)] = {
            "fold": "mean±std",
            **{
                col: f"{means.get(col, np.nan):.4f}±{stds.get(col, np.nan):.4f}"
                for col in numeric_cols
            },
        }
        combined.to_csv(path, index=False)
        logger.info("Saved per-fold metrics and aggregate statistics to {}", path)


def summarize_class_distribution(
    reports: Mapping[int, ClassificationReport],
) -> pd.DataFrame:
    """Create a dataframe summarizing class counts and fractions per fold."""

    rows = []
    for fold, report in reports.items():
        for cls, info in report.class_distribution.items():
            rows.append(
                {
                    "fold": fold,
                    "class": cls,
                    "count": info["count"],
                    "fraction": info["fraction"],
                }
            )
    return pd.DataFrame(rows)


def paired_t_test(metric_a: Sequence[float], metric_b: Sequence[float]) -> float:
    """Compute a paired t-test between two metric sequences."""

    from scipy.stats import ttest_rel

    statistic, p_value = ttest_rel(metric_a, metric_b, nan_policy="omit")
    logger.debug("Paired t-test computed", statistic=statistic, p_value=p_value)
    return float(p_value)


@dataclass
class Report:
    f1_macro: float
    f1_micro: float
    precision_macro: float
    recall_macro: float
    balanced_accuracy: float
    roc_auc: Optional[float]
    pr_auc: Optional[float]
    log_loss: Optional[float]
    cohen_kappa: Optional[float]
    mcc: Optional[float]
    confusion: np.ndarray


def compute_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_prob: Optional[np.ndarray],
    class_ids: Sequence[int],
) -> Report:
    """Safe metrics computation even if a fold has a single class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(class_ids)

    # базовые метрики (не падают при 1 классе)
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_micro = f1_score(y_true, y_pred, labels=labels, average="micro", zero_division=0)
    precision_macro = precision_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    recall_macro = recall_score(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    # по умолчанию метрики, требующие ≥2 классов, ставим nan/None
    roc_auc = None
    pr_auc = None
    xent = None

    # roc/pr считаем только если в y_true ≥2 уникальных класса и есть вероятности
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            roc_auc_val = roc_auc_score(
                y_true, y_prob, multi_class="ovr", labels=labels
            )
            roc_auc = float(roc_auc_val)
        except Exception:
            roc_auc = None
        try:
            pr_list = []
            for c in labels:
                y_bin = (y_true == c).astype(int)
                pr_list.append(average_precision_score(y_bin, y_prob[:, c]))
            pr_auc = float(np.mean(pr_list))
        except Exception:
            pr_auc = None

    # log-loss можно считать и для 1 класса, если явно передать labels,
    # но sklearn всё равно ругается — поэтому также оборачиваем в try/except.
    if y_prob is not None:
        try:
            xent_val = log_loss(y_true, y_prob, labels=labels)
            xent = float(xent_val)
        except Exception:
            xent = None

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    try:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
    except Exception:
        kappa = None
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        mcc = None

    return Report(
        f1_macro=float(f1_macro),
        f1_micro=float(f1_micro),
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        balanced_accuracy=float(bal_acc),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        log_loss=xent,
        cohen_kappa=kappa,
        mcc=mcc,
        confusion=cm,
    )
