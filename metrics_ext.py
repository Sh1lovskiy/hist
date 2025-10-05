from __future__ import annotations
from typing import Dict, Tuple, List
import numpy as np
from sklearn.metrics import (
    f1_score,
    balanced_accuracy_score,
    roc_auc_score,
    average_precision_score,
)


def compute_cls_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None, labels: List[int]
) -> Dict[str, float]:
    """Classification metrics at slide-level."""
    out = {
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        # One-vs-rest ROC-AUC and PR-AUC (macro)
        try:
            roc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average="macro", labels=labels
            )
            prc = average_precision_score(
                np.eye(len(labels))[y_true], y_prob, average="macro"
            )
            out["auc_macro_ovr"] = float(roc)
            out["pr_auc_macro"] = float(prc)
        except Exception:
            pass
    return out


def iou_dice_from_patch_labels(
    y_true: np.ndarray, y_pred: np.ndarray, n_cls: int
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    IoU/Dice computed on patch grids derived from polygon masks:
    treat each patch as a pixel with class label.
    """
    iou_c, dice_c = {}, {}
    for c in range(n_cls):
        t = y_true == c
        p = y_pred == c
        inter = np.logical_and(t, p).sum()
        union = np.logical_or(t, p).sum()
        iou = inter / union if union > 0 else 0.0
        dice = (2 * inter) / (t.sum() + p.sum()) if (t.sum() + p.sum()) > 0 else 0.0
        iou_c[f"iou_{c}"] = float(iou)
        dice_c[f"dice_{c}"] = float(dice)

    iou_mean = float(np.mean(list(iou_c.values()))) if iou_c else 0.0
    dice_mean = float(np.mean(list(dice_c.values()))) if dice_c else 0.0
    iou_c["iou_mean"] = iou_mean
    dice_c["dice_mean"] = dice_mean
    return iou_c, dice_c


def compute_cls_metrics(y_true, y_pred, y_prob, labels_all):
    """
    Safe metrics: works even if y_true has <2 classes.
    If y_prob is for a reduced label set, pass labels_all=None.
    """
    out = {
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
    }
    # ROC-AUC/PR-AUC only if >=2 classes in y_true
    present = np.unique(y_true)
    if y_prob is not None and len(present) > 1:
        try:
            # y_prob is already over present classes (see remap below)
            roc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            prc = average_precision_score(
                np.eye(y_prob.shape[1])[y_true], y_prob, average="macro"
            )
            out["auc_macro_ovr"] = float(roc)
            out["pr_auc_macro"] = float(prc)
        except Exception:
            pass
    return out
