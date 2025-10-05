"""Publication-ready plotting utilities for MIL experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import math
from sklearn.manifold import TSNE

try:  # pragma: no cover - optional dependency
    import umap  # type: ignore
except Exception:  # pragma: no cover
    umap = None

sns.set_theme(context="talk", style="whitegrid")


def _setup_figure(width: float = 8.0, height: float = 5.0) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(width, height), dpi=300)
    return fig, ax


def plot_losses(history: Mapping[str, Sequence[float]], path: Path) -> None:
    """Plot train and validation loss curves."""

    fig, ax = _setup_figure()
    for split, values in history.items():
        ax.plot(values, label=split, linewidth=2.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics: Mapping[str, Sequence[float]], path: Path) -> None:
    """Plot multiple metrics trajectories on a single axes."""

    fig, ax = _setup_figure()
    for name, values in metrics.items():
        ax.plot(values, label=name, linewidth=2.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    all_values = []
    for series in metrics.values():
        for value in series:
            if value is None:
                continue
            val = float(value)
            if not math.isnan(val) and not math.isinf(val):
                all_values.append(val)
    if all_values:
        min_v = min(all_values)
        max_v = max(all_values)
        margin = max((max_v - min_v) * 0.1, 0.05)
        ax.set_ylim(min_v - margin, max_v + margin)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    normalize: bool = True,
) -> None:
    """Plot an annotated confusion matrix with optional normalization."""

    if normalize:
        with np.errstate(all="ignore"):
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
        matrix = np.nan_to_num(matrix)
    fig, ax = _setup_figure(6.0, 6.0)[0:2]
    cmap = sns.color_palette("magma", as_cmap=True)
    im = ax.imshow(matrix, cmap=cmap)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    thresh = matrix.max() / 2 if matrix.size > 0 else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text_color = "white" if value > thresh else "black"
            ax.text(j, i, f"{value:.2f}" if normalize else int(value), ha="center", va="center", color=text_color)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_multi_class_curves(
    curves: Mapping[str, Mapping[str, np.ndarray | float]],
    path: Path,
    curve_type: str,
    title: str,
) -> None:
    """Generic helper to visualise ROC or PR curves for multiple classes."""

    fig, ax = _setup_figure()
    for label, info in curves.items():
        if curve_type == "roc":
            x = np.asarray(info["fpr"])
            y = np.asarray(info["tpr"])
        else:
            x = np.asarray(info["recall"])
            y = np.asarray(info["precision"])
        auc_value = float(info.get("auc", np.nan))
        ax.plot(x, y, linewidth=2.0, label=f"{label} (AUC={auc_value:.3f})")
    if curve_type == "roc":
        ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1.2)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
    else:
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_embeddings(
    embeddings: np.ndarray,
    labels: Sequence[str],
    predictions: Sequence[str] | None,
    path: Path,
    method: str = "tsne",
    random_state: int = 42,
) -> None:
    """Visualise feature embeddings via t-SNE or UMAP."""

    if method.lower() == "umap":
        if umap is None:
            raise RuntimeError("umap-learn is not installed; cannot compute UMAP embeddings")
        reducer = umap.UMAP(n_components=2, random_state=random_state, metric="cosine")
        coords = reducer.fit_transform(embeddings)
    else:
        reducer = TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")
        coords = reducer.fit_transform(embeddings)
    fig, ax = _setup_figure()
    palette = sns.color_palette("tab10", len(set(labels)))
    color_map = {cls: palette[idx % len(palette)] for idx, cls in enumerate(sorted(set(labels)))}
    for cls in sorted(set(labels)):
        mask = np.array(labels) == cls
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[color_map[cls]],
            label=cls,
            s=40,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )
    if predictions is not None:
        mismatched = np.array(labels) != np.array(predictions)
        if mismatched.any():
            ax.scatter(
                coords[mismatched, 0],
                coords[mismatched, 1],
                facecolors="none",
                edgecolors="red",
                s=120,
                linewidths=1.5,
                label="Misclassified",
            )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{method.upper()} feature projection")
    ax.legend(loc="best", frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_radar(metrics: Mapping[str, Sequence[float]], labels: Sequence[str], path: Path) -> None:
    """Plot a radar chart for model comparison."""

    num_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(7, 7), dpi=300)
    ax = plt.subplot(111, polar=True)
    for name, values in metrics.items():
        data = list(values)
        data += data[:1]
        ax.plot(angles, data, linewidth=2, label=name)
        ax.fill(angles, data, alpha=0.1)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_bar(metrics: Mapping[str, float], ylabel: str, path: Path) -> None:
    """Render a bar plot comparing runs for a single metric."""

    fig, ax = _setup_figure()
    labels = list(metrics.keys())
    values = [metrics[label] for label in labels]
    sns.barplot(x=labels, y=values, ax=ax, palette="viridis")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Model")
    ax.set_ylim(0.0, 1.0)
    for index, value in enumerate(values):
        ax.text(index, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def overlay_attention_heatmap(
    thumbnail: np.ndarray,
    heatmap: np.ndarray,
    path: Path,
    alpha: float = 0.6,
    cmap: str = "magma",
) -> None:
    """Overlay an attention heatmap on top of a thumbnail image."""

    norm_heatmap = heatmap.copy()
    if norm_heatmap.max() > 0:
        norm_heatmap = norm_heatmap / norm_heatmap.max()
    colored = plt.get_cmap(cmap)(norm_heatmap)[..., :3]
    overlay = (1 - alpha) * (thumbnail / 255.0) + alpha * colored
    overlay = np.clip(overlay, 0.0, 1.0)
    fig, ax = _setup_figure(6.0, 6.0)
    ax.imshow(overlay)
    ax.axis("off")
    fig.tight_layout(pad=0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
