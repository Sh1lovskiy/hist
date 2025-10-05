"""Aggregate experiment summaries and plot side-by-side comparisons."""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def load_summary(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def aggregate_runs(run_dirs: List[Path]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for run in run_dirs:
        summary_path = run / "summary.csv"
        if not summary_path.exists():
            continue
        rows.extend(load_summary(summary_path))
    return rows


def plot_comparison(rows: List[Dict[str, str]], metric: str, path: Path) -> None:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        label = row.get("config", row.get("fold", "run"))
        grouped.setdefault(label, []).append(float(row[metric]))
    labels = list(grouped.keys())
    values = [sum(v) / len(v) for v in grouped.values()]
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    ax.bar(labels, values, alpha=0.8)
    ax.set_ylabel(metric)
    ax.set_xlabel("Run")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser("compare_experiments")
    parser.add_argument("runs", type=Path, nargs="+", help="Run directories")
    parser.add_argument("--metric", default="f1_macro")
    parser.add_argument("--output", type=Path, default=Path("runs/comparison.png"))
    args = parser.parse_args()
    rows = aggregate_runs(args.runs)
    plot_comparison(rows, args.metric, args.output)


if __name__ == "__main__":
    main()
