"""Utilities to aggregate and compare multiple MIL experiment runs."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import pandas as pd
from loguru import logger

from plotting import plot_bar, plot_radar


def load_summary(run_dir: Path) -> pd.DataFrame:
    """Load a summary.csv file and annotate it with the run name."""

    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found in {run_dir}")
    df = pd.read_csv(summary_path)
    df.insert(0, "run", run_dir.name)
    if "fold" in df.columns:
        numeric_mask = pd.to_numeric(df["fold"], errors="coerce").notna()
        df = df[numeric_mask]
    return df


def aggregate_runs(run_dirs: Sequence[Path]) -> pd.DataFrame:
    """Concatenate summary tables from several runs."""

    frames = []
    for run_dir in run_dirs:
        try:
            frames.append(load_summary(run_dir))
        except FileNotFoundError as exc:
            logger.warning(str(exc))
    if not frames:
        raise RuntimeError("No valid runs were provided")
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded {} summary rows", len(combined))
    return combined


def compute_statistics(df: pd.DataFrame, metrics: Sequence[str]) -> pd.DataFrame:
    """Compute mean ± std for each metric per run."""

    grouped = df.groupby("run")
    stats = grouped[metrics].agg(["mean", "std"])
    formatted = pd.DataFrame(index=stats.index)
    for metric in metrics:
        mean_series = stats[(metric, "mean")].fillna(0.0)
        std_series = stats[(metric, "std")].fillna(0.0)
        formatted[metric] = mean_series.map(lambda m: f"{m:.4f}") + "±" + std_series.map(
            lambda s: f"{s:.4f}"
        )
    formatted.insert(0, "run", formatted.index)
    formatted.reset_index(drop=True, inplace=True)
    return formatted


def rank_models(stats: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank models using the specified metric (higher is better)."""

    metric_values = stats[metric].str.split("±").str[0].astype(float)
    stats = stats.copy()
    stats["rank"] = metric_values.rank(ascending=False, method="min").astype(int)
    return stats.sort_values("rank")


def export_tables(formatted: pd.DataFrame, out_dir: Path) -> None:
    """Save CSV, Markdown and LaTeX tables summarising comparisons."""

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_comparison.csv"
    md_path = out_dir / "model_comparison.md"
    tex_path = out_dir / "model_comparison.tex"
    formatted.to_csv(csv_path, index=False)
    md_path.write_text(formatted.to_markdown(index=False))
    tex_path.write_text(formatted.to_latex(index=False))
    logger.info("Exported comparison tables to {}", out_dir)


def create_plots(df: pd.DataFrame, metrics: Sequence[str], out_dir: Path) -> None:
    """Generate radar and bar plots for model comparison."""

    mean_df = df.copy()
    radar_data = {}
    for _, row in mean_df.iterrows():
        radar_data[row["run"]] = [float(row[metric].split("±")[0]) for metric in metrics]
    plot_radar(radar_data, metrics, out_dir / "metrics_radar.png")
    for metric in metrics:
        metric_values = {row["run"]: float(row[metric].split("±")[0]) for _, row in mean_df.iterrows()}
        plot_bar(metric_values, metric, out_dir / f"{metric}_bar.png")


def parse_args() -> argparse.Namespace:
    description = "Aggregate summary.csv files across runs and generate reports."
    epilog = """Examples:\n  python -m compare_results \\n    --inputs runs/abmil_vit224 runs/transmil_vit224 runs/hiermil_vit224 \\\n    --out runs/comparison \\\n    --metrics f1_macro balanced_accuracy roc_auc_macro mcc\n"""
    parser = argparse.ArgumentParser(
        "compare_results",
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--inputs", type=Path, nargs="+", required=True, help="Experiment directories")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for reports")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["f1_macro", "balanced_accuracy", "roc_auc_macro", "mcc"],
        help="Metrics to aggregate",
    )
    parser.add_argument(
        "--ranking-metric",
        default="f1_macro",
        help="Metric used for ranking models (must be contained in --metrics)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = aggregate_runs(args.inputs)
    formatted = compute_statistics(df, args.metrics)
    ranked = rank_models(formatted, args.ranking_metric)
    export_tables(ranked, args.out)
    create_plots(ranked, args.metrics, args.out)
    logger.success("Comparison artefacts stored in {}", args.out)


if __name__ == "__main__":
    main()
