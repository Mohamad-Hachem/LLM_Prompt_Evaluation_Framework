"""Metrics and plots for prompt evaluation results."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from src.classifier import INVALID_LABEL, VALID_LABELS

LOGGER = logging.getLogger(__name__)

PROMPT_ORDER = ["zero_shot", "few_shot", "chain_of_thought", "role_structured"]


def compute_metrics(
    predictions_df: pd.DataFrame,
    output_path: str | Path = "results/metrics_summary.csv",
) -> pd.DataFrame:
    """Compute per-condition quality, latency, cost, and calibration metrics.

    Args:
        predictions_df: Raw prediction rows from ``run_full_evaluation``.
        output_path: CSV path where the metrics summary should be saved.

    Returns:
        One row per prompt strategy and model condition.

    Example:
        >>> rows = [{"prompt_strategy": "zero_shot", "model": "gpt-4o-mini", "true_label": "order_status", "predicted_label": "order_status", "latency_ms": 100, "cost_usd": 0.001}]
        >>> compute_metrics(pd.DataFrame(rows), output_path="metrics_tmp.csv")["accuracy"].iloc[0]
        1.0
    """

    _validate_prediction_columns(predictions_df)

    metric_rows: list[dict[str, float | str]] = []
    grouped = predictions_df.groupby(["prompt_strategy", "model"], sort=False, dropna=False)

    for (prompt_strategy, model), group in grouped:
        y_true = group["true_label"].astype(str)
        y_pred = group["predicted_label"].fillna(INVALID_LABEL).astype(str)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(VALID_LABELS),
            average="macro",
            zero_division=0,
        )
        per_class_precision, per_class_recall, _, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(VALID_LABELS),
            average=None,
            zero_division=0,
        )

        total_cost = pd.to_numeric(group["cost_usd"], errors="coerce").fillna(0).sum()
        n_predictions = len(group)
        ece = np.nan
        if prompt_strategy == "role_structured":
            ece = _expected_calibration_error(group)

        row: dict[str, float | str] = {
            "prompt_strategy": str(prompt_strategy),
            "model": str(model),
            "accuracy": float((y_true == y_pred).mean()),
            "macro_precision": float(precision),
            "macro_recall": float(recall),
            "macro_f1": float(f1),
        }
        for label, label_precision, label_recall in zip(
            VALID_LABELS,
            per_class_precision,
            per_class_recall,
            strict=True,
        ):
            row[f"precision_{label}"] = float(label_precision)
            row[f"recall_{label}"] = float(label_recall)

        row.update(
            {
                "mean_latency_ms": float(pd.to_numeric(group["latency_ms"], errors="coerce").mean()),
                "p95_latency_ms": float(pd.to_numeric(group["latency_ms"], errors="coerce").quantile(0.95)),
                "total_cost_usd": float(total_cost),
                "cost_per_1k_predictions": float((total_cost / n_predictions) * 1000),
                "invalid_rate": float((y_pred == INVALID_LABEL).mean()),
                "confidence_calibration_ece": float(ece) if not np.isnan(ece) else np.nan,
            }
        )
        metric_rows.append(row)

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df = _sort_metrics(metrics_df)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(output_path, index=False)
    LOGGER.info("Saved metrics summary to %s", output_path)
    return metrics_df


def plot_results(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, output_dir: str | Path) -> None:
    """Create the four portfolio plots from summary metrics and predictions.

    Args:
        metrics_df: Output of ``compute_metrics``.
        predictions_df: Raw predictions dataframe.
        output_dir: Results directory. Plots are written to ``output_dir/plots``.

    Example:
        >>> metrics = pd.DataFrame([{"prompt_strategy": "zero_shot", "model": "gpt-4o-mini", "accuracy": 1.0, "mean_latency_ms": 100, "cost_per_1k_predictions": 0.01, "macro_f1": 1.0}])
        >>> preds = pd.DataFrame([{"prompt_strategy": "zero_shot", "model": "gpt-4o-mini", "true_label": "order_status", "predicted_label": "order_status"}])
        >>> plot_results(metrics, preds, "results_tmp")
    """

    plots_dir = Path(output_dir) / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    _plot_accuracy_comparison(metrics_df, plots_dir / "accuracy_comparison.png")
    _plot_latency_vs_accuracy(metrics_df, plots_dir / "latency_vs_accuracy.png")
    _plot_confusion_matrix_best(metrics_df, predictions_df, plots_dir / "confusion_matrix_best.png")
    _plot_cost_vs_accuracy(metrics_df, plots_dir / "cost_vs_accuracy.png")
    LOGGER.info("Saved plots to %s", plots_dir)


def _validate_prediction_columns(predictions_df: pd.DataFrame) -> None:
    required_columns = {
        "prompt_strategy",
        "model",
        "true_label",
        "predicted_label",
        "latency_ms",
        "cost_usd",
    }
    missing = required_columns.difference(predictions_df.columns)
    if missing:
        raise ValueError(f"Predictions dataframe is missing columns: {sorted(missing)}")


def _expected_calibration_error(group: pd.DataFrame, bins: int = 10) -> float:
    confidences = pd.to_numeric(group.get("confidence"), errors="coerce")
    valid_mask = confidences.between(0, 1, inclusive="both")
    if not valid_mask.any():
        return np.nan

    confidences = confidences[valid_mask]
    correctness = (group.loc[valid_mask, "true_label"] == group.loc[valid_mask, "predicted_label"]).astype(float)
    bin_ids = np.minimum((confidences.to_numpy() * bins).astype(int), bins - 1)

    ece = 0.0
    total = len(confidences)
    for bin_id in range(bins):
        in_bin = bin_ids == bin_id
        if not in_bin.any():
            continue
        bin_confidence = float(confidences.to_numpy()[in_bin].mean())
        bin_accuracy = float(correctness.to_numpy()[in_bin].mean())
        ece += (in_bin.sum() / total) * abs(bin_accuracy - bin_confidence)
    return float(ece)


def _sort_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df
    sorted_df = metrics_df.copy()
    sorted_df["prompt_strategy"] = pd.Categorical(
        sorted_df["prompt_strategy"],
        categories=PROMPT_ORDER,
        ordered=True,
    )
    sorted_df = sorted_df.sort_values(["prompt_strategy", "model"]).reset_index(drop=True)
    sorted_df["prompt_strategy"] = sorted_df["prompt_strategy"].astype(str)
    return sorted_df


def _plot_accuracy_comparison(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_df = _sort_metrics(metrics_df)
    sns.barplot(data=plot_df, x="prompt_strategy", y="accuracy", hue="model", ax=ax)
    ax.set_title("Accuracy by Prompt Strategy and Model")
    ax.set_xlabel("Prompt strategy")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.2f", padding=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_latency_vs_accuracy(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = metrics_df.copy()
    sns.scatterplot(data=plot_df, x="mean_latency_ms", y="accuracy", hue="model", s=90, ax=ax)
    _annotate_conditions(ax, plot_df, "mean_latency_ms")
    _draw_pareto_frontier(ax, plot_df, x_col="mean_latency_ms", y_col="accuracy")
    ax.set_title("Latency vs Accuracy")
    ax.set_xlabel("Mean latency (ms)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_cost_vs_accuracy(metrics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    plot_df = metrics_df.copy()
    plot_df = plot_df[pd.to_numeric(plot_df["cost_per_1k_predictions"], errors="coerce") > 0]
    if plot_df.empty:
        ax.text(0.5, 0.5, "No positive cost data available", ha="center", va="center", transform=ax.transAxes)
    else:
        sns.scatterplot(data=plot_df, x="cost_per_1k_predictions", y="accuracy", hue="model", s=90, ax=ax)
        _annotate_conditions(ax, plot_df, "cost_per_1k_predictions")
        _draw_pareto_frontier(ax, plot_df, x_col="cost_per_1k_predictions", y_col="accuracy")
        ax.set_xscale("log")
    ax.set_title("Cost vs Accuracy")
    ax.set_xlabel("Cost per 1k predictions (USD, log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_confusion_matrix_best(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, output_path: Path) -> None:
    if metrics_df.empty:
        raise ValueError("Cannot plot best confusion matrix from an empty metrics dataframe.")

    best_row = metrics_df.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
    subset = predictions_df[
        (predictions_df["prompt_strategy"] == best_row["prompt_strategy"])
        & (predictions_df["model"] == best_row["model"])
    ]

    labels = list(VALID_LABELS)
    if (subset["predicted_label"] == INVALID_LABEL).any():
        labels.append(INVALID_LABEL)

    matrix = confusion_matrix(subset["true_label"], subset["predicted_label"], labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix: {best_row['prompt_strategy']} / {best_row['model']}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _annotate_conditions(ax: plt.Axes, plot_df: pd.DataFrame, x_col: str) -> None:
    for _, row in plot_df.iterrows():
        label = f"{row['prompt_strategy']}/{row['model']}"
        ax.annotate(
            label,
            (row[x_col], row["accuracy"]),
            textcoords="offset points",
            xytext=(6, 5),
            ha="left",
            fontsize=8,
        )


def _draw_pareto_frontier(ax: plt.Axes, plot_df: pd.DataFrame, x_col: str, y_col: str) -> None:
    frontier = _pareto_frontier(plot_df, x_col=x_col, y_col=y_col)
    if len(frontier) < 2:
        return
    ax.plot(frontier[x_col], frontier[y_col], color="black", linestyle="--", linewidth=1, label="Pareto frontier")
    ax.legend()


def _pareto_frontier(plot_df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    sorted_df = plot_df.sort_values([x_col, y_col], ascending=[True, False])
    frontier_rows = []
    best_y = -np.inf
    for _, row in sorted_df.iterrows():
        if row[y_col] >= best_y:
            frontier_rows.append(row)
            best_y = row[y_col]
    if not frontier_rows:
        return pd.DataFrame(columns=plot_df.columns)
    return pd.DataFrame(frontier_rows)
