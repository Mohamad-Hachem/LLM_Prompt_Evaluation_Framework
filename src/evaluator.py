"""Computes metrics and creates plots for the prompt evaluation benchmark."""

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


class PromptEvaluationAnalyzer:
    """
    Handles metrics, calibration, and visualization for the evaluation results.

    The runner creates raw predictions. This class turns those raw predictions
    into the tables and plots we need for the GitHub portfolio write-up.
    """

    def __init__(self, predictions_df: pd.DataFrame):
        self.predictions_df = predictions_df.copy()

    def computing_all_metrics(
        self,
        output_path: str | Path = "results/metrics_summary.csv",
    ) -> pd.DataFrame:
        """
        Computes one metrics row for each prompt/model condition.
        """

        self.checking_that_prediction_columns_exist()

        metric_rows: list[dict[str, float | str]] = []
        grouped_predictions = self.predictions_df.groupby(
            ["prompt_strategy", "model"],
            sort=False,
            dropna=False,
        )

        for (prompt_strategy, model), group in grouped_predictions:
            metric_rows.append(self.calculating_metrics_for_one_condition(prompt_strategy, model, group))

        metrics_df = pd.DataFrame(metric_rows)
        metrics_df = self.sorting_metrics_by_prompt_order(metrics_df)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(output_path, index=False)

        LOGGER.info("Saved metrics summary to %s", output_path)

        return metrics_df

    def calculating_metrics_for_one_condition(
        self,
        prompt_strategy: str,
        model: str,
        group: pd.DataFrame,
    ) -> dict[str, float | str]:
        """
        Calculates accuracy, F1, per-class scores, latency, cost, and calibration.
        """

        true_labels = group["true_label"].astype(str)
        predicted_labels = group["predicted_label"].fillna(INVALID_LABEL).astype(str)

        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(VALID_LABELS),
            average="macro",
            zero_division=0,
        )
        per_class_precision, per_class_recall, _, _ = precision_recall_fscore_support(
            true_labels,
            predicted_labels,
            labels=list(VALID_LABELS),
            average=None,
            zero_division=0,
        )

        total_cost = pd.to_numeric(group["cost_usd"], errors="coerce").fillna(0).sum()
        number_of_predictions = len(group)
        calibration_ece = np.nan

        if prompt_strategy == "role_structured":
            calibration_ece = self.calculating_expected_calibration_error(group)

        metrics_row: dict[str, float | str] = {
            "prompt_strategy": str(prompt_strategy),
            "model": str(model),
            "accuracy": float((true_labels == predicted_labels).mean()),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
        }

        # we are adding per-class metrics so we can see which labels are hard
        for label, label_precision, label_recall in zip(
            VALID_LABELS,
            per_class_precision,
            per_class_recall,
            strict=True,
        ):
            metrics_row[f"precision_{label}"] = float(label_precision)
            metrics_row[f"recall_{label}"] = float(label_recall)

        metrics_row.update(
            {
                "mean_latency_ms": float(pd.to_numeric(group["latency_ms"], errors="coerce").mean()),
                "p95_latency_ms": float(pd.to_numeric(group["latency_ms"], errors="coerce").quantile(0.95)),
                "total_cost_usd": float(total_cost),
                "cost_per_1k_predictions": float((total_cost / number_of_predictions) * 1000),
                "invalid_rate": float((predicted_labels == INVALID_LABEL).mean()),
                "confidence_calibration_ece": (
                    float(calibration_ece) if not np.isnan(calibration_ece) else np.nan
                ),
            }
        )

        return metrics_row

    def checking_that_prediction_columns_exist(self) -> None:
        """
        Makes sure the raw predictions dataframe has all columns needed for metrics.
        """

        required_columns = {
            "prompt_strategy",
            "model",
            "true_label",
            "predicted_label",
            "latency_ms",
            "cost_usd",
        }
        missing_columns = required_columns.difference(self.predictions_df.columns)

        if missing_columns:
            raise ValueError(f"Predictions dataframe is missing columns: {sorted(missing_columns)}")

    def calculating_expected_calibration_error(self, group: pd.DataFrame, bins: int = 10) -> float:
        """
        Calculates ECE for structured-output prompts that include confidence.
        """

        confidences = pd.to_numeric(group.get("confidence"), errors="coerce")
        valid_confidence_mask = confidences.between(0, 1, inclusive="both")

        if not valid_confidence_mask.any():
            return np.nan

        valid_confidences = confidences[valid_confidence_mask]
        correctness = (
            group.loc[valid_confidence_mask, "true_label"]
            == group.loc[valid_confidence_mask, "predicted_label"]
        ).astype(float)
        bin_ids = np.minimum((valid_confidences.to_numpy() * bins).astype(int), bins - 1)

        ece = 0.0
        total_examples = len(valid_confidences)

        for bin_id in range(bins):
            examples_in_bin = bin_ids == bin_id

            if not examples_in_bin.any():
                continue

            bin_confidence = float(valid_confidences.to_numpy()[examples_in_bin].mean())
            bin_accuracy = float(correctness.to_numpy()[examples_in_bin].mean())
            ece += (examples_in_bin.sum() / total_examples) * abs(bin_accuracy - bin_confidence)

        return float(ece)

    def creating_all_result_plots(self, metrics_df: pd.DataFrame, output_dir: str | Path) -> None:
        """
        Creates all plot images used in the README.
        """

        plots_dir = Path(output_dir) / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        sns.set_theme(style="whitegrid", context="notebook")

        self.plotting_accuracy_comparison(metrics_df, plots_dir / "accuracy_comparison.png")
        self.plotting_latency_vs_accuracy(metrics_df, plots_dir / "latency_vs_accuracy.png")
        self.plotting_best_confusion_matrix(metrics_df, plots_dir / "confusion_matrix_best.png")
        self.plotting_cost_vs_accuracy(metrics_df, plots_dir / "cost_vs_accuracy.png")

        LOGGER.info("Saved plots to %s", plots_dir)

    def plotting_accuracy_comparison(self, metrics_df: pd.DataFrame, output_path: Path) -> None:
        """
        Creates the grouped bar chart comparing accuracy.
        """

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_df = self.sorting_metrics_by_prompt_order(metrics_df)
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

    def plotting_latency_vs_accuracy(self, metrics_df: pd.DataFrame, output_path: Path) -> None:
        """
        Creates the scatter plot that compares speed and quality.
        """

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_df = metrics_df.copy()
        sns.scatterplot(data=plot_df, x="mean_latency_ms", y="accuracy", hue="model", s=90, ax=ax)

        self.annotating_each_condition_on_plot(ax, plot_df, "mean_latency_ms")
        self.drawing_the_pareto_frontier(ax, plot_df, x_col="mean_latency_ms", y_col="accuracy")

        ax.set_title("Latency vs Accuracy")
        ax.set_xlabel("Mean latency (ms)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def plotting_cost_vs_accuracy(self, metrics_df: pd.DataFrame, output_path: Path) -> None:
        """
        Creates the scatter plot that compares price and quality.
        """

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_df = metrics_df.copy()
        plot_df = plot_df[pd.to_numeric(plot_df["cost_per_1k_predictions"], errors="coerce") > 0]

        if plot_df.empty:
            ax.text(0.5, 0.5, "No positive cost data available", ha="center", va="center", transform=ax.transAxes)
        else:
            sns.scatterplot(data=plot_df, x="cost_per_1k_predictions", y="accuracy", hue="model", s=90, ax=ax)
            self.annotating_each_condition_on_plot(ax, plot_df, "cost_per_1k_predictions")
            self.drawing_the_pareto_frontier(ax, plot_df, x_col="cost_per_1k_predictions", y_col="accuracy")
            ax.set_xscale("log")

        ax.set_title("Cost vs Accuracy")
        ax.set_xlabel("Cost per 1k predictions (USD, log scale)")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)

        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    def plotting_best_confusion_matrix(self, metrics_df: pd.DataFrame, output_path: Path) -> None:
        """
        Finds the best condition by accuracy and plots its confusion matrix.
        """

        if metrics_df.empty:
            raise ValueError("Cannot plot best confusion matrix from an empty metrics dataframe.")

        best_row = metrics_df.sort_values(["accuracy", "macro_f1"], ascending=False).iloc[0]
        best_predictions = self.predictions_df[
            (self.predictions_df["prompt_strategy"] == best_row["prompt_strategy"])
            & (self.predictions_df["model"] == best_row["model"])
        ]

        labels = list(VALID_LABELS)

        if (best_predictions["predicted_label"] == INVALID_LABEL).any():
            labels.append(INVALID_LABEL)

        matrix = confusion_matrix(best_predictions["true_label"], best_predictions["predicted_label"], labels=labels)

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

    def annotating_each_condition_on_plot(self, ax: plt.Axes, plot_df: pd.DataFrame, x_col: str) -> None:
        """
        Adds small labels to each scatter point so the plot is readable.
        """

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

    def drawing_the_pareto_frontier(
        self,
        ax: plt.Axes,
        plot_df: pd.DataFrame,
        x_col: str,
        y_col: str,
    ) -> None:
        """
        Draws the Pareto frontier line on latency/cost tradeoff plots.
        """

        frontier = self.finding_the_pareto_frontier(plot_df, x_col=x_col, y_col=y_col)

        if len(frontier) < 2:
            return

        ax.plot(frontier[x_col], frontier[y_col], color="black", linestyle="--", linewidth=1, label="Pareto frontier")
        ax.legend()

    def finding_the_pareto_frontier(self, plot_df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
        """
        Finds the points where no other condition is both cheaper/faster and better.
        """

        sorted_df = plot_df.sort_values([x_col, y_col], ascending=[True, False])
        frontier_rows = []
        best_y_value = -np.inf

        for _, row in sorted_df.iterrows():
            if row[y_col] >= best_y_value:
                frontier_rows.append(row)
                best_y_value = row[y_col]

        if not frontier_rows:
            return pd.DataFrame(columns=plot_df.columns)

        return pd.DataFrame(frontier_rows)

    def sorting_metrics_by_prompt_order(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts prompt strategies in a human-readable order for tables and plots.
        """

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


def compute_metrics(
    predictions_df: pd.DataFrame,
    output_path: str | Path = "results/metrics_summary.csv",
) -> pd.DataFrame:
    """
    Public helper used by `run_eval.py` to compute metrics.
    """

    analyzer = PromptEvaluationAnalyzer(predictions_df)
    return analyzer.computing_all_metrics(output_path)


def plot_results(metrics_df: pd.DataFrame, predictions_df: pd.DataFrame, output_dir: str | Path) -> None:
    """
    Public helper used by `run_eval.py` to create plots.
    """

    analyzer = PromptEvaluationAnalyzer(predictions_df)
    analyzer.creating_all_result_plots(metrics_df, output_dir)
