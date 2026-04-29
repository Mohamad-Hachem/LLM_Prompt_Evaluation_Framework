"""Entry point for running the full prompt evaluation framework."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.evaluator import compute_metrics, plot_results
from src.runner import run_full_evaluation


class PromptEvaluationApplication:
    """
    Runs the project from beginning to end.

    This class is the script version of the full workflow:
    load API keys, run all model calls, compute metrics, create plots, and print
    a short summary of the best prompt/model combinations.
    """

    def __init__(self):
        self.test_set_path = "data/test_set.csv"
        self.results_dir = Path("results")

    def running_the_application(self) -> None:
        """
        Runs the full benchmark pipeline.
        """

        self.setting_up_logging()
        self.loading_environment_variables()
        self.checking_that_api_keys_exist()

        predictions_df = run_full_evaluation(self.test_set_path, str(self.results_dir))
        metrics_df = compute_metrics(predictions_df, self.results_dir / "metrics_summary.csv")
        plot_results(metrics_df, predictions_df, self.results_dir)
        self.printing_best_conditions_summary(metrics_df)

    def setting_up_logging(self) -> None:
        """
        Sets logging so the user can see progress and saved files.
        """

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def loading_environment_variables(self) -> None:
        """
        Loads API keys from the `.env` file.
        """

        load_dotenv()

    def checking_that_api_keys_exist(self) -> None:
        """
        Stops early if either API key is missing.
        """

        required_keys = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
        missing_keys = [key for key in required_keys if not os.getenv(key)]

        if missing_keys:
            missing_keys_text = ", ".join(missing_keys)
            raise SystemExit(f"Missing required environment variable(s): {missing_keys_text}")

    def printing_best_conditions_summary(self, metrics_df: pd.DataFrame) -> None:
        """
        Prints a compact summary table after the benchmark finishes.
        """

        summary_rows = [
            self.finding_best_row_for_metric(metrics_df, "highest_accuracy", "accuracy", ascending=False),
            self.finding_best_row_for_metric(metrics_df, "highest_macro_f1", "macro_f1", ascending=False),
            self.finding_best_row_for_metric(metrics_df, "lowest_mean_latency", "mean_latency_ms", ascending=True),
            self.finding_best_row_for_metric(metrics_df, "lowest_cost", "cost_per_1k_predictions", ascending=True),
            self.finding_best_row_for_metric(metrics_df, "lowest_invalid_rate", "invalid_rate", ascending=True),
        ]

        calibration_df = metrics_df.dropna(subset=["confidence_calibration_ece"])

        if not calibration_df.empty:
            summary_rows.append(
                self.finding_best_row_for_metric(
                    calibration_df,
                    "best_calibration_ece",
                    "confidence_calibration_ece",
                    ascending=True,
                )
            )

        summary_df = pd.DataFrame(summary_rows)
        print("\nBest conditions by metric:")
        print(summary_df.to_string(index=False))

    def finding_best_row_for_metric(
        self,
        metrics_df: pd.DataFrame,
        metric_name: str,
        metric_column: str,
        ascending: bool,
    ) -> dict[str, object]:
        """
        Finds the best prompt/model row for one metric.
        """

        best_row = metrics_df.sort_values(metric_column, ascending=ascending).iloc[0]

        return {
            "metric": metric_name,
            "prompt_strategy": best_row["prompt_strategy"],
            "model": best_row["model"],
            "value": best_row[metric_column],
        }


def main() -> None:
    """
    Small public entry point for the command line.
    """

    PromptEvaluationApplication().running_the_application()


if __name__ == "__main__":
    main()
