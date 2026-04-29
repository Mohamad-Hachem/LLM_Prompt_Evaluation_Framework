"""Command-line entry point for the prompt evaluation benchmark."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.evaluator import compute_metrics, plot_results
from src.runner import run_full_evaluation


def main() -> None:
    """Run the full evaluation pipeline from API calls through plots.

    Example:
        >>> # doctest: +SKIP
        >>> main()
    """

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_dotenv()
    _validate_api_keys()

    results_dir = Path("results")
    predictions_df = run_full_evaluation("data/test_set.csv", str(results_dir))
    metrics_df = compute_metrics(predictions_df, results_dir / "metrics_summary.csv")
    plot_results(metrics_df, predictions_df, results_dir)
    _print_summary(metrics_df)


def _validate_api_keys() -> None:
    missing = [key for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY") if not os.getenv(key)]
    if missing:
        missing_list = ", ".join(missing)
        raise SystemExit(f"Missing required environment variable(s): {missing_list}")


def _print_summary(metrics_df: pd.DataFrame) -> None:
    summary_rows = [
        _best_row(metrics_df, "highest_accuracy", "accuracy", ascending=False),
        _best_row(metrics_df, "highest_macro_f1", "macro_f1", ascending=False),
        _best_row(metrics_df, "lowest_mean_latency", "mean_latency_ms", ascending=True),
        _best_row(metrics_df, "lowest_cost", "cost_per_1k_predictions", ascending=True),
        _best_row(metrics_df, "lowest_invalid_rate", "invalid_rate", ascending=True),
    ]

    calibration_df = metrics_df.dropna(subset=["confidence_calibration_ece"])
    if not calibration_df.empty:
        summary_rows.append(
            _best_row(calibration_df, "best_calibration_ece", "confidence_calibration_ece", ascending=True)
        )

    summary_df = pd.DataFrame(summary_rows)
    print("\nBest conditions by metric:")
    print(summary_df.to_string(index=False))


def _best_row(metrics_df: pd.DataFrame, metric_name: str, metric_column: str, ascending: bool) -> dict[str, object]:
    row = metrics_df.sort_values(metric_column, ascending=ascending).iloc[0]
    return {
        "metric": metric_name,
        "prompt_strategy": row["prompt_strategy"],
        "model": row["model"],
        "value": row[metric_column],
    }


if __name__ == "__main__":
    main()
