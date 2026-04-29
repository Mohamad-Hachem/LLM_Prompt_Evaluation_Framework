"""Runs the full prompt evaluation across prompts, models, and test examples."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.classifier import INVALID_LABEL, classify
from src.llm_clients import (
    CLAUDE_HAIKU_MODEL,
    GPT_4O_MINI_MODEL,
    PRICING_USD_PER_MILLION,
    call_claude,
    call_openai,
)


LOGGER = logging.getLogger(__name__)

PROMPT_FILES: dict[str, str] = {
    "zero_shot": "zero_shot.txt",
    "few_shot": "few_shot.txt",
    "chain_of_thought": "chain_of_thought.txt",
    "role_structured": "role_structured.txt",
}

MODEL_CALLABLES = {
    CLAUDE_HAIKU_MODEL: call_claude,
    GPT_4O_MINI_MODEL: call_openai,
}


class PromptEvaluationRunner:
    """
    Handles running the complete benchmark.

    The reason we are making this class is to keep the evaluation steps easy to
    follow: load the data, load the prompts, estimate cost, ask the user, run all
    model calls, then save the raw predictions.
    """

    def __init__(self, test_set_path: str, output_dir: str, confirm: bool = True):
        self.project_root = Path(__file__).resolve().parents[1]
        self.test_set_path = self.resolving_path_from_project_root(test_set_path)
        self.output_dir = self.resolving_path_from_project_root(output_dir)
        self.confirm = confirm

    def running_the_full_evaluation(self) -> pd.DataFrame:
        """
        Runs every prompt/model condition and saves `raw_predictions.csv`.
        """

        test_set = self.loading_the_test_set()
        prompt_templates = self.loading_all_prompt_templates()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        estimated_cost = self.estimating_the_total_evaluation_cost(test_set, prompt_templates)
        total_calls = len(test_set) * len(prompt_templates) * len(MODEL_CALLABLES)

        LOGGER.info(
            "Estimated upper-bound API cost for %s calls: $%.4f.",
            total_calls,
            estimated_cost,
        )

        if self.confirm:
            self.asking_user_to_confirm_before_spending_money(total_calls, estimated_cost)

        predictions = self.running_all_prompt_and_model_conditions(test_set, prompt_templates, total_calls)
        predictions_df = pd.DataFrame(predictions)
        raw_predictions_path = self.output_dir / "raw_predictions.csv"
        predictions_df.to_csv(raw_predictions_path, index=False)

        LOGGER.info("Saved raw predictions to %s", raw_predictions_path)

        return predictions_df

    def loading_the_test_set(self) -> pd.DataFrame:
        """
        Loads the labeled CSV dataset and checks that the required columns exist.
        """

        test_set = pd.read_csv(self.test_set_path)
        expected_columns = {"id", "message", "true_label"}
        missing_columns = expected_columns.difference(test_set.columns)

        if missing_columns:
            raise ValueError(f"Test set is missing columns: {sorted(missing_columns)}")

        return test_set

    def loading_all_prompt_templates(self) -> dict[str, str]:
        """
        Loads all prompt files from the `prompts/` folder.
        """

        prompts_dir = self.project_root / "prompts"
        prompt_templates: dict[str, str] = {}

        for prompt_strategy, file_name in PROMPT_FILES.items():
            prompt_path = prompts_dir / file_name
            prompt_templates[prompt_strategy] = prompt_path.read_text(encoding="utf-8")

        return prompt_templates

    def asking_user_to_confirm_before_spending_money(self, total_calls: int, estimated_cost: float) -> None:
        """
        Gives the user one last chance before starting API calls.
        """

        user_reply = input(f"Proceed with {total_calls} API calls? Estimated cost: ${estimated_cost:.2f}. [y/N]: ")

        if user_reply.strip().lower() not in {"y", "yes"}:
            raise RuntimeError("Evaluation cancelled by user.")

    def running_all_prompt_and_model_conditions(
        self,
        test_set: pd.DataFrame,
        prompt_templates: dict[str, str],
        total_calls: int,
    ) -> list[dict[str, Any]]:
        """
        Loops over every prompt, every model, and every message.
        """

        prediction_rows: list[dict[str, Any]] = []
        progress_bar = tqdm(total=total_calls, desc="Evaluating", unit="call")

        try:
            for prompt_strategy, prompt_template in prompt_templates.items():
                for model_name, model_callable in MODEL_CALLABLES.items():
                    for record in test_set.itertuples(index=False):
                        prediction_result = self.classifying_one_message_without_stopping_the_evaluation(
                            message=str(record.message),
                            prompt_template=prompt_template,
                            model_callable=model_callable,
                            prompt_strategy=prompt_strategy,
                            model_name=model_name,
                            record_id=record.id,
                        )

                        prediction_rows.append(
                            self.creating_raw_prediction_row(
                                record=record,
                                prompt_strategy=prompt_strategy,
                                model_name=model_name,
                                prediction_result=prediction_result,
                            )
                        )
                        progress_bar.update(1)
        finally:
            progress_bar.close()

        return prediction_rows

    def classifying_one_message_without_stopping_the_evaluation(
        self,
        message: str,
        prompt_template: str,
        model_callable: Any,
        prompt_strategy: str,
        model_name: str,
        record_id: Any,
    ) -> dict[str, Any]:
        """
        Classifies one message and returns `INVALID` if that one call fails.
        """

        try:
            return classify(message, prompt_template, model_callable)
        except Exception as error:  # noqa: BLE001 - one failed call should not stop 799 other calls.
            LOGGER.exception(
                "Classification failed for id=%s prompt=%s model=%s. Marking as INVALID.",
                record_id,
                prompt_strategy,
                model_name,
            )

            return {
                "predicted_label": INVALID_LABEL,
                "confidence": None,
                "reasoning": None,
                "latency_ms": 0.0,
                "cost_usd": 0.0,
                "raw_output": f"ERROR: {error}",
            }

    def creating_raw_prediction_row(
        self,
        record: Any,
        prompt_strategy: str,
        model_name: str,
        prediction_result: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Creates the exact row that will be saved in `raw_predictions.csv`.
        """

        return {
            "id": record.id,
            "message": record.message,
            "true_label": record.true_label,
            "prompt_strategy": prompt_strategy,
            "model": model_name,
            "predicted_label": prediction_result["predicted_label"],
            "confidence": prediction_result["confidence"],
            "reasoning": prediction_result["reasoning"],
            "latency_ms": prediction_result["latency_ms"],
            "cost_usd": prediction_result["cost_usd"],
            "correct": prediction_result["predicted_label"] == record.true_label,
            "raw_output": prediction_result["raw_output"],
        }

    def estimating_the_total_evaluation_cost(
        self,
        test_set: pd.DataFrame,
        prompt_templates: dict[str, str],
    ) -> float:
        """
        Estimates API cost before the benchmark starts.

        This is only an estimate because the actual number of output tokens is
        only known after the API returns.
        """

        estimated_cost = 0.0

        for prompt_strategy, prompt_template in prompt_templates.items():
            output_tokens = self.choosing_estimated_output_tokens(prompt_strategy)

            for model_name in MODEL_CALLABLES:
                prices = PRICING_USD_PER_MILLION[model_name]

                for message in test_set["message"]:
                    rendered_prompt = prompt_template.format(message=message)
                    input_tokens = self.approximating_token_count(rendered_prompt)
                    estimated_cost += (input_tokens / 1_000_000) * prices["input"]
                    estimated_cost += (output_tokens / 1_000_000) * prices["output"]

        return estimated_cost

    def choosing_estimated_output_tokens(self, prompt_strategy: str) -> int:
        """
        Uses a bigger estimate for prompts that ask for reasoning or JSON.
        """

        if prompt_strategy in {"chain_of_thought", "role_structured"}:
            return 300

        return 50

    def approximating_token_count(self, text: str) -> int:
        """
        Uses a simple character-based token estimate for pre-run cost estimates.
        """

        return max(1, math.ceil(len(text) / 4))

    def resolving_path_from_project_root(self, path_value: str) -> Path:
        """
        Allows callers to pass either absolute paths or project-relative paths.
        """

        path = Path(path_value)

        if path.is_absolute():
            return path

        return self.project_root / path


def run_full_evaluation(
    test_set_path: str,
    output_dir: str,
    confirm: bool = True,
) -> pd.DataFrame:
    """
    Public helper that runs the full evaluation.
    """

    runner = PromptEvaluationRunner(test_set_path, output_dir, confirm)
    return runner.running_the_full_evaluation()
