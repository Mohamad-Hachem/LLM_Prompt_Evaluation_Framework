"""Evaluation orchestration across prompts, models, and the test set."""

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


def run_full_evaluation(
    test_set_path: str,
    output_dir: str,
    confirm: bool = True,
) -> pd.DataFrame:
    """Run all prompt and model combinations and save raw predictions.

    Args:
        test_set_path: Path to the labeled CSV test set.
        output_dir: Directory where ``raw_predictions.csv`` will be written.
        confirm: Whether to ask for interactive confirmation before API calls.

    Returns:
        Raw predictions dataframe with one row per API call.

    Example:
        >>> # doctest: +SKIP
        >>> run_full_evaluation("data/test_set.csv", "results")
    """

    project_root = Path(__file__).resolve().parents[1]
    test_set = _load_test_set(_resolve_path(project_root, test_set_path))
    prompt_templates = _load_prompt_templates(project_root / "prompts")
    output_path = _resolve_path(project_root, output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    estimated_cost = _estimate_evaluation_cost(test_set, prompt_templates)
    total_calls = len(test_set) * len(prompt_templates) * len(MODEL_CALLABLES)
    LOGGER.info(
        "Estimated upper-bound API cost for %s calls: $%.4f.",
        total_calls,
        estimated_cost,
    )

    if confirm:
        reply = input(f"Proceed with {total_calls} API calls? Estimated cost: ${estimated_cost:.2f}. [y/N]: ")
        if reply.strip().lower() not in {"y", "yes"}:
            raise RuntimeError("Evaluation cancelled by user.")

    predictions = _run_conditions(test_set, prompt_templates, total_calls)
    predictions_df = pd.DataFrame(predictions)
    predictions_csv = output_path / "raw_predictions.csv"
    predictions_df.to_csv(predictions_csv, index=False)
    LOGGER.info("Saved raw predictions to %s", predictions_csv)
    return predictions_df


def _run_conditions(
    test_set: pd.DataFrame,
    prompt_templates: dict[str, str],
    total_calls: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    progress = tqdm(total=total_calls, desc="Evaluating", unit="call")

    try:
        for prompt_strategy, prompt_template in prompt_templates.items():
            for model_name, model_callable in MODEL_CALLABLES.items():
                for record in test_set.itertuples(index=False):
                    result = _classify_with_failure_handling(
                        message=str(record.message),
                        prompt_template=prompt_template,
                        model_callable=model_callable,
                        prompt_strategy=prompt_strategy,
                        model_name=model_name,
                        record_id=record.id,
                    )
                    rows.append(
                        {
                            "id": record.id,
                            "message": record.message,
                            "true_label": record.true_label,
                            "prompt_strategy": prompt_strategy,
                            "model": model_name,
                            "predicted_label": result["predicted_label"],
                            "confidence": result["confidence"],
                            "reasoning": result["reasoning"],
                            "latency_ms": result["latency_ms"],
                            "cost_usd": result["cost_usd"],
                            "correct": result["predicted_label"] == record.true_label,
                            "raw_output": result["raw_output"],
                        }
                    )
                    progress.update(1)
    finally:
        progress.close()

    return rows


def _classify_with_failure_handling(
    message: str,
    prompt_template: str,
    model_callable: Any,
    prompt_strategy: str,
    model_name: str,
    record_id: Any,
) -> dict[str, Any]:
    try:
        return classify(message, prompt_template, model_callable)
    except Exception as exc:  # noqa: BLE001 - one failed API call should not stop the benchmark.
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
            "raw_output": f"ERROR: {exc}",
        }


def _load_test_set(path: Path) -> pd.DataFrame:
    test_set = pd.read_csv(path)
    expected_columns = {"id", "message", "true_label"}
    missing = expected_columns.difference(test_set.columns)
    if missing:
        raise ValueError(f"Test set is missing columns: {sorted(missing)}")
    return test_set


def _load_prompt_templates(prompts_dir: Path) -> dict[str, str]:
    templates: dict[str, str] = {}
    for strategy, file_name in PROMPT_FILES.items():
        prompt_path = prompts_dir / file_name
        templates[strategy] = prompt_path.read_text(encoding="utf-8")
    return templates


def _estimate_evaluation_cost(test_set: pd.DataFrame, prompt_templates: dict[str, str]) -> float:
    estimated_cost = 0.0
    for prompt_strategy, prompt_template in prompt_templates.items():
        output_tokens = 300 if prompt_strategy in {"chain_of_thought", "role_structured"} else 50
        for model_name in MODEL_CALLABLES:
            rates = PRICING_USD_PER_MILLION[model_name]
            for message in test_set["message"]:
                rendered_prompt = prompt_template.format(message=message)
                input_tokens = _approximate_token_count(rendered_prompt)
                estimated_cost += (input_tokens / 1_000_000) * rates["input"]
                estimated_cost += (output_tokens / 1_000_000) * rates["output"]
    return estimated_cost


def _approximate_token_count(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _resolve_path(project_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root / path
