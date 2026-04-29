"""Prompt rendering and model-output parsing for support ticket classification."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any


VALID_LABELS: tuple[str, ...] = (
    "order_status",
    "refund_request",
    "product_question",
    "shipping_issue",
    "complaint",
    "account_issue",
)

INVALID_LABEL = "INVALID"


class CustomerSupportClassifier:
    """
    Handles classifying one customer support message at a time.

    This class is intentionally small: it renders one prompt, calls one model,
    and then parses the raw LLM output into a clean label that the evaluator can
    score.
    """

    def __init__(self, prompt_template: str, model_callable: Callable[[str], dict[str, Any]]):
        self.prompt_template = prompt_template
        self.model_callable = model_callable

    def classifying_a_customer_message(self, message: str) -> dict[str, Any]:
        """
        Renders the prompt, calls the model, and returns the parsed prediction.
        """

        prompt = self.rendering_prompt_with_customer_message(message)
        model_response = self.model_callable(prompt)
        raw_output = str(model_response.get("raw_output", ""))

        prompt_strategy = self.checking_which_prompt_style_we_are_using()
        parsed_output = self.parsing_the_model_output(raw_output, prompt_strategy)

        return {
            "predicted_label": parsed_output["predicted_label"],
            "confidence": parsed_output["confidence"],
            "reasoning": parsed_output["reasoning"],
            "latency_ms": float(model_response.get("latency_ms", 0.0) or 0.0),
            "cost_usd": float(model_response.get("cost_usd", 0.0) or 0.0),
            "raw_output": raw_output,
        }

    def rendering_prompt_with_customer_message(self, message: str) -> str:
        """
        Inserts the customer message into the prompt template.
        """

        return self.prompt_template.format(message=message)

    def checking_which_prompt_style_we_are_using(self) -> str:
        """
        Figures out how the model output should be parsed.
        """

        normalized_template = self.prompt_template.lower()

        if "valid json" in normalized_template and "confidence" in normalized_template:
            return "role_structured"

        if "prefixed with `label:`" in normalized_template or "prefixed with label:" in normalized_template:
            return "chain_of_thought"

        return "label_only"

    def parsing_the_model_output(self, raw_output: str, prompt_strategy: str) -> dict[str, Any]:
        """
        Sends the raw output to the correct parser for that prompt strategy.
        """

        if prompt_strategy == "role_structured":
            return self.parsing_structured_json_output(raw_output)

        if prompt_strategy == "chain_of_thought":
            return self.parsing_chain_of_thought_output(raw_output)

        return self.parsing_label_only_output(raw_output)

    def parsing_label_only_output(self, raw_output: str) -> dict[str, Any]:
        """
        Parses zero-shot and few-shot outputs where the model should return only a label.
        """

        candidate_label = raw_output.strip().lower()

        if candidate_label not in VALID_LABELS:
            candidate_label = INVALID_LABEL

        return {
            "predicted_label": candidate_label,
            "confidence": None,
            "reasoning": None,
        }

    def parsing_chain_of_thought_output(self, raw_output: str) -> dict[str, Any]:
        """
        Parses reasoning text and extracts the final `Label:` line.
        """

        label_match = re.search(r"(?im)^\s*Label:\s*([A-Za-z_]+)\s*$", raw_output)

        if not label_match:
            return {
                "predicted_label": INVALID_LABEL,
                "confidence": None,
                "reasoning": raw_output.strip() or None,
            }

        candidate_label = label_match.group(1).strip().lower()
        reasoning = raw_output[: label_match.start()].strip() or None

        if candidate_label not in VALID_LABELS:
            candidate_label = INVALID_LABEL

        return {
            "predicted_label": candidate_label,
            "confidence": None,
            "reasoning": reasoning,
        }

    def parsing_structured_json_output(self, raw_output: str) -> dict[str, Any]:
        """
        Parses JSON output from the structured prompt.
        """

        try:
            json_payload = json.loads(raw_output.strip())
        except json.JSONDecodeError:
            return {
                "predicted_label": INVALID_LABEL,
                "confidence": None,
                "reasoning": None,
            }

        if not isinstance(json_payload, dict):
            return {
                "predicted_label": INVALID_LABEL,
                "confidence": None,
                "reasoning": None,
            }

        candidate_label = str(json_payload.get("label", "")).strip().lower()
        confidence = self.parsing_confidence_value(json_payload.get("confidence"))
        reasoning_value = json_payload.get("reasoning")
        reasoning = str(reasoning_value).strip() if reasoning_value is not None else None

        if candidate_label not in VALID_LABELS:
            candidate_label = INVALID_LABEL

        return {
            "predicted_label": candidate_label,
            "confidence": confidence,
            "reasoning": reasoning or None,
        }

    def parsing_confidence_value(self, confidence_value: Any) -> float | None:
        """
        Makes sure confidence is a number between 0 and 1.
        """

        try:
            confidence = float(confidence_value)
        except (TypeError, ValueError):
            return None

        if 0 <= confidence <= 1:
            return confidence

        return None


def classify(
    message: str,
    prompt_template: str,
    model_callable: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """
    Public helper that classifies one customer message.
    """

    classifier = CustomerSupportClassifier(prompt_template, model_callable)
    return classifier.classifying_a_customer_message(message)
