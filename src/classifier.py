"""Prompt rendering and output parsing for ticket classification."""

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


def classify(
    message: str,
    prompt_template: str,
    model_callable: Callable[[str], dict[str, Any]],
) -> dict[str, Any]:
    """Classify one customer message with one rendered prompt and model callable.

    Args:
        message: Customer support message to classify.
        prompt_template: Prompt template containing a ``{message}`` placeholder.
        model_callable: Function that accepts a rendered prompt and returns LLM metadata.

    Returns:
        Parsed prediction and selected model metadata.

    Example:
        >>> def fake_model(prompt: str) -> dict[str, object]:
        ...     return {"raw_output": "order_status", "latency_ms": 10.0, "cost_usd": 0.01}
        >>> classify("Where is my package?", "{message}", fake_model)["predicted_label"]
        'order_status'
    """

    prompt = prompt_template.format(message=message)
    response = model_callable(prompt)
    raw_output = str(response.get("raw_output", ""))

    prompt_mode = _infer_prompt_mode(prompt_template)
    parsed = _parse_output(raw_output, prompt_mode)

    return {
        "predicted_label": parsed["predicted_label"],
        "confidence": parsed["confidence"],
        "reasoning": parsed["reasoning"],
        "latency_ms": float(response.get("latency_ms", 0.0) or 0.0),
        "cost_usd": float(response.get("cost_usd", 0.0) or 0.0),
        "raw_output": raw_output,
    }


def _infer_prompt_mode(prompt_template: str) -> str:
    normalized = prompt_template.lower()
    if "valid json" in normalized and "confidence" in normalized:
        return "role_structured"
    if "prefixed with `label:`" in normalized or "prefixed with label:" in normalized:
        return "chain_of_thought"
    return "label_only"


def _parse_output(raw_output: str, prompt_mode: str) -> dict[str, Any]:
    if prompt_mode == "role_structured":
        return _parse_structured_json(raw_output)
    if prompt_mode == "chain_of_thought":
        return _parse_chain_of_thought(raw_output)
    return _parse_label_only(raw_output)


def _parse_label_only(raw_output: str) -> dict[str, Any]:
    candidate = raw_output.strip().lower()
    return {
        "predicted_label": candidate if candidate in VALID_LABELS else INVALID_LABEL,
        "confidence": None,
        "reasoning": None,
    }


def _parse_chain_of_thought(raw_output: str) -> dict[str, Any]:
    match = re.search(r"(?im)^\s*Label:\s*([A-Za-z_]+)\s*$", raw_output)
    if not match:
        return {"predicted_label": INVALID_LABEL, "confidence": None, "reasoning": raw_output.strip() or None}

    candidate = match.group(1).strip().lower()
    reasoning = raw_output[: match.start()].strip() or None
    return {
        "predicted_label": candidate if candidate in VALID_LABELS else INVALID_LABEL,
        "confidence": None,
        "reasoning": reasoning,
    }


def _parse_structured_json(raw_output: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw_output.strip())
    except json.JSONDecodeError:
        return {"predicted_label": INVALID_LABEL, "confidence": None, "reasoning": None}

    if not isinstance(payload, dict):
        return {"predicted_label": INVALID_LABEL, "confidence": None, "reasoning": None}

    candidate = str(payload.get("label", "")).strip().lower()
    confidence = _parse_confidence(payload.get("confidence"))
    reasoning_value = payload.get("reasoning")
    reasoning = str(reasoning_value).strip() if reasoning_value is not None else None

    return {
        "predicted_label": candidate if candidate in VALID_LABELS else INVALID_LABEL,
        "confidence": confidence,
        "reasoning": reasoning or None,
    }


def _parse_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if 0 <= confidence <= 1:
        return confidence
    return None
