"""Raw SDK clients for prompt classification experiments.

The pricing constants below are hardcoded for reproducible portfolio analysis:

- Claude Haiku 4.5: $1.00 per 1M input tokens, $5.00 per 1M output tokens.
- GPT-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens.
"""

from __future__ import annotations

import logging
import os
import random
import time
from collections.abc import Callable
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

LOGGER = logging.getLogger(__name__)

CLAUDE_HAIKU_MODEL = "claude-haiku-4-5"
GPT_4O_MINI_MODEL = "gpt-4o-mini"

PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    CLAUDE_HAIKU_MODEL: {"input": 1.00, "output": 5.00},
    GPT_4O_MINI_MODEL: {"input": 0.15, "output": 0.60},
}

MAX_RETRIES = 3
BASE_BACKOFF_SECONDS = 1.0


def call_claude(prompt: str, model: str = CLAUDE_HAIKU_MODEL) -> dict[str, Any]:
    """Call Claude Haiku with retry, timing, token accounting, and cost.

    Args:
        prompt: Fully rendered prompt text.
        model: Anthropic model ID to call.

    Returns:
        A dictionary containing raw text, latency, token usage, and estimated cost.

    Example:
        >>> response = call_claude("Classify this: Where is my package?")
        >>> sorted(response)
        ['cost_usd', 'input_tokens', 'latency_ms', 'output_tokens', 'raw_output']
    """

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    max_tokens = _select_max_tokens(prompt)

    def operation() -> Any:
        return client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

    start = time.perf_counter()
    response = _with_retries(operation)
    latency_ms = (time.perf_counter() - start) * 1000

    raw_output = _extract_anthropic_text(response)
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    return {
        "raw_output": raw_output,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": _compute_cost_usd(model, input_tokens, output_tokens),
    }


def call_openai(prompt: str, model: str = GPT_4O_MINI_MODEL) -> dict[str, Any]:
    """Call GPT-4o-mini with retry, timing, token accounting, and cost.

    Args:
        prompt: Fully rendered prompt text.
        model: OpenAI model ID to call.

    Returns:
        A dictionary containing raw text, latency, token usage, and estimated cost.

    Example:
        >>> response = call_openai("Classify this: I need a refund.")
        >>> sorted(response)
        ['cost_usd', 'input_tokens', 'latency_ms', 'output_tokens', 'raw_output']
    """

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    max_tokens = _select_max_tokens(prompt)

    def operation() -> Any:
        return client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0,
        )

    start = time.perf_counter()
    response = _with_retries(operation)
    latency_ms = (time.perf_counter() - start) * 1000

    raw_output = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    return {
        "raw_output": raw_output,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": _compute_cost_usd(model, input_tokens, output_tokens),
    }


def _with_retries(operation: Callable[[], Any]) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - SDKs use different exception trees.
            last_error = exc
            if attempt == MAX_RETRIES or not _is_transient_error(exc):
                raise

            sleep_seconds = BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
            sleep_seconds += random.uniform(0, 0.25)
            LOGGER.warning(
                "Transient LLM API error on attempt %s/%s: %s. Retrying in %.2fs.",
                attempt,
                MAX_RETRIES,
                exc,
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Retry loop exited without returning a response.")


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)

    if isinstance(status_code, int):
        return status_code in {408, 409, 429} or status_code >= 500

    transient_names = (
        "apiconnectionerror",
        "apierror",
        "apitimeout",
        "internalserver",
        "ratelimit",
        "serviceunavailable",
        "timeout",
    )
    class_name = exc.__class__.__name__.lower()
    return any(marker in class_name for marker in transient_names)


def _select_max_tokens(prompt: str) -> int:
    normalized = prompt.lower()
    is_structured = "valid json" in normalized and "confidence" in normalized
    is_chain_of_thought = "prefixed with `label:`" in normalized or "prefixed with label:" in normalized
    return 300 if is_structured or is_chain_of_thought else 50


def _extract_anthropic_text(response: Any) -> str:
    blocks = getattr(response, "content", []) or []
    text_parts: list[str] = []
    for block in blocks:
        text = getattr(block, "text", None)
        if text is not None:
            text_parts.append(text)
        elif isinstance(block, dict) and block.get("type") == "text":
            text_parts.append(str(block.get("text", "")))
    return "".join(text_parts)


def _compute_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _pricing_for_model(model)
    return (
        (input_tokens / 1_000_000) * rates["input"]
        + (output_tokens / 1_000_000) * rates["output"]
    )


def _pricing_for_model(model: str) -> dict[str, float]:
    if model in PRICING_USD_PER_MILLION:
        return PRICING_USD_PER_MILLION[model]
    if model.startswith("claude"):
        return PRICING_USD_PER_MILLION[CLAUDE_HAIKU_MODEL]
    if model.startswith("gpt-4o-mini"):
        return PRICING_USD_PER_MILLION[GPT_4O_MINI_MODEL]
    raise ValueError(f"No pricing configured for model: {model}")
