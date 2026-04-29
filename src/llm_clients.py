"""LLM client wrappers for the prompt evaluation project.

This file is where we call the real hosted models. I kept the public functions
`call_claude` and `call_openai` at the bottom because the rest of the project
uses them directly, but the main logic lives in the `LLMClientCaller` class.
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

# Prices used for the portfolio benchmark.
# Claude Haiku 4.5: $1 / 1M input tokens and $5 / 1M output tokens.
# GPT-4o-mini: $0.15 / 1M input tokens and $0.60 / 1M output tokens.
PRICING_USD_PER_MILLION: dict[str, dict[str, float]] = {
    CLAUDE_HAIKU_MODEL: {"input": 1.00, "output": 5.00},
    GPT_4O_MINI_MODEL: {"input": 0.15, "output": 0.60},
}


class LLMClientCaller:
    """
    Handles calling OpenAI and Anthropic models in a consistent way.

    The reason we are putting this logic in one class is to make the evaluation
    fair: every model call uses the same retry logic, timing logic, token
    accounting, and cost calculation.
    """

    MAX_RETRIES = 3
    BASE_BACKOFF_SECONDS = 1.0

    def calling_claude_model(self, prompt: str, model: str = CLAUDE_HAIKU_MODEL) -> dict[str, Any]:
        """
        Calls Claude Haiku and returns the output plus latency, tokens, and cost.
        """

        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        max_tokens = self.choosing_max_tokens_for_prompt(prompt)

        def api_call() -> Any:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )

        start_time = time.perf_counter()
        response = self.retrying_api_call_if_it_fails(api_call)
        latency_ms = (time.perf_counter() - start_time) * 1000

        raw_output = self.extracting_text_from_anthropic_response(response)
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

        return {
            "raw_output": raw_output,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": self.calculating_cost_usd(model, input_tokens, output_tokens),
        }

    def calling_openai_model(self, prompt: str, model: str = GPT_4O_MINI_MODEL) -> dict[str, Any]:
        """
        Calls GPT-4o-mini and returns the output plus latency, tokens, and cost.
        """

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        max_tokens = self.choosing_max_tokens_for_prompt(prompt)

        def api_call() -> Any:
            return client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0,
            )

        start_time = time.perf_counter()
        response = self.retrying_api_call_if_it_fails(api_call)
        latency_ms = (time.perf_counter() - start_time) * 1000

        raw_output = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

        return {
            "raw_output": raw_output,
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": self.calculating_cost_usd(model, input_tokens, output_tokens),
        }

    def retrying_api_call_if_it_fails(self, api_call: Callable[[], Any]) -> Any:
        """
        Retries temporary API failures so one rate-limit blip does not kill the run.
        """

        last_error: Exception | None = None

        for attempt_number in range(1, self.MAX_RETRIES + 1):
            try:
                return api_call()
            except Exception as error:  # noqa: BLE001 - API SDKs have different exception classes.
                last_error = error

                if attempt_number == self.MAX_RETRIES or not self.checking_if_error_is_temporary(error):
                    raise

                seconds_to_sleep = self.BASE_BACKOFF_SECONDS * (2 ** (attempt_number - 1))
                seconds_to_sleep += random.uniform(0, 0.25)
                LOGGER.warning(
                    "Temporary LLM API error on attempt %s/%s: %s. Retrying in %.2fs.",
                    attempt_number,
                    self.MAX_RETRIES,
                    error,
                    seconds_to_sleep,
                )
                time.sleep(seconds_to_sleep)

        if last_error is not None:
            raise last_error

        raise RuntimeError("The retry loop ended without returning a response.")

    def checking_if_error_is_temporary(self, error: Exception) -> bool:
        """
        Checks whether an API error is worth retrying.
        """

        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)

        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)

        if isinstance(status_code, int):
            return status_code in {408, 409, 429} or status_code >= 500

        temporary_error_names = (
            "apiconnectionerror",
            "apierror",
            "apitimeout",
            "internalserver",
            "ratelimit",
            "serviceunavailable",
            "timeout",
        )
        class_name = error.__class__.__name__.lower()
        error_message = str(error).lower()

        return any(name in class_name or name in error_message for name in temporary_error_names)

    def choosing_max_tokens_for_prompt(self, prompt: str) -> int:
        """
        Chooses a larger response budget for reasoning or JSON prompts.
        """

        normalized_prompt = prompt.lower()
        prompt_is_structured = "valid json" in normalized_prompt and "confidence" in normalized_prompt
        prompt_is_chain_of_thought = "prefixed with `label:`" in normalized_prompt
        prompt_is_chain_of_thought = prompt_is_chain_of_thought or "prefixed with label:" in normalized_prompt

        if prompt_is_structured or prompt_is_chain_of_thought:
            return 300

        return 50

    def extracting_text_from_anthropic_response(self, response: Any) -> str:
        """
        Extracts text content from an Anthropic message response.
        """

        text_parts: list[str] = []
        content_blocks = getattr(response, "content", []) or []

        for block in content_blocks:
            text = getattr(block, "text", None)

            if text is not None:
                text_parts.append(text)
            elif isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))

        return "".join(text_parts)

    def calculating_cost_usd(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """
        Calculates the estimated dollar cost from token counts.
        """

        prices = self.getting_pricing_for_model(model)
        input_cost = (input_tokens / 1_000_000) * prices["input"]
        output_cost = (output_tokens / 1_000_000) * prices["output"]

        return input_cost + output_cost

    def getting_pricing_for_model(self, model: str) -> dict[str, float]:
        """
        Returns pricing for the model we are evaluating.
        """

        if model in PRICING_USD_PER_MILLION:
            return PRICING_USD_PER_MILLION[model]

        if model.startswith("claude"):
            return PRICING_USD_PER_MILLION[CLAUDE_HAIKU_MODEL]

        if model.startswith("gpt-4o-mini"):
            return PRICING_USD_PER_MILLION[GPT_4O_MINI_MODEL]

        raise ValueError(f"No pricing configured for model: {model}")


# These functions keep the rest of the project simple.
def call_claude(prompt: str, model: str = CLAUDE_HAIKU_MODEL) -> dict[str, Any]:
    """
    Public helper used by the runner to call Claude.
    """

    return LLMClientCaller().calling_claude_model(prompt, model)


def call_openai(prompt: str, model: str = GPT_4O_MINI_MODEL) -> dict[str, Any]:
    """
    Public helper used by the runner to call GPT-4o-mini.
    """

    return LLMClientCaller().calling_openai_model(prompt, model)
