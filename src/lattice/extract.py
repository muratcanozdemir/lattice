"""Schema-validated extraction over an LLMClient.

This is the one place lattice deliberately diverges from fenic's default:
fenic's semantic.extract degrades a failed row to None silently. That's a
reasonable default for an exploratory pipeline and a dangerous default for
anything feeding a decision. Here, failure mode is a required choice at the
call site, not a library default — see FailureMode.

Two distinct failure classes are handled differently:
  - Transport/HTTP failures (timeouts, 5xx, rate limits) are retried by
    LLMClient itself, below this layer. This module never sees them as
    retryable; an LLMError from the client propagates immediately.
  - Validation failures (the model returned malformed JSON, or
    JSON that doesn't satisfy the schema) are retried *here*, by re-asking
    with the validation error appended to the prompt, because a different
    sampling pass with the concrete error in context is the only lever
    available to fix it. This is bounded by max_validation_retries and is
    independent of LLMClient's own retry budget.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from lattice.client import LLMClient

ModelT = TypeVar("ModelT", bound=BaseModel)

_RETRY_PROMPT_TEMPLATE = (
    "Your previous response did not match the required schema.\n\n"
    "Previous response:\n{previous}\n\n"
    "Validation error:\n{error}\n\n"
    "Respond again with ONLY valid JSON matching the schema. "
    "No prose, no markdown fences."
)


class FailureMode(Enum):
    """How extract() behaves once validation retries are exhausted.

    There is no default - every call site picks one. A pipeline feeding
    a downstream decision should almost always choose RAISE; a pipeline
    doing best-effort enrichment over a large, noisy corpus may choose
    NONE deliberately, but that's a choice to state, not a fallback to
    inherit silently.
    """

    RAISE = "raise"  # fail closed: ExtractionError propagates
    NONE = "none"  # graceful degradation: returns None, caller must check


class ExtractionError(Exception):
    """Raised under FailureMode.RAISE when validation retries are exhausted."""

    def __init__(self, message: str, *, last_response: str, last_error: str) -> None:
        super().__init__(message)
        self.last_response = last_response
        self.last_error = last_error


def _build_response_format(schema: type[BaseModel]) -> dict[str, object]:
    """OpenAI structured-outputs response_format shape.

    Supported by OpenAI, and by llama.cpp/vLLM when built with grammar-
    constrained decoding enabled. Servers that don't support it will
    generally ignore an unrecognized field rather than error - but if a
    given backend rejects it outright, that's a backend capability gap,
    not something this wrapper can paper over without losing the point
    of doing schema-constrained decoding in the first place.
    """
    json_schema = schema.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": schema.__name__,
            "schema": json_schema,
            "strict": True,
        },
    }


def _parse_and_validate(text: str, schema: type[ModelT]) -> ModelT:
    """Raises json.JSONDecodeError or pydantic.ValidationError on failure."""
    stripped = text.strip()
    # Some backends wrap output in markdown fences despite instructions not to.
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    data = json.loads(stripped)
    return schema.model_validate(data)


async def extract(
    client: LLMClient,
    *,
    prompt: str,
    schema: type[ModelT],
    failure_mode: FailureMode,
    max_validation_retries: int = 2,
    temperature: float = 0.0,
    system_prompt: str | None = None,
) -> ModelT | None:
    """Extract a schema-validated object from a single LLM call.

    failure_mode is required, not defaulted - see FailureMode docstring.
    Returns ModelT under FailureMode.RAISE (or raises ExtractionError).
    Returns ModelT | None under FailureMode.NONE.
    """
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response_format = _build_response_format(schema)

    last_text = ""
    last_error_str = ""

    for attempt in range(max_validation_retries + 1):
        result = await client.acomplete(
            messages,
            temperature=temperature,
            response_format=response_format,
        )
        last_text = result.text
        try:
            return _parse_and_validate(result.text, schema)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error_str = str(exc)
            if attempt >= max_validation_retries:
                break
            messages.append({"role": "assistant", "content": result.text})
            messages.append(
                {
                    "role": "user",
                    "content": _RETRY_PROMPT_TEMPLATE.format(
                        previous=result.text, error=last_error_str
                    ),
                }
            )

    if failure_mode is FailureMode.NONE:
        return None
    raise ExtractionError(
        f"extraction failed validation after {max_validation_retries + 1} attempts",
        last_response=last_text,
        last_error=last_error_str,
    )
