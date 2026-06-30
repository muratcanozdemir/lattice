"""Async client for OpenAI-compatible chat completion endpoints.

Works unmodified against llama.cpp's llama-server, vLLM, OpenAI, or any
other server implementing POST /v1/chat/completions with the standard
request/response shape. No provider-specific branching.

Rate limiting is two independent buckets (requests/min, tokens/min) rather
than a single combined limiter, because providers enforce both
independently and hitting either one returns a 429.

Token-per-minute accounting is necessarily approximate before a response
arrives: prompt tokens are estimated via a cheap heuristic and reserved
up front, then the bucket is reconciled against actual usage once the
response lands. This under- or over-reserves slightly but converges over
a session; it is not exact, and is not meant to be.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from lattice.metrics import MetricsCollector

DEFAULT_TIMEOUT_SECONDS = 30.0
RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


def _estimate_tokens(text: str) -> int:
    """Cheap chars/4 heuristic. Reserved capacity only, never billed."""
    return max(1, len(text) // 4)


class _TokenBucket:
    """Continuous-refill token bucket. capacity == per-minute limit."""

    def __init__(self, capacity: float) -> None:
        self.capacity = capacity
        self._tokens = capacity
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    def _refill(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.capacity, self._tokens + elapsed * (self.capacity / 60.0)
        )
        self._last_refill = now

    async def acquire(self, amount: float) -> None:
        if amount > self.capacity:
            # Can't ever satisfy this from a bucket this size; don't deadlock,
            # just let it through. Caller's estimate was wrong or limit is too low.
            return
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= amount:
                    self._tokens -= amount
                    return
                deficit = amount - self._tokens
                wait = deficit / (self.capacity / 60.0)
            await asyncio.sleep(wait)

    def adjust(self, delta: float) -> None:
        """Reconcile a prior reservation against actual usage. delta can be negative."""
        self._tokens = max(0.0, min(self.capacity, self._tokens - delta))


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float | None = None


@dataclass
class ChatResult:
    text: str
    usage: Usage
    model: str
    raw: dict[str, Any] = field(repr=False)


class LLMError(Exception):
    """Raised when a request exhausts retries or fails non-retryably."""


@dataclass
class ClientConfig:
    base_url: str
    api_key: str | None = None
    model: str = ""
    rpm: float = 60.0
    tpm: float = 60_000.0
    max_concurrency: int = 8
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    # USD per 1K tokens, (prompt, completion). None disables cost tracking.
    price_per_1k: tuple[float, float] | None = None


class LLMClient:
    """Async chat-completion client. One instance per provider/model."""

    def __init__(
        self, config: ClientConfig, *, metrics: MetricsCollector | None = None
    ) -> None:
        self.config = config
        self.metrics = metrics
        self._rpm_bucket = _TokenBucket(config.rpm)
        self._tpm_bucket = _TokenBucket(config.tpm)
        self._semaphore = asyncio.Semaphore(config.max_concurrency)
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        self._http = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_seconds,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "LLMClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ChatResult:
        prompt_text = " ".join(m.get("content", "") for m in messages)
        reserved = float(_estimate_tokens(prompt_text) + (max_tokens or 512))

        await self._rpm_bucket.acquire(1.0)
        await self._tpm_bucket.acquire(reserved)

        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        if extra_body:
            payload.update(extra_body)

        async with self._semaphore:
            try:
                data = await self._request_with_retry(payload)
            except LLMError:
                if self.metrics is not None:
                    await self.metrics.record_failure()
                raise

        usage_raw = data.get("usage", {})
        usage = Usage(
            prompt_tokens=usage_raw.get("prompt_tokens", 0),
            completion_tokens=usage_raw.get("completion_tokens", 0),
            total_tokens=usage_raw.get("total_tokens", 0),
        )
        if self.config.price_per_1k is not None:
            p_price, c_price = self.config.price_per_1k
            usage.cost_usd = (
                usage.prompt_tokens / 1000 * p_price
                + usage.completion_tokens / 1000 * c_price
            )

        if self.metrics is not None:
            await self.metrics.record_success(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                cost_usd=usage.cost_usd,
            )

        # Reconcile the tpm reservation against what actually happened.
        actual = float(usage.total_tokens) if usage.total_tokens else reserved
        self._tpm_bucket.adjust(actual - reserved)

        choice = data["choices"][0]["message"]
        return ChatResult(
            text=choice.get("content", ""),
            usage=usage,
            model=data.get("model", self.config.model),
            raw=data,
        )

    async def _request_with_retry(self, payload: dict[str, Any]) -> dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                resp = await self._http.post("/v1/chat/completions", json=payload)
                if (
                    resp.status_code >= 400
                    and resp.status_code not in RETRYABLE_STATUS_CODES
                ):
                    # Non-retryable client/server error (4xx other than 429, etc).
                    # Fail immediately - retrying a 400 burns the budget for nothing.
                    resp.raise_for_status()
                if resp.status_code in RETRYABLE_STATUS_CODES:
                    raise httpx.HTTPStatusError(
                        f"retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                resp.raise_for_status()
                result: dict[str, Any] = resp.json()
                return result
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code if exc.response is not None else None
                if status is not None and status not in RETRYABLE_STATUS_CODES:
                    raise LLMError(f"non-retryable status {status}: {exc}") from exc
                last_exc = exc
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_exc = exc
            else:
                continue
            if attempt >= self.config.max_retries:
                break
            sleep_for = self.config.backoff_base_seconds * (2**attempt)
            sleep_for += random.uniform(0, sleep_for * 0.1)  # jitter
            await asyncio.sleep(sleep_for)
        raise LLMError(
            f"request failed after {self.config.max_retries + 1} attempts: {last_exc}"
        ) from last_exc
