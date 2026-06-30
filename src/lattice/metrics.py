"""Aggregate token/cost accounting across a pipeline run.

Per-call accounting (Usage, ChatResult.usage) already exists from client.py
- every acomplete() returns it. This module is purely the summation layer:
attach a MetricsCollector to an LLMClient and it accumulates automatically;
read PipelineMetrics off the collector whenever you want a snapshot.

Deliberately not a context-manager-scoped "run" concept - a single
LLMClient may be reused across many logical pipeline runs (it's one
instance per provider/model, per the client's own docstring), and forcing
a one-collector-per-run lifecycle would fight that. Create one collector
per unit of accounting you care about (per pipeline invocation, per day,
per table-build) and attach it to whichever client calls belong to it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field


@dataclass
class PipelineMetrics:
    """Accumulated totals. Mutated in place by MetricsCollector.record()."""

    num_calls: int = 0
    num_failed_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    # True only if every recorded call had cost_usd set (price_per_1k was
    # configured). If any call's cost is unknown, total_cost_usd undercounts
    # silently unless the caller checks this first.
    cost_complete: bool = True

    def __str__(self) -> str:
        cost = f"${self.total_cost_usd:.4f}" if self.cost_complete else (
            f"${self.total_cost_usd:.4f} (incomplete - some calls had no pricing)"
        )
        return (
            f"PipelineMetrics(calls={self.num_calls}, failed={self.num_failed_calls}, "
            f"tokens={self.total_tokens} "
            f"[prompt={self.total_prompt_tokens}, completion={self.total_completion_tokens}], "
            f"cost={cost})"
        )


class MetricsCollector:
    """Thread/task-safe accumulator. Pass to LLMClient(metrics=...)."""

    def __init__(self) -> None:
        self._metrics = PipelineMetrics()
        self._lock = asyncio.Lock()

    @property
    def metrics(self) -> PipelineMetrics:
        """Returns a snapshot (copy) - safe to read concurrently with writes."""
        return PipelineMetrics(
            num_calls=self._metrics.num_calls,
            num_failed_calls=self._metrics.num_failed_calls,
            total_prompt_tokens=self._metrics.total_prompt_tokens,
            total_completion_tokens=self._metrics.total_completion_tokens,
            total_tokens=self._metrics.total_tokens,
            total_cost_usd=self._metrics.total_cost_usd,
            cost_complete=self._metrics.cost_complete,
        )

    async def record_success(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float | None,
    ) -> None:
        async with self._lock:
            self._metrics.num_calls += 1
            self._metrics.total_prompt_tokens += prompt_tokens
            self._metrics.total_completion_tokens += completion_tokens
            self._metrics.total_tokens += total_tokens
            if cost_usd is None:
                self._metrics.cost_complete = False
            else:
                self._metrics.total_cost_usd += cost_usd

    async def record_failure(self) -> None:
        async with self._lock:
            self._metrics.num_calls += 1
            self._metrics.num_failed_calls += 1
