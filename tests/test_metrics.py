import httpx
import pytest
import respx

from lattice import ClientConfig, LLMClient, LLMError, MetricsCollector


def _chat_response(total: int = 30) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": "hi"}}],
        "usage": {
            "prompt_tokens": total - 10,
            "completion_tokens": 10,
            "total_tokens": total,
        },
    }


@pytest.mark.asyncio
async def test_metrics_accumulate_across_calls():
    collector = MetricsCollector()
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response(total=30))
        )
        config = ClientConfig(
            base_url="http://localhost:8080",
            model="test-model",
            price_per_1k=(0.01, 0.03),
        )
        async with LLMClient(config, metrics=collector) as client:
            for _ in range(3):
                await client.acomplete([{"role": "user", "content": "hi"}])

    m = collector.metrics
    assert m.num_calls == 3
    assert m.num_failed_calls == 0
    assert m.total_tokens == 90
    assert m.cost_complete is True
    assert m.total_cost_usd > 0


@pytest.mark.asyncio
async def test_metrics_without_pricing_marks_cost_incomplete():
    collector = MetricsCollector()
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config, metrics=collector) as client:
            await client.acomplete([{"role": "user", "content": "hi"}])

    assert collector.metrics.cost_complete is False
    assert collector.metrics.total_cost_usd == 0.0


@pytest.mark.asyncio
async def test_metrics_records_failures_separately():
    collector = MetricsCollector()
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(503, json={"error": "down"})
        )
        config = ClientConfig(
            base_url="http://localhost:8080",
            model="test-model",
            max_retries=0,
            backoff_base_seconds=0.01,
        )
        async with LLMClient(config, metrics=collector) as client:
            with pytest.raises(LLMError):
                await client.acomplete([{"role": "user", "content": "hi"}])

    m = collector.metrics
    assert m.num_calls == 1
    assert m.num_failed_calls == 1
    assert m.total_tokens == 0


@pytest.mark.asyncio
async def test_metrics_str_reflects_state():
    collector = MetricsCollector()
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response())
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config, metrics=collector) as client:
            await client.acomplete([{"role": "user", "content": "hi"}])
    text = str(collector.metrics)
    assert "calls=1" in text
    assert "incomplete" in text
