import httpx
import pytest
import respx

from lattice import ClientConfig, LLMClient, LLMError


def _completion_response(content: str = "hello", total: int = 30) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {
            "prompt_tokens": total - 10,
            "completion_tokens": 10,
            "total_tokens": total,
        },
    }


@pytest.mark.asyncio
async def test_basic_completion():
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion_response())
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await client.acomplete([{"role": "user", "content": "hi"}])
            assert result.text == "hello"
            assert result.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_retries_on_429_then_succeeds():
    with respx.mock(base_url="http://localhost:8080") as mock:
        route = mock.post("/v1/chat/completions")
        route.side_effect = [
            httpx.Response(429, json={"error": "rate limited"}),
            httpx.Response(200, json=_completion_response()),
        ]
        config = ClientConfig(
            base_url="http://localhost:8080",
            model="test-model",
            max_retries=2,
            backoff_base_seconds=0.01,
        )
        async with LLMClient(config) as client:
            result = await client.acomplete([{"role": "user", "content": "hi"}])
            assert result.text == "hello"
            assert route.call_count == 2


@pytest.mark.asyncio
async def test_exhausts_retries_and_raises():
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(503, json={"error": "unavailable"})
        )
        config = ClientConfig(
            base_url="http://localhost:8080",
            model="test-model",
            max_retries=1,
            backoff_base_seconds=0.01,
        )
        async with LLMClient(config) as client:
            with pytest.raises(LLMError):
                await client.acomplete([{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_non_retryable_status_raises_immediately():
    with respx.mock(base_url="http://localhost:8080") as mock:
        route = mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(400, json={"error": "bad request"})
        )
        config = ClientConfig(
            base_url="http://localhost:8080", model="test-model", max_retries=3
        )
        async with LLMClient(config) as client:
            with pytest.raises(LLMError):
                await client.acomplete([{"role": "user", "content": "hi"}])
            assert route.call_count == 1


@pytest.mark.asyncio
async def test_cost_tracking_when_price_set():
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_completion_response(total=2000))
        )
        config = ClientConfig(
            base_url="http://localhost:8080",
            model="test-model",
            price_per_1k=(0.01, 0.03),
        )
        async with LLMClient(config) as client:
            result = await client.acomplete([{"role": "user", "content": "hi"}])
            # prompt=1990, completion=10
            expected = (1990 / 1000 * 0.01) + (10 / 1000 * 0.03)
            assert result.usage.cost_usd == pytest.approx(expected)


@pytest.mark.asyncio
async def test_concurrency_bound_is_respected():
    in_flight = 0
    max_seen = 0

    async def handler(request):
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        import asyncio

        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(200, json=_completion_response())

    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        config = ClientConfig(
            base_url="http://localhost:8080",
            model="test-model",
            max_concurrency=2,
            rpm=10_000,
            tpm=10_000_000,
        )
        async with LLMClient(config) as client:
            import asyncio

            await asyncio.gather(
                *[
                    client.acomplete([{"role": "user", "content": "hi"}])
                    for _ in range(6)
                ]
            )
        assert max_seen <= 2
