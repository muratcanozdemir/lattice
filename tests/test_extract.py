import httpx
import pytest
import respx
from pydantic import BaseModel

from lattice import ClientConfig, ExtractionError, FailureMode, LLMClient, extract


class Preference(BaseModel):
    category: str
    value: str


def _chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.asyncio
async def test_extract_succeeds_on_first_valid_response():
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_chat_response('{"category": "diet", "value": "vegetarian"}'),
            )
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await extract(
                client,
                prompt="extract preference from: I'm vegetarian",
                schema=Preference,
                failure_mode=FailureMode.RAISE,
            )
            assert result == Preference(category="diet", value="vegetarian")


@pytest.mark.asyncio
async def test_extract_strips_markdown_fences():
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200,
                json=_chat_response(
                    '```json\n{"category": "diet", "value": "vegan"}\n```'
                ),
            )
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await extract(
                client,
                prompt="x",
                schema=Preference,
                failure_mode=FailureMode.RAISE,
            )
            assert result == Preference(category="diet", value="vegan")


@pytest.mark.asyncio
async def test_extract_retries_on_invalid_json_then_succeeds():
    with respx.mock(base_url="http://localhost:8080") as mock:
        route = mock.post("/v1/chat/completions")
        route.side_effect = [
            httpx.Response(200, json=_chat_response("not json at all")),
            httpx.Response(
                200,
                json=_chat_response('{"category": "diet", "value": "vegan"}'),
            ),
        ]
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await extract(
                client,
                prompt="x",
                schema=Preference,
                failure_mode=FailureMode.RAISE,
                max_validation_retries=2,
            )
            assert result == Preference(category="diet", value="vegan")
            assert route.call_count == 2


@pytest.mark.asyncio
async def test_extract_retries_on_schema_mismatch_then_succeeds():
    with respx.mock(base_url="http://localhost:8080") as mock:
        route = mock.post("/v1/chat/completions")
        route.side_effect = [
            # valid JSON, wrong shape - missing required field
            httpx.Response(200, json=_chat_response('{"category": "diet"}')),
            httpx.Response(
                200,
                json=_chat_response('{"category": "diet", "value": "vegan"}'),
            ),
        ]
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await extract(
                client,
                prompt="x",
                schema=Preference,
                failure_mode=FailureMode.RAISE,
                max_validation_retries=2,
            )
            assert result == Preference(category="diet", value="vegan")
            assert route.call_count == 2


@pytest.mark.asyncio
async def test_extract_fail_closed_raises_after_exhausting_retries():
    with respx.mock(base_url="http://localhost:8080") as mock:
        route = mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response("still not json"))
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            with pytest.raises(ExtractionError) as exc_info:
                await extract(
                    client,
                    prompt="x",
                    schema=Preference,
                    failure_mode=FailureMode.RAISE,
                    max_validation_retries=1,
                )
            assert route.call_count == 2  # initial + 1 retry
            assert exc_info.value.last_response == "still not json"


@pytest.mark.asyncio
async def test_extract_graceful_degradation_returns_none():
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(200, json=_chat_response("still not json"))
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await extract(
                client,
                prompt="x",
                schema=Preference,
                failure_mode=FailureMode.NONE,
                max_validation_retries=1,
            )
            assert result is None


@pytest.mark.asyncio
async def test_extract_sends_json_schema_response_format():
    captured: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json as _json

        captured.append(_json.loads(request.content))
        return httpx.Response(
            200,
            json=_chat_response('{"category": "diet", "value": "vegan"}'),
        )

    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            await extract(
                client, prompt="x", schema=Preference, failure_mode=FailureMode.RAISE
            )

    sent = captured[0]
    assert sent["response_format"]["type"] == "json_schema"
    assert sent["response_format"]["json_schema"]["name"] == "Preference"
