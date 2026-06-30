import asyncio

import httpx
import polars as pl
import pytest
import respx
from pydantic import BaseModel

from lattice import ClientConfig, FailureMode, LLMClient
from lattice.polars_ext import semantic_extract, semantic_extract_async


class Sentiment(BaseModel):
    label: str
    confidence: float


class OptionalField(BaseModel):
    label: str
    note: str | None


def _chat_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "model": "test-model",
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.mark.asyncio
async def test_semantic_extract_async_adds_struct_column():
    df = pl.DataFrame({"text": ["great product", "terrible experience"]})

    responses = [
        httpx.Response(
            200, json=_chat_response('{"label": "positive", "confidence": 0.9}')
        ),
        httpx.Response(
            200, json=_chat_response('{"label": "negative", "confidence": 0.8}')
        ),
    ]

    with respx.mock(base_url="http://localhost:8080") as mock:
        route = mock.post("/v1/chat/completions")
        route.side_effect = responses
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await semantic_extract_async(
                df,
                text_column="text",
                output_column="sentiment",
                client=client,
                schema=Sentiment,
                prompt_template="Classify sentiment: {text}",
                failure_mode=FailureMode.RAISE,
            )

    assert result.columns == ["text", "sentiment"]
    unpacked = result.unnest("sentiment")
    assert unpacked["label"].to_list() == ["positive", "negative"]
    assert unpacked["confidence"].to_list() == pytest.approx([0.9, 0.8])


@pytest.mark.asyncio
async def test_semantic_extract_optional_field_round_trips():
    df = pl.DataFrame({"text": ["x"]})
    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(
            return_value=httpx.Response(
                200, json=_chat_response('{"label": "a", "note": null}')
            )
        )
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await semantic_extract_async(
                df,
                text_column="text",
                output_column="out",
                client=client,
                schema=OptionalField,
                prompt_template="{text}",
                failure_mode=FailureMode.RAISE,
            )
    unpacked = result.unnest("out")
    assert unpacked["label"].to_list() == ["a"]
    assert unpacked["note"].to_list() == [None]


@pytest.mark.asyncio
async def test_semantic_extract_graceful_degradation_produces_null_struct_row():
    df = pl.DataFrame({"text": ["good", "unparseable"]})
    with respx.mock(base_url="http://localhost:8080") as mock:
        route = mock.post("/v1/chat/completions")
        route.side_effect = [
            httpx.Response(
                200, json=_chat_response('{"label": "positive", "confidence": 0.9}')
            ),
            httpx.Response(200, json=_chat_response("not json")),
        ]
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            result = await semantic_extract_async(
                df,
                text_column="text",
                output_column="sentiment",
                client=client,
                schema=Sentiment,
                prompt_template="{text}",
                failure_mode=FailureMode.NONE,
                max_validation_retries=0,
            )
    unpacked = result.unnest("sentiment")
    assert unpacked["label"].to_list() == ["positive", None]
    assert unpacked["confidence"].to_list()[1] is None


@pytest.mark.asyncio
async def test_semantic_extract_runs_rows_concurrently():
    """Cross-row concurrency comes from asyncio.gather + the client's own
    semaphore, not from anything Polars-specific - this just confirms the
    bridge doesn't accidentally serialize rows."""
    df = pl.DataFrame({"text": [f"row{i}" for i in range(5)]})
    in_flight = 0
    max_seen = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, max_seen
        in_flight += 1
        max_seen = max(max_seen, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(
            200, json=_chat_response('{"label": "x", "confidence": 0.5}')
        )

    with respx.mock(base_url="http://localhost:8080") as mock:
        mock.post("/v1/chat/completions").mock(side_effect=handler)
        config = ClientConfig(
            base_url="http://localhost:8080",
            model="test-model",
            max_concurrency=5,
            rpm=10_000,
            tpm=10_000_000,
        )
        async with LLMClient(config) as client:
            await semantic_extract_async(
                df,
                text_column="text",
                output_column="out",
                client=client,
                schema=Sentiment,
                prompt_template="{text}",
                failure_mode=FailureMode.RAISE,
            )
    assert max_seen > 1


@pytest.mark.filterwarnings("ignore:coroutine .* was never awaited:RuntimeWarning")
def test_semantic_extract_raises_when_called_from_running_loop():
    """The sync wrapper calls asyncio.run() and cannot be used from inside
    an already-running loop - semantic_extract_async() is the escape hatch.

    Note: asyncio.run() constructs the coroutine before checking for a
    running loop, so the RuntimeError fires before anything awaits it -
    that produces an expected, harmless "never awaited" warning, silenced
    above rather than left to mask a real one later.
    """

    async def _inner() -> None:
        df = pl.DataFrame({"text": ["x"]})
        config = ClientConfig(base_url="http://localhost:8080", model="test-model")
        async with LLMClient(config) as client:
            with pytest.raises(RuntimeError):
                semantic_extract(
                    df,
                    text_column="text",
                    output_column="out",
                    client=client,
                    schema=Sentiment,
                    prompt_template="{text}",
                    failure_mode=FailureMode.RAISE,
                )

    asyncio.run(_inner())


def test_unsupported_field_type_raises_not_implemented() -> None:
    class Nested(BaseModel):
        inner: dict[str, str]

    from lattice.polars_ext import _struct_dtype

    with pytest.raises(NotImplementedError):
        _struct_dtype(Nested)
