# lattice

Thin, dependency-minimal semantic dataframe ops over Polars. No platform,
no planner, no BYO-keys-to-someone-else's-product.

Three hard dependencies: `polars`, `pydantic`, `httpx`. Works against any
OpenAI-compatible `/v1/chat/completions` endpoint — local or cloud — with
no provider-specific branching.

## Quickstart: local inference via llama.cpp

Start a llama.cpp server exposing the OpenAI-compatible API (CPU-only,
no GPU required):

```bash
llama-server -m ./models/your-model.gguf --host 0.0.0.0 --port 8080
```

Point `lattice` at it:

```python
import asyncio
from lattice import ClientConfig, LLMClient

async def main():
    config = ClientConfig(
        base_url="http://localhost:8080",
        model="local",          # llama.cpp ignores this field, but it's required by the schema
        api_key=None,           # no auth needed for a local server
        rpm=600,                # local server, set high — you're not rate-limited by a vendor
        tpm=1_000_000,
        max_concurrency=4,      # bound by your CPU thread budget, not a vendor quota
        timeout_seconds=120,    # CPU inference is slower than a hosted GPU endpoint
    )
    async with LLMClient(config) as client:
        result = await client.acomplete(
            [{"role": "user", "content": "Summarize: the quick brown fox..."}]
        )
        print(result.text)
        print(result.usage)

asyncio.run(main())
```

## Pointing at a hosted provider instead

Same client, different `base_url`/`api_key`/pricing:

```python
config = ClientConfig(
    base_url="https://api.openai.com",
    api_key="sk-...",
    model="gpt-4o-mini",
    rpm=500,
    tpm=200_000,
    price_per_1k=(0.15, 0.60),  # USD per 1K (prompt, completion) tokens, for cost tracking
)
```

## `sim_join`: exact vs approximate

```python
from lattice import sim_join

# Exact (default) - deterministic, chunked for bounded memory.
matches = sim_join(
    queries, documents,
    left_embedding_col="embedding", right_embedding_col="embedding",
    k=5,
)

# Approximate - HNSW via usearch, for large `right` tables where
# brute-force compute (not just memory) becomes the bottleneck.
matches = sim_join(
    queries, documents,
    left_embedding_col="embedding", right_embedding_col="embedding",
    k=5, method="approximate",
)
```

## What this is not

No dataframe planner, no MCP auto-generation, no managed table storage.
Parquet + Polars is the storage layer. If you need cost-aware reordering
of cheap filters ahead of expensive LLM calls, write your pipeline in
that order — there's no optimizer doing it for you, and there isn't
meant to be one.

