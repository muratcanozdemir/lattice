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

## Status

All 7 WBS items implemented. 44 tests, `mypy --strict` clean across all
7 source files, no live network calls (HTTP-level calls are mocked via
`respx`; `sim_join`'s approximate path runs a real local `usearch` index
in-process, nothing over the network). CI runs the same commands on
every push/PR via GitHub Actions (`.github/workflows/ci.yml`) — `uv sync`,
`mypy --strict`, `pytest`.

- **Item 1** — async LLM client core: dual rpm/tpm token-bucket rate
  limiting, retry with backoff+jitter on 429/5xx, OpenAI-compatible
  `base_url`.
- **Item 2** — structured extraction: pydantic-schema-validated output,
  required `FailureMode` (`RAISE`/`NONE`) at every call site, validation
  retries independent of the client's HTTP retry budget.
- **Item 3** — Polars bridge: `semantic_extract()`/`semantic_extract_async()`.
  DataFrame-level, not a lazy expression — `.collect()` first, call this,
  continue. Flat schemas only (`str`/`int`/`float`/`bool`, optionally
  `Optional[...]`); nested models/lists/dicts raise `NotImplementedError`
  by design.
- **Item 4** — `MetricsCollector`/`PipelineMetrics`: optional, attached
  via `LLMClient(config, metrics=...)`. Tracks calls, failures, token
  totals, and cost — `cost_complete` is `False` if any recorded call had
  no `price_per_1k` configured, so a partial total can't silently pass
  as a full one.
- **Item 5** — `sim_join`: two explicit methods, no silent magic.
  `method="exact"` (default) is the same cosine top-k as before, now
  chunked over the right side (`chunk_size`, default 4096) so memory is
  `O(n_left × chunk_size)` instead of `O(n_left × n_right)` — compute is
  still quadratic, only the memory cliff is fixed. `method="approximate"`
  uses an HNSW index via `usearch` for sub-quadratic query time, at the
  cost of small/tunable recall loss. **Sixth dependency:** `usearch`,
  added deliberately for this — picked over `faiss-cpu` because it ships
  prebuilt CPU wheels with no MKL/OpenBLAS pull-in, benchmarks faster
  than FAISS on CPU-only hardware, and has native Go bindings (worth
  noting given the stack, though nothing here builds toward that — it's
  available, not designed-for). Distance/similarity conversion
  (`similarity = 1 - distance` for `metric='cos'`) was verified against
  an installed `usearch==2.25.3`, not assumed from training data.
- **Item 6** — `write_snapshot`/`read_latest`/`rollback`: timestamped
  parquet files + a `manifest.json` pointer per table. Rollback
  repoints the pointer at an older snapshot without deleting anything,
  so it's itself reversible. No locking — concurrent writers to the
  same table race on the manifest, last write wins. Fine for
  single-process pipeline runs; not scoped for concurrent-writer safety.
- **Item 7** — CI: GitHub Actions, `uv sync --all-extras --group dev`
  → `mypy --strict` → `pytest`. No Jenkins, per your stack.
