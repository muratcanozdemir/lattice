"""Bridge between LLMClient/extract() and Polars DataFrames.

The fiddly part, as flagged in the charter: Polars' `map_elements` and
`map_batches` callbacks are synchronous. They're called from Polars'
own (possibly multi-threaded) execution, not from an asyncio event loop.
There is no way to `await` inside one without either:

  (a) running a fresh event loop per callback invocation (defeats the
      purpose - you lose cross-row concurrency, every row becomes a
      serial round trip), or
  (b) stepping outside the lazy/expression API entirely: pull the column
      out as a list, run one asyncio.gather() over the whole batch with
      the client's own semaphore providing the concurrency bound, then
      reattach the results as a column.

This module does (b). It is therefore a DataFrame-level function, not a
Polars expression/plugin - you cannot compose it inside a
`.select(fc.semantic.extract(...))`-style lazy expression chain the way
fenic does, because that would require solving (a). If you want it
inside a lazy pipeline, materialize up to this point with `.collect()`
first, call this function, then continue.

semantic_extract() wraps asyncio.run() for convenience at a top-level
call site. If you're already inside a running event loop (a Jupyter
kernel, an async web handler, etc.), asyncio.run() will raise - use
semantic_extract_async() directly and await it instead.
"""

from __future__ import annotations

import asyncio
import types
import typing
from typing import Any

import polars as pl
from pydantic import BaseModel

from lattice.client import LLMClient
from lattice.extract import FailureMode, extract

_PRIMITIVE_DTYPE_MAP: dict[type, pl.DataType] = {
    str: pl.Utf8(),
    int: pl.Int64(),
    float: pl.Float64(),
    bool: pl.Boolean(),
}


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    """Returns (inner_type, is_optional) for `T | None` / `Optional[T]`."""
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _field_dtype(annotation: Any) -> pl.DataType:
    inner, _ = _unwrap_optional(annotation)
    dtype = _PRIMITIVE_DTYPE_MAP.get(inner)
    if dtype is None:
        raise NotImplementedError(
            f"semantic_extract supports flat schemas of str/int/float/bool "
            f"(optionally wrapped in Optional) only; got field type {inner!r}. "
            f"Nested models, lists, and dicts aren't supported - this is a "
            f"deliberate scope cut, not an oversight. Flatten the schema or "
            f"post-process with regular Polars struct/unnest operations."
        )
    return dtype


def _struct_dtype(schema: type[BaseModel]) -> pl.Struct:
    fields = {
        name: _field_dtype(info.annotation)
        for name, info in schema.model_fields.items()
    }
    return pl.Struct(fields)


def _model_to_dict(model: BaseModel | None, schema: type[BaseModel]) -> dict[str, Any]:
    if model is None:
        return dict.fromkeys(schema.model_fields)
    return model.model_dump()


async def semantic_extract_async(
    df: pl.DataFrame,
    *,
    text_column: str,
    output_column: str,
    client: LLMClient,
    schema: type[BaseModel],
    prompt_template: str,
    failure_mode: FailureMode,
    max_validation_retries: int = 2,
    system_prompt: str | None = None,
) -> pl.DataFrame:
    """Async entry point - await this directly if you're already in a loop.

    prompt_template is a str.format() template; the source text for each
    row is substituted as {text}. Cross-row concurrency is bounded by
    client's own max_concurrency/rpm/tpm, not by anything in this module -
    all rows are submitted via asyncio.gather and the client's semaphore
    does the throttling.
    """
    texts = df.get_column(text_column).to_list()

    async def _one(text: str) -> BaseModel | None:
        prompt = prompt_template.format(text=text)
        return await extract(
            client,
            prompt=prompt,
            schema=schema,
            failure_mode=failure_mode,
            max_validation_retries=max_validation_retries,
            system_prompt=system_prompt,
        )

    results = await asyncio.gather(*[_one(t) for t in texts])

    dicts = [_model_to_dict(r, schema) for r in results]
    struct_series = pl.Series(output_column, dicts, dtype=_struct_dtype(schema))
    return df.with_columns(struct_series)


def semantic_extract(
    df: pl.DataFrame,
    *,
    text_column: str,
    output_column: str,
    client: LLMClient,
    schema: type[BaseModel],
    prompt_template: str,
    failure_mode: FailureMode,
    max_validation_retries: int = 2,
    system_prompt: str | None = None,
) -> pl.DataFrame:
    """Sync convenience wrapper. Raises RuntimeError if called from inside
    a running event loop - use semantic_extract_async() there instead."""
    return asyncio.run(
        semantic_extract_async(
            df,
            text_column=text_column,
            output_column=output_column,
            client=client,
            schema=schema,
            prompt_template=prompt_template,
            failure_mode=failure_mode,
            max_validation_retries=max_validation_retries,
            system_prompt=system_prompt,
        )
    )
