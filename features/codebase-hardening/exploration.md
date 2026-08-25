# Codebase Exploration: Codebase Hardening

**Generated:** 2026-08-25 20:00
**Feature:** Audit `lattice` for bugs, test-coverage gaps, general improvements, and CI/CD issues. Not a new feature — a hardening pass over the existing library (`src/lattice/`, 1029 lines across 7 modules; `tests/`, 977 lines across 6 files, 44 tests, 97% line coverage).

**Baseline verified locally** (no `gh` auth available, so CI status was reproduced by running the exact commands `.github/workflows/ci.yml` runs, not inferred):
- `uv sync --all-extras --group dev` — clean
- `uv run mypy --strict src/lattice` — `Success: no issues found in 7 source files`
- `uv run pytest -v` — `44 passed`
- `uvx black --check src/lattice tests` — `13 files would be left unchanged` (but see CI/CD findings — this step is not as safe as it looks)

So the pipeline is currently green. There's nothing actively broken to "fix" in the sense of a failing run; the CI/CD findings below are about fragility that hasn't bitten yet (unpinned tool versions, no least-privilege permissions), consistent with the repo's `fixed sbom` × 3 commit history — the release workflow has already needed multiple after-the-fact fixes for exactly this class of issue.

## Confirmed Bugs

All three reproduced locally with a minimal repro script (not just read from the source).

### 1. `LLMClient.acomplete` crashes on `"usage": null` in the response body (`src/lattice/client.py:175`)

```python
usage_raw = data.get("usage", {})   # only substitutes {} if the key is MISSING
usage = Usage(prompt_tokens=usage_raw.get("prompt_tokens", 0), ...)  # AttributeError if usage_raw is None
```

`dict.get(key, default)` only falls back to `default` when the key is absent — not when it's present with value `None`. Several real OpenAI-compatible servers (older llama.cpp builds, some proxy layers, certain streaming-adjacent code paths) send `"usage": null` rather than omitting the field. Reproduced: mocking a 200 response with `"usage": null` raises `AttributeError: 'NoneType' object has no attribute 'get'` — an unhandled exception, not the library's own `LLMError`. It also happens *after* `_request_with_retry` returns successfully, so it isn't caught by the `except LLMError` wrapper around that call (`client.py:168-173`) — `metrics.record_failure()` never fires either, so a failure here isn't even visible in `MetricsCollector` if one is attached.

### 2. `LLMClient.acomplete` crashes on `content: None` in an input message (`src/lattice/client.py:149`)

```python
prompt_text = " ".join(m.get("content", "") for m in messages)
```

Same `.get(key, default)` pitfall. OpenAI-style tool-calling puts `content: None` on assistant messages that only carry `tool_calls`. Reproduced: passing `[{"role": "assistant", "content": None}, ...]` raises `TypeError: sequence item 0: expected str instance, NoneType found` before any HTTP request is even sent. `lattice` doesn't claim to support tool-calling today, but nothing stops a caller from passing multi-turn history that includes a prior assistant turn shaped this way, and the failure mode is an unhandled `TypeError`, not a documented `LLMError`.

### 3. `sim_join(..., chunk_size=<negative>)` silently returns a wrong (empty) result instead of raising (`src/lattice/sim_join.py:136`)

`chunk_size=0` fails loudly (`ValueError: range() arg 3 must not be zero` from `range(0, n_right, chunk_size)`), but `chunk_size=-1` (or any negative value) makes `range(0, n_right, chunk_size)` yield zero iterations — the loop body never runs, `best_sims`/`best_idx` stay at their initial `(n_left, 0)` shape, and `sim_join` returns `pl.DataFrame()` (empty, zero columns) with no error and no warning, indistinguishable from the documented "empty input" case. Reproduced with 2 left rows × 3 right rows, `chunk_size=-1`, `k=2` → silently returns an empty 0×0 frame instead of the expected 4-row result. There's no validation on `chunk_size` at all (`_validate_inputs` only checks `k` and dimension match).

## Test Coverage Gaps

Overall line coverage is 97% (406 stmts / 12 missed) via `pytest-cov` (not a project dependency — installed ad hoc via `uv run --with pytest-cov --with coverage`, since neither is in `pyproject.toml`'s dev group). The misses cluster around exactly the paths least tested by the existing suite, several of which overlap the bugs above:

- **`client.py:60`** — `_TokenBucket.acquire`'s early-return when a single request's `amount` exceeds bucket `capacity`. No test ever requests more than capacity in one call.
- **`client.py:67-69`** — the actual *wait* branch of `_TokenBucket.acquire` (deficit computed, `asyncio.sleep` invoked). Existing rate-limit-adjacent test (`test_concurrency_bound_is_respected`) uses `rpm=10_000, tpm=10_000_000` specifically so the bucket never blocks; nothing exercises real throttling.
- **`client.py:124`** — `Authorization` header construction. No test in `test_client.py` ever sets `api_key` on `ClientConfig`.
- **`client.py:161`, `165`** — `max_tokens` and `extra_body` being merged into the request payload. No test passes either kwarg to `acomplete`, so the payload-shaping branches are untested (and `extra_body`'s "caller can override any top-level payload key including `model`/`messages`" behavior — `payload.update(extra_body)` — is undocumented and untested).
- **`client.py:234-235`** — the `except (httpx.TimeoutException, httpx.TransportError)` retry branch. Existing retry tests (`test_retries_on_429_then_succeeds`, `test_exhausts_retries_and_raises`) only exercise HTTP-status-based retries via `respx` mock responses; nothing simulates a connection error or timeout to prove the transport-exception retry path actually works.
- **`extract.py:117`** — `messages.append({"role": "system", ...})` when `system_prompt` is passed to `extract()`. No test in `test_extract.py` passes `system_prompt`.
- **`snapshot.py:92`** — `list_snapshots()` returning `[]` for a table with no manifest. Untested.
- **`snapshot.py:103`** — `rollback()` raising `FileNotFoundError` for a table with no manifest. Untested (only `rollback` to an *unknown snapshot of an existing table* is tested, at `test_rollback_to_unknown_snapshot_raises`).

Additionally — not a coverage-tool miss, but a real gap — **none of the three confirmed bugs above have a regression test**, which is exactly why they weren't caught: `usage: null`, `content: None`, and negative `chunk_size` are all inputs the current suite never constructs.

## Improvements (non-bug)

- **`asyncio.gather` in `semantic_extract_async` (`polars_ext.py:121`) has no `return_exceptions=True`.** Under `FailureMode.RAISE`, if row 3 of 1000 fails validation after retries, `gather` raises immediately and propagates — the other 999 in-flight coroutines keep running in the background but their results (and any of *their* exceptions) are discarded, and asyncio will log "exception was never retrieved" warnings for any of them that also fail. Worth deciding deliberately: either document that a single `RAISE` failure aborts the whole batch (current behavior, but the in-flight-waste + orphaned-exception-warnings are a side effect, not a design choice), or collect partial results/exceptions explicitly.
- **`_parse_and_validate`'s markdown-fence stripping (`extract.py:90-94`) is narrow.** `stripped.strip("`")` correctly handles the documented case (```` ```json ... ``` ````) but only recognizes a lowercase `json` language tag after stripping backticks — a fence like ```` ```JSON ```` or ```` ``` ```` (no language tag, which is common) falls through differently: no tag → the code doesn't skip 4 chars but still proceeds to `json.loads` on whatever's left, which happens to work only because `.strip("`")` already removed the fence markers themselves. This works today but is fragile/under-tested — only `test_extract_strips_markdown_fences` exercises this at all, and only the documented-case shape.
- **Packaging: `pyproject.toml` splits dev tooling across two mechanisms.** `pytest`/`pytest-asyncio`/`respx` live in `[project.optional-dependencies].dev` (installed via `uv sync --all-extras`), while `mypy` lives in `[dependency-groups].dev` (installed via `uv sync --group dev`; PEP 735, not shipped as an installable extra). `optional-dependencies` extras are published in the wheel's metadata — meaning `pip install lattice[dev]` advertises and installs test-only dependencies (`pytest`, `respx`) to any downstream consumer of the published package, not just contributors. Consolidating all dev tooling into `[dependency-groups].dev` would keep the published package's extras surface clean (there currently are no legitimate user-facing extras) and match `uv`'s own recommended split between "things a user might opt into" and "things only a contributor needs."
- **Repo hygiene: `tests/__pycache__/*.pyc` (6 files) are tracked in git**, committed in the very first commit (`8586011 first`), compiled against `cpython-313` — a different interpreter than the `>=3.11` the project targets and than the `3.11.16` this environment resolved. There is **no `.gitignore` anywhere in the repo**, so every local `pytest`/`mypy` run risks re-polluting `git status` with `__pycache__/`, `.pytest_cache/`, `.mypy_cache/`, `.venv/`, and `.coverage` (all observed as untracked cruft during this exploration, cleaned up before writing this doc). Adding a standard Python `.gitignore` and removing the already-tracked `.pyc` files is a quick, safe fix.
- **`_TokenBucket.adjust`'s reconciliation (`client.py:196-198`) treats `usage.total_tokens == 0` as "unknown, don't reconcile."** `actual = float(usage.total_tokens) if usage.total_tokens else reserved` means a server that legitimately reports 0 total tokens (edge case, but possible for e.g. a cached/empty completion) is treated identically to a server that omitted usage entirely — the reservation is left untouched rather than corrected down. Minor, but worth a comment or explicit handling since it's a silent behavioral choice at a truthiness boundary (`0` vs missing).

## CI/CD

Reproduced the exact `ci.yml` steps locally (see baseline above) — the pipeline is currently green. Findings here are about robustness, not an active red build:

- **`uvx black --check` (`ci.yml`, last step) has no pinned Black version and no `--target-version`.** Running it locally surfaced: `Warning: Python 3.11 cannot parse code formatted for Python 3.15. To fix this: run Black with Python 3.15, set --target-version to py311, or use --fast to skip the safety check.` — `uvx` resolved the newest available Black, which defaults its target inference to the *running* interpreter rather than the project's `requires-python = ">=3.11"`. It happens to pass today, but the check's pass/fail behavior is not reproducible or pinned — a future Black release changing a default could flip this red (or silently accept code a truly-3.11-targeted run would reformat) with no corresponding change in this repo. Same unpinned-tool pattern as the `cyclonedx-bom` SBOM step in `release.yml`, which already needed three follow-up commits (`fixed sbom` ×3) to get working — worth fixing proactively here rather than reactively later. Pin a Black version (e.g. add it to `[dependency-groups].dev` instead of `uvx`-ing an arbitrary latest) and pass `--target-version py311`.
- **`ci.yml` has no `permissions:` block**, so the workflow's `GITHUB_TOKEN` gets whatever the repo/org default is (potentially read-write) even though this workflow only checks out code and runs tests — it needs no write access to anything. `release.yml`'s `publish` job does scope `permissions: contents: write, packages: write` explicitly, showing the pattern is already known in this repo; `ci.yml` (and `release.yml`'s `build`/`sbom` jobs) just don't use it yet. Least-privilege default (`permissions: contents: read` at the workflow level) is a low-cost hardening step.
- **Inconsistent action-pinning strategy between the two workflows.** `release.yml` pins third-party actions to full commit SHA with a version comment (`actions/upload-artifact@ea165f8d...# v4.6.2`, `softprops/action-gh-release@da05d552...# v2.2.2`) — good supply-chain practice. `ci.yml` uses floating major-version tags throughout (`actions/checkout@v4`, `astral-sh/setup-uv@v3`). Both workflows also use `actions/checkout@v4` unpinned. Worth applying the same SHA-pinning already adopted in `release.yml` consistently to `ci.yml`.
- **No `concurrency:` group on either workflow.** A rapid sequence of pushes to the same PR/branch queues and runs every CI invocation to completion rather than canceling superseded ones — pure wasted minutes on a project this size, more relevant as push frequency grows.
- **No Dependabot/Renovate config** (`.github/dependabot.yml` absent) despite `uv.lock` pinning transitive versions exactly — there's currently no automated path for picking up security patches in `polars`, `httpx`, `pydantic`, `numpy`, or `usearch` short of a human noticing.

## Cross-Cutting Patterns

- **The `.get(key, default)`-with-`None`-value pitfall appears twice independently** (`client.py:149` and `client.py:175`) — same root cause (conflating "key missing" with "key present but null"), same module, same function's call graph. A single defensive helper (e.g. `data.get("usage") or {}`) or a shared review pass for this pattern across the module would catch both at once rather than as two unrelated one-line fixes.
- **Validation is inconsistent within `sim_join`'s own parameter set**: `k` is validated (`_validate_inputs` raises `ValueError` for `k < 1`), but `chunk_size` — a parameter with an identical "must be a positive int" contract — isn't validated at all, leading to the silent-empty-result bug above. Any fix should validate `chunk_size` the same way `k` already is, for consistency with the existing convention in that same function.
- **The project's own documented philosophy (README "What this is not," module docstrings throughout) is "no silent magic," explicit failure modes, deliberate choices over inherited defaults** — e.g. `FailureMode` has no default specifically so callers can't inherit silent degradation. The `sim_join` negative-`chunk_size` bug and the two `.get()`-with-`None` bugs are direct violations of that stated design principle (silent wrong-answer / unhandled crash instead of an explicit, documented failure), which makes them a good match for how this codebase already wants to handle errors elsewhere — the fix pattern (raise clearly, document the contract) already exists in the same files as precedent (e.g. `_validate_inputs`'s `k` check, `ExtractionError`'s explicit failure classes).
