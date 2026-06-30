"""sim_join: cosine-similarity k-NN over two embedded columns.

Two methods, chosen explicitly by the caller - no silent magic:

  method="exact" (default): same brute-force cosine top-k as before, but
    now chunked over the right side so memory is O(n_left * chunk_size)
    instead of O(n_left * n_right). Still O(n_left * n_right) compute -
    chunking fixes the memory cliff, not the asymptotic cost. Results
    are exact and deterministic.

  method="approximate": HNSW via usearch. Genuinely sub-quadratic at
    query time once the index is built. Not exact - HNSW trades a small,
    tunable recall loss for speed; results may omit a true top-k neighbor
    occasionally, especially at low `expansion_search`.

Picked usearch over faiss-cpu: usearch ships prebuilt CPU wheels with no
MKL/OpenBLAS dependency (faiss-cpu typically pulls one in), is
benchmarked faster than FAISS on CPU-only hardware, and - notably for a
Go-heavy stack - has native Go bindings, so an index built here could in
principle be reused from Go tooling later without round-tripping through
Python. That's a real consideration, not just a numbers thing, but
flagging it as a "nice to have noticed," not a designed-for integration
point - nothing here builds towards that, it's just available if wanted.

usearch's metric='cos' returns *distance*, not similarity: similarity =
1 - distance. Verified empirically against an installed usearch 2.25.3,
not assumed from memory - see the distance-semantics check before this
was written. Zero-vector queries return distance 1.0 (similarity 0.0)
without raising, matching the exact path's zero-vector handling.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import numpy.typing as npt
import polars as pl

_DEFAULT_CHUNK_SIZE = 4096

FloatArray = npt.NDArray[np.float64]


def _validate_inputs(
    left: pl.DataFrame, right: pl.DataFrame, left_embedding_col: str, right_embedding_col: str, k: int
) -> tuple[FloatArray, FloatArray] | None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if left.height == 0 or right.height == 0:
        return None
    left_emb: FloatArray = np.asarray(left.get_column(left_embedding_col).to_list(), dtype=np.float64)
    right_emb: FloatArray = np.asarray(
        right.get_column(right_embedding_col).to_list(), dtype=np.float64
    )
    if left_emb.ndim != 2 or right_emb.ndim != 2 or left_emb.shape[1] != right_emb.shape[1]:
        raise ValueError(
            f"embedding dimension mismatch: left {left_emb.shape} vs right {right_emb.shape}"
        )
    return left_emb, right_emb


def _unit(vectors: FloatArray) -> FloatArray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    result: FloatArray = np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms != 0)
    return result


def _assemble(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    top_idx_per_row: list[npt.NDArray[np.int64]],
    top_sim_per_row: list[FloatArray],
    similarity_col: str,
    left_prefix: str,
    right_prefix: str,
) -> pl.DataFrame:
    left_dicts = left.to_dicts()
    right_dicts = right.to_dicts()
    out_rows: list[dict[str, object]] = []
    for i, ldict in enumerate(left_dicts):
        idxs = top_idx_per_row[i]
        sims = top_sim_per_row[i]
        for ridx, sim in zip(idxs, sims, strict=True):
            combined: dict[str, object] = {f"{left_prefix}{key}": val for key, val in ldict.items()}
            combined.update(
                {f"{right_prefix}{key}": val for key, val in right_dicts[int(ridx)].items()}
            )
            combined[similarity_col] = float(sim)
            out_rows.append(combined)
    return pl.DataFrame(out_rows)


def _sim_join_exact_chunked(
    left_emb: FloatArray,
    right_emb: FloatArray,
    *,
    k: int,
    chunk_size: int,
) -> tuple[list[npt.NDArray[np.int64]], list[FloatArray]]:
    """Exact cosine top-k, processing the right side in chunks so peak
    memory is O(n_left * chunk_size) rather than O(n_left * n_right).

    Running best-so-far per left row is carried across chunks: after each
    chunk, merge its candidates with the current best-k and re-select the
    top-k. Compute is still O(n_left * n_right) - this bounds memory, not
    asymptotic time.
    """
    left_unit = _unit(left_emb)
    right_unit = _unit(right_emb)
    n_left = left_unit.shape[0]
    n_right = right_unit.shape[0]
    eff_k = min(k, n_right)

    best_sims = np.full((n_left, 0), -np.inf, dtype=np.float64)
    best_idx = np.full((n_left, 0), -1, dtype=np.int64)

    for start in range(0, n_right, chunk_size):
        end = min(start + chunk_size, n_right)
        chunk = right_unit[start:end]  # (chunk_n, dim)
        chunk_sims = left_unit @ chunk.T  # (n_left, chunk_n)
        chunk_idx = np.arange(start, end, dtype=np.int64)[None, :].repeat(n_left, axis=0)

        merged_sims = np.concatenate([best_sims, chunk_sims], axis=1)
        merged_idx = np.concatenate([best_idx, chunk_idx], axis=1)

        width = merged_sims.shape[1]
        take = min(eff_k, width)
        # argpartition per row for the new best-`take`, then keep them
        # (not yet fully sorted - sorted once at the very end).
        part = np.argpartition(-merged_sims, take - 1, axis=1)[:, :take]
        best_sims = np.take_along_axis(merged_sims, part, axis=1)
        best_idx = np.take_along_axis(merged_idx, part, axis=1)

    order = np.argsort(-best_sims, axis=1)
    sorted_sims = np.take_along_axis(best_sims, order, axis=1)
    sorted_idx = np.take_along_axis(best_idx, order, axis=1)

    return [sorted_idx[i] for i in range(n_left)], [sorted_sims[i] for i in range(n_left)]


def _sim_join_approximate(
    left_emb: FloatArray,
    right_emb: FloatArray,
    *,
    k: int,
) -> tuple[list[npt.NDArray[np.int64]], list[FloatArray]]:
    from usearch.index import BatchMatches, Index, Matches  # local: only needed for this path

    n_right = right_emb.shape[0]
    eff_k = min(k, n_right)
    dim = right_emb.shape[1]

    index: Index = Index(ndim=dim, metric="cos")
    index.add(np.arange(n_right, dtype=np.int64), right_emb)

    matches = index.search(left_emb, eff_k)
    # usearch returns a single Matches for a single query vector and a
    # BatchMatches (indexable to per-query Matches) for a 2D batch -
    # narrow on the actual returned type rather than inferring it from
    # left_emb's shape, so this stays correct if usearch's own dispatch
    # rule ever differs from "ndim==1 means single query".
    matches_list: list[Matches]
    if isinstance(matches, BatchMatches):
        matches_list = [matches[i] for i in range(len(matches))]
    else:
        matches_list = [matches]

    top_idx_per_row = [np.asarray(m.keys, dtype=np.int64) for m in matches_list]
    top_sim_per_row = [1.0 - np.asarray(m.distances, dtype=np.float64) for m in matches_list]
    return top_idx_per_row, top_sim_per_row


def sim_join(
    left: pl.DataFrame,
    right: pl.DataFrame,
    *,
    left_embedding_col: str,
    right_embedding_col: str,
    k: int,
    method: Literal["exact", "approximate"] = "exact",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    similarity_col: str = "similarity",
    left_prefix: str = "left_",
    right_prefix: str = "right_",
) -> pl.DataFrame:
    """For each row in `left`, find the top-k most similar rows in `right`
    by cosine similarity of their embedding columns (list[float] or
    array[float] columns of equal width).

    method="exact" (default) is deterministic and chunked for bounded
    memory. method="approximate" uses an HNSW index (usearch) and trades
    a small, tunable recall loss for sub-quadratic query time - pick it
    deliberately for large `right` tables, not by default.

    Output columns are every column of `left` prefixed with `left_prefix`,
    every column of `right` prefixed with `right_prefix`, plus
    `similarity_col`. One output row per (left row, matched right row)
    pair, ordered by left row then descending similarity.

    Zero vectors get similarity 0.0 against everything rather than
    raising or producing NaN, in both methods.
    """
    validated = _validate_inputs(left, right, left_embedding_col, right_embedding_col, k)
    if validated is None:
        return pl.DataFrame()
    left_emb, right_emb = validated

    if method == "exact":
        top_idx, top_sim = _sim_join_exact_chunked(left_emb, right_emb, k=k, chunk_size=chunk_size)
    elif method == "approximate":
        top_idx, top_sim = _sim_join_approximate(left_emb, right_emb, k=k)
    else:
        raise ValueError(f"method must be 'exact' or 'approximate', got {method!r}")

    return _assemble(
        left,
        right,
        top_idx_per_row=top_idx,
        top_sim_per_row=top_sim,
        similarity_col=similarity_col,
        left_prefix=left_prefix,
        right_prefix=right_prefix,
    )
