import numpy as np
import pytest
import polars as pl

from lattice import sim_join


def test_sim_join_finds_exact_match_as_top_result():
    left = pl.DataFrame({"id": [1], "embedding": [[1.0, 0.0, 0.0]]})
    right = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
        }
    )
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=2
    )
    assert result.height == 2
    # exact match (id 10) should be first, similarity ~1.0
    assert result["right_id"].to_list()[0] == 10
    assert result["similarity"].to_list()[0] == pytest.approx(1.0)
    # orthogonal vector (id 11) should not appear in top-2 ahead of the 0.7/0.7 vector
    assert result["right_id"].to_list()[1] == 12


def test_sim_join_respects_k():
    left = pl.DataFrame({"id": [1], "embedding": [[1.0, 0.0]]})
    right = pl.DataFrame({"id": [10, 11, 12], "embedding": [[1.0, 0.0], [0.9, 0.1], [0.1, 0.9]]})
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=1
    )
    assert result.height == 1
    assert result["right_id"].to_list() == [10]


def test_sim_join_caps_k_at_right_height():
    left = pl.DataFrame({"id": [1], "embedding": [[1.0, 0.0]]})
    right = pl.DataFrame({"id": [10], "embedding": [[1.0, 0.0]]})
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=5
    )
    assert result.height == 1  # only one right row exists, can't return 5


def test_sim_join_handles_zero_vector_without_raising():
    left = pl.DataFrame({"id": [1], "embedding": [[0.0, 0.0]]})
    right = pl.DataFrame({"id": [10], "embedding": [[1.0, 0.0]]})
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=1
    )
    assert result["similarity"].to_list() == [0.0]


def test_sim_join_prefixes_columns_to_avoid_collision():
    left = pl.DataFrame({"id": [1], "label": ["L"], "embedding": [[1.0, 0.0]]})
    right = pl.DataFrame({"id": [10], "label": ["R"], "embedding": [[1.0, 0.0]]})
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=1
    )
    assert set(result.columns) == {
        "left_id",
        "left_label",
        "left_embedding",
        "right_id",
        "right_label",
        "right_embedding",
        "similarity",
    }
    assert result["left_id"].to_list() == [1]
    assert result["right_id"].to_list() == [10]


def test_sim_join_rejects_invalid_k():
    left = pl.DataFrame({"embedding": [[1.0]]})
    right = pl.DataFrame({"embedding": [[1.0]]})
    with pytest.raises(ValueError):
        sim_join(left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=0)


def test_sim_join_rejects_dimension_mismatch():
    left = pl.DataFrame({"embedding": [[1.0, 0.0]]})
    right = pl.DataFrame({"embedding": [[1.0, 0.0, 0.0]]})
    with pytest.raises(ValueError):
        sim_join(left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=1)


def test_sim_join_empty_input_returns_empty_frame():
    left = pl.DataFrame({"embedding": []}, schema={"embedding": pl.List(pl.Float64)})
    right = pl.DataFrame({"embedding": [[1.0]]})
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=1
    )
    assert result.height == 0


def test_sim_join_chunked_matches_unchunked_with_tiny_chunk_size():
    """chunk_size smaller than n_right forces multiple merge rounds -
    confirms the running best-k survives across chunk boundaries."""
    rng = np.random.default_rng(42)
    n_right = 23
    left = pl.DataFrame({"id": [0], "embedding": [rng.normal(size=8).tolist()]})
    right = pl.DataFrame(
        {
            "id": list(range(n_right)),
            "embedding": [rng.normal(size=8).tolist() for _ in range(n_right)],
        }
    )
    full = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding",
        k=5, chunk_size=1000,
    )
    tiny_chunks = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding",
        k=5, chunk_size=3,  # forces 8 merge rounds for 23 right rows
    )
    assert full["right_id"].to_list() == tiny_chunks["right_id"].to_list()
    assert full["similarity"].to_list() == pytest.approx(tiny_chunks["similarity"].to_list())


def test_sim_join_chunk_boundary_does_not_lose_the_best_match():
    """Regression guard: put the true best match in the very first chunk,
    confirm it survives being merged-and-reselected across later chunks
    rather than getting evicted by a partial top-k bug."""
    rng = np.random.default_rng(7)
    left = pl.DataFrame({"id": [0], "embedding": [[1.0, 0.0]]})
    # first row is the exact match; everything after is deliberately worse
    right_vecs = [[1.0, 0.0]] + [
        (rng.normal(size=2) * 0.01).tolist() for _ in range(50)
    ]
    right = pl.DataFrame({"id": list(range(51)), "embedding": right_vecs})
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding",
        k=1, chunk_size=4,
    )
    assert result["right_id"].to_list() == [0]
    assert result["similarity"].to_list()[0] == pytest.approx(1.0)


def test_sim_join_rejects_unknown_method():
    left = pl.DataFrame({"embedding": [[1.0]]})
    right = pl.DataFrame({"embedding": [[1.0]]})
    with pytest.raises(ValueError):
        sim_join(
            left, right, left_embedding_col="embedding", right_embedding_col="embedding",
            k=1, method="magic",  # type: ignore[arg-type]
        )


def test_sim_join_approximate_finds_exact_match_as_top_result():
    left = pl.DataFrame({"id": [1], "embedding": [[1.0, 0.0, 0.0]]})
    right = pl.DataFrame(
        {
            "id": [10, 11, 12],
            "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7, 0.7, 0.0]],
        }
    )
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding",
        k=2, method="approximate",
    )
    assert result.height == 2
    assert result["right_id"].to_list()[0] == 10
    assert result["similarity"].to_list()[0] == pytest.approx(1.0, abs=1e-5)


def test_sim_join_approximate_agrees_with_exact_on_well_separated_clusters():
    """HNSW recall isn't guaranteed at any k, but on well-separated
    clusters with generous ef it should agree with brute force - this
    is a sanity check, not a recall guarantee."""
    rng = np.random.default_rng(123)
    n_right = 200
    dim = 16
    right_vecs = rng.normal(size=(n_right, dim))
    left_vecs = rng.normal(size=(5, dim))

    left = pl.DataFrame({"id": list(range(5)), "embedding": left_vecs.tolist()})
    right = pl.DataFrame({"id": list(range(n_right)), "embedding": right_vecs.tolist()})

    exact = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding", k=1
    )
    approx = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding",
        k=1, method="approximate",
    )
    # top-1 nearest neighbor should agree for at least most of the 5 queries
    # on a dataset this small - not asserting all 5 to avoid HNSW-recall flakiness.
    agreement = sum(
        a == b for a, b in zip(exact["right_id"].to_list(), approx["right_id"].to_list())
    )
    assert agreement >= 4


def test_sim_join_approximate_caps_k_at_right_height():
    left = pl.DataFrame({"id": [1], "embedding": [[1.0, 0.0]]})
    right = pl.DataFrame({"id": [10], "embedding": [[1.0, 0.0]]})
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding",
        k=5, method="approximate",
    )
    assert result.height == 1


def test_sim_join_approximate_handles_multiple_left_rows():
    """Exercises the BatchMatches code path (n_left > 1), not just the
    single-query Matches path."""
    left = pl.DataFrame(
        {"id": [1, 2], "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    )
    right = pl.DataFrame(
        {"id": [10, 11], "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
    )
    result = sim_join(
        left, right, left_embedding_col="embedding", right_embedding_col="embedding",
        k=1, method="approximate",
    )
    assert result.height == 2
    by_left = dict(zip(result["left_id"].to_list(), result["right_id"].to_list(), strict=True))
    assert by_left == {1: 10, 2: 11}
