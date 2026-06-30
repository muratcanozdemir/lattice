from datetime import UTC, datetime

import polars as pl
import pytest
import polars.testing as plt

from lattice import list_snapshots, read_latest, rollback, write_snapshot


def test_write_then_read_latest(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3]})
    write_snapshot(df, table="t1", root=tmp_path)
    result = read_latest("t1", root=tmp_path)
    plt.assert_frame_equal(result, df)


def test_multiple_writes_latest_points_at_most_recent(tmp_path):
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"a": [2]})
    write_snapshot(
        df1, table="t1", root=tmp_path, timestamp=datetime(2025, 1, 1, tzinfo=UTC)
    )
    write_snapshot(
        df2, table="t1", root=tmp_path, timestamp=datetime(2025, 1, 2, tzinfo=UTC)
    )
    result = read_latest("t1", root=tmp_path)
    plt.assert_frame_equal(result, df2)
    assert len(list_snapshots("t1", root=tmp_path)) == 2


def test_rollback_repoints_latest_without_deleting_history(tmp_path):
    df1 = pl.DataFrame({"a": [1]})
    df2 = pl.DataFrame({"a": [2]})
    write_snapshot(
        df1, table="t1", root=tmp_path, timestamp=datetime(2025, 1, 1, tzinfo=UTC)
    )
    write_snapshot(
        df2, table="t1", root=tmp_path, timestamp=datetime(2025, 1, 2, tzinfo=UTC)
    )
    first_snapshot = list_snapshots("t1", root=tmp_path)[0]

    rollback("t1", root=tmp_path, to_snapshot=first_snapshot)
    result = read_latest("t1", root=tmp_path)
    plt.assert_frame_equal(result, df1)
    # history is untouched - both snapshots still listed, rollback is reversible
    assert len(list_snapshots("t1", root=tmp_path)) == 2


def test_rollback_to_unknown_snapshot_raises(tmp_path):
    df = pl.DataFrame({"a": [1]})
    write_snapshot(df, table="t1", root=tmp_path)
    with pytest.raises(ValueError):
        rollback("t1", root=tmp_path, to_snapshot="nonexistent.parquet")


def test_read_latest_on_unknown_table_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_latest("nonexistent_table", root=tmp_path)


def test_tables_are_isolated(tmp_path):
    df_a = pl.DataFrame({"a": [1]})
    df_b = pl.DataFrame({"b": [2]})
    write_snapshot(df_a, table="table_a", root=tmp_path)
    write_snapshot(df_b, table="table_b", root=tmp_path)
    plt.assert_frame_equal(read_latest("table_a", root=tmp_path), df_a)
    plt.assert_frame_equal(read_latest("table_b", root=tmp_path), df_b)
