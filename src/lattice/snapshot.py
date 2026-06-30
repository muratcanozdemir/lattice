"""Snapshot/manifest write helper.

This is deliberately the simplest thing that works: write a timestamped
parquet file, update a manifest.json pointing at it as "latest". Rollback
is repointing the manifest at an older snapshot filename - no table
storage engine, no special format, just files + a pointer.

One directory per table: {root}/{table}/{table}_{timestamp}.parquet,
plus {root}/{table}/manifest.json. Concurrent writers to the SAME table
will race on manifest.json (last write wins, no locking) - this is fine
for the single-process pipeline-run usage this is meant for. If you need
concurrent-writer safety, that's a different (and heavier) problem than
this helper is scoped to solve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%S%fZ"


@dataclass
class SnapshotManifest:
    latest: str
    snapshots: list[str]


def _manifest_path(root: Path, table: str) -> Path:
    return root / table / "manifest.json"


def _read_manifest(root: Path, table: str) -> SnapshotManifest | None:
    path = _manifest_path(root, table)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return SnapshotManifest(latest=data["latest"], snapshots=data["snapshots"])


def _write_manifest(root: Path, table: str, manifest: SnapshotManifest) -> None:
    path = _manifest_path(root, table)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"latest": manifest.latest, "snapshots": manifest.snapshots}, indent=2
        )
    )


def write_snapshot(
    df: pl.DataFrame,
    *,
    table: str,
    root: Path | str,
    timestamp: datetime | None = None,
) -> Path:
    """Writes df as a new timestamped snapshot and points 'latest' at it.
    Returns the path written."""
    root = Path(root)
    ts = timestamp or datetime.now(UTC)
    filename = f"{table}_{ts.strftime(_TIMESTAMP_FORMAT)}.parquet"
    table_dir = root / table
    table_dir.mkdir(parents=True, exist_ok=True)
    file_path = table_dir / filename
    df.write_parquet(file_path)

    existing = _read_manifest(root, table)
    snapshots = existing.snapshots if existing else []
    snapshots.append(filename)
    _write_manifest(root, table, SnapshotManifest(latest=filename, snapshots=snapshots))
    return file_path


def read_latest(table: str, root: Path | str) -> pl.DataFrame:
    root = Path(root)
    manifest = _read_manifest(root, table)
    if manifest is None:
        raise FileNotFoundError(f"no manifest for table {table!r} under {root}")
    return pl.read_parquet(root / table / manifest.latest)


def list_snapshots(table: str, root: Path | str) -> list[str]:
    root = Path(root)
    manifest = _read_manifest(root, table)
    if manifest is None:
        return []
    return list(manifest.snapshots)


def rollback(table: str, root: Path | str, *, to_snapshot: str) -> None:
    """Repoints 'latest' at an existing, older snapshot file. Does not
    delete or rewrite anything - the rolled-back-from snapshot stays on
    disk and in history, so this is itself reversible."""
    root = Path(root)
    manifest = _read_manifest(root, table)
    if manifest is None:
        raise FileNotFoundError(f"no manifest for table {table!r} under {root}")
    if to_snapshot not in manifest.snapshots:
        raise ValueError(
            f"{to_snapshot!r} is not a known snapshot of {table!r}: {manifest.snapshots}"
        )
    _write_manifest(
        root, table, SnapshotManifest(latest=to_snapshot, snapshots=manifest.snapshots)
    )
