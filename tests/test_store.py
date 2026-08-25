"""Schema migration: an existing database must gain new columns, not break."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.store import DONE, QUEUED, Store  # noqa: E402

# The schema as it shipped before rush support existed.
OLD_SCHEMA = """
CREATE TABLE jobs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    video        TEXT NOT NULL,
    title        TEXT DEFAULT '',
    series_key   TEXT DEFAULT '',
    status       TEXT NOT NULL,
    origin       TEXT DEFAULT '',
    detail       TEXT DEFAULT '',
    output       TEXT DEFAULT '',
    error        TEXT DEFAULT '',
    progress     REAL DEFAULT 0,
    stage        TEXT DEFAULT '',
    stats        TEXT DEFAULT '{}',
    usage        TEXT DEFAULT '{}',
    trigger      TEXT DEFAULT '',
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL
);
"""


def test_an_old_database_gains_the_new_columns(tmp_path):
    path = tmp_path / "old.db"
    db = sqlite3.connect(path)
    db.executescript(OLD_SCHEMA)
    db.execute("INSERT INTO jobs (video,status,created_at) VALUES ('/m/a.mkv','done',1.0)")
    db.commit()
    db.close()

    store = Store(path)                     # must migrate, not explode
    job = store.recent(1)[0]
    assert job["video"] == "/m/a.mkv"
    assert job["provider"] == "" and job["priority"] == 0

    # and the new features work against the migrated table
    new_id = store.enqueue("/m/b.mkv", provider="anthropic", priority=1)
    assert store.get(new_id)["priority"] == 1


def test_migration_is_idempotent(tmp_path):
    path = tmp_path / "fresh.db"
    Store(path)
    store = Store(path)
    assert store.enqueue("/m/c.mkv") is not None


def test_priority_jobs_are_claimed_first(tmp_path):
    store = Store(tmp_path / "q.db")
    store.enqueue("/m/normal.mkv")
    rush = store.enqueue("/m/rush.mkv", provider="anthropic", priority=1)
    assert store.claim_next()["id"] == rush


def test_the_rush_lane_ignores_ordinary_jobs(tmp_path):
    store = Store(tmp_path / "q2.db")
    store.enqueue("/m/normal.mkv")
    assert store.claim_next(priority_only=True) is None

    rush = store.enqueue("/m/rush.mkv", priority=1)
    assert store.claim_next(priority_only=True)["id"] == rush


def test_requeue_can_switch_provider(tmp_path):
    store = Store(tmp_path / "q3.db")
    first = store.enqueue("/m/x.mkv")
    store.finish(first, DONE)
    second = store.requeue(first, provider="anthropic", priority=1)
    row = store.get(second)
    assert row["provider"] == "anthropic" and row["priority"] == 1
    assert row["trigger"] == "rush" and row["status"] == QUEUED
