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


def test_a_job_interrupted_by_a_restart_is_requeued_not_failed(tmp_path):
    """A redeploy used to fail whatever was mid-flight. On a provider where one
    file takes over an hour, that threw away real work every single deploy."""
    path = tmp_path / "restart.db"
    store = Store(path)
    job = store.enqueue("/m/long.mkv")
    claimed = store.claim_next()
    assert claimed["id"] == job and claimed["status"] == "running"

    reopened = Store(path)              # simulates the container restarting
    row = reopened.get(job)
    assert row["status"] == QUEUED
    assert "restart" in row["stage"]
    assert reopened.claim_next()["id"] == job


def test_the_running_job_is_listed_first(tmp_path):
    """Newest-first buried the job doing work under everything queued since."""
    store = Store(tmp_path / "order.db")
    running = store.enqueue("/m/a.mkv")
    store.claim_next()
    for n in range(5):
        store.enqueue(f"/m/later{n}.mkv")

    assert store.recent(10)[0]["id"] != running                 # newest-first
    assert store.recent(10, running_first=True)[0]["id"] == running


def test_work_by_backend_shows_who_actually_did_the_translating(tmp_path):
    """'healthy' only says a backend answers. This says whether it pulled its
    weight, which is the question once a second machine is added."""
    store = Store(tmp_path / "share.db")

    for backend, cues, status in (("gpu-a", 300, "done"), ("gpu-a", 200, "done"),
                                  ("gpu-b", 100, "done"), ("gpu-b", 0, "failed")):
        job = store.enqueue(f"/m/{backend}-{cues}-{status}.mkv")
        store.claim_next()
        store.update(job, backend=backend)
        store.finish(job, status, stats={"total_cues": cues})

    work = store.work_by_backend()
    assert work["gpu-a"] == {"done": 2, "failed": 0, "cues": 500}
    assert work["gpu-b"] == {"done": 1, "failed": 1, "cues": 100}


def test_a_job_with_no_backend_recorded_is_left_out(tmp_path):
    store = Store(tmp_path / "nb.db")
    job = store.enqueue("/m/x.mkv")
    store.claim_next()
    store.finish(job, "done", stats={"total_cues": 50})
    assert store.work_by_backend() == {}
