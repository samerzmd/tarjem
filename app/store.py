"""SQLite-backed job log and glossary cache.

The glossary cache is what makes a series feel coherent: the brief built while
translating S01E01 is reused for every later episode, so character names never
drift halfway through a season.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
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
    bz_kind      TEXT DEFAULT '',
    bz_series_id INTEGER DEFAULT 0,
    bz_item_id   INTEGER DEFAULT 0,
    created_at   REAL NOT NULL,
    started_at   REAL,
    finished_at  REAL
);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS jobs_video  ON jobs(video);

CREATE TABLE IF NOT EXISTS glossaries (
    key        TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""

QUEUED, RUNNING, DONE, FAILED, SKIPPED = "queued", "running", "done", "failed", "skipped"
ACTIVE = (QUEUED, RUNNING)


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        with self._lock:
            self._db.executescript(SCHEMA)
            # A restart mid-job would otherwise leave a permanent "running" row.
            self._db.execute(
                "UPDATE jobs SET status=?, error=? WHERE status=?",
                (FAILED, "interrupted by restart", RUNNING),
            )
            self._db.commit()

    # -- jobs -------------------------------------------------------------

    def enqueue(
        self,
        video: str,
        title: str = "",
        series_key: str = "",
        trigger: str = "",
        bz_kind: str = "",
        bz_series_id: int = 0,
        bz_item_id: int = 0,
    ) -> int | None:
        """Queue a video. Returns None if it is already queued or running."""
        with self._lock:
            existing = self._db.execute(
                "SELECT id FROM jobs WHERE video=? AND status IN (?,?)", (video, QUEUED, RUNNING)
            ).fetchone()
            if existing:
                return None
            cur = self._db.execute(
                "INSERT INTO jobs (video,title,series_key,status,trigger,"
                "bz_kind,bz_series_id,bz_item_id,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
                (video, title, series_key, QUEUED, trigger,
                 bz_kind, bz_series_id, bz_item_id, time.time()),
            )
            self._db.commit()
            return int(cur.lastrowid)

    def claim_next(self) -> sqlite3.Row | None:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM jobs WHERE status=? ORDER BY id LIMIT 1", (QUEUED,)
            ).fetchone()
            if not row:
                return None
            self._db.execute(
                "UPDATE jobs SET status=?, started_at=? WHERE id=?", (RUNNING, time.time(), row["id"])
            )
            self._db.commit()
            return self._db.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()

    def update(self, job_id: int, **fields: Any) -> None:
        if not fields:
            return
        for key in ("stats", "usage"):
            if key in fields and not isinstance(fields[key], str):
                fields[key] = json.dumps(fields[key])
        assignments = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._db.execute(
                f"UPDATE jobs SET {assignments} WHERE id=?", (*fields.values(), job_id)
            )
            self._db.commit()

    def finish(self, job_id: int, status: str, **fields: Any) -> None:
        self.update(job_id, status=status, finished_at=time.time(), **fields)

    def get(self, job_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row(row) if row else None

    def recent(self, limit: int = 50, status: str | None = None) -> list[dict]:
        query = "SELECT * FROM jobs"
        params: tuple = ()
        if status:
            query += " WHERE status=?"
            params = (status,)
        query += " ORDER BY id DESC LIMIT ?"
        with self._lock:
            rows = self._db.execute(query, (*params, limit)).fetchall()
        return [_row(r) for r in rows]

    def counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._db.execute("SELECT status, COUNT(*) c FROM jobs GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}

    def succeeded_for(self, video: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM jobs WHERE video=? AND status=? LIMIT 1", (video, DONE)
            ).fetchone()
        return row is not None

    def is_pending(self, video: str) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM jobs WHERE video=? AND status IN (?,?) LIMIT 1",
                (video, QUEUED, RUNNING),
            ).fetchone()
        return row is not None

    def requeue(self, job_id: int) -> bool:
        with self._lock:
            row = self._db.execute(
                "SELECT video,title,series_key,bz_kind,bz_series_id,bz_item_id "
                "FROM jobs WHERE id=?", (job_id,)
            ).fetchone()
        if not row:
            return False
        return self.enqueue(
            row["video"], row["title"], row["series_key"], "retry",
            row["bz_kind"], row["bz_series_id"], row["bz_item_id"],
        ) is not None

    # -- glossaries -------------------------------------------------------

    def get_glossary(self, key: str) -> dict | None:
        if not key:
            return None
        with self._lock:
            row = self._db.execute("SELECT payload FROM glossaries WHERE key=?", (key,)).fetchone()
        return json.loads(row["payload"]) if row else None

    def put_glossary(self, key: str, payload: dict) -> None:
        if not key:
            return
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO glossaries (key,payload,created_at) VALUES (?,?,?)",
                (key, json.dumps(payload, ensure_ascii=False), time.time()),
            )
            self._db.commit()

    def drop_glossary(self, key: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM glossaries WHERE key=?", (key,))
            self._db.commit()

    def glossaries(self) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT key, payload, created_at FROM glossaries ORDER BY created_at DESC"
            ).fetchall()
        out = []
        for row in rows:
            payload = json.loads(row["payload"])
            out.append({
                "key": row["key"],
                "created_at": row["created_at"],
                "terms": len(payload.get("terms", [])),
                "summary": payload.get("summary", ""),
            })
        return out


def _row(row: sqlite3.Row) -> dict:
    data = dict(row)
    for key in ("stats", "usage"):
        try:
            data[key] = json.loads(data.get(key) or "{}")
        except json.JSONDecodeError:
            data[key] = {}
    return data
