"""Durable SQLite queue shared by independent API and worker processes."""
from __future__ import annotations

import json
import fcntl
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.getenv("H3_DATA_DIR", str(ROOT / "data")))
JOBS, INPUTS = DATA / "jobs", DATA / "inputs"


def now():
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def database():
    DATA.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DATA / "jobs.sqlite3", timeout=30)
    con.row_factory = sqlite3.Row
    try:
        with (DATA / "database-init.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            con.execute("PRAGMA journal_mode=WAL")
            con.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
              id TEXT PRIMARY KEY, model TEXT NOT NULL, task TEXT NOT NULL,
              payload TEXT NOT NULL, status TEXT NOT NULL, priority INTEGER NOT NULL DEFAULT 0,
              worker_id TEXT, cancel_requested INTEGER NOT NULL DEFAULT 0,
              result TEXT, error_code TEXT, error_message TEXT,
              created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, priority DESC, created_at);
            CREATE TABLE IF NOT EXISTS workers (
              id TEXT PRIMARY KEY, pid INTEGER NOT NULL, devices TEXT NOT NULL,
              status TEXT NOT NULL, job_id TEXT, heartbeat TEXT NOT NULL);
            """)
        yield con
        con.commit()
    except BaseException:
        con.rollback()
        raise
    finally:
        con.close()


def update(job_id, **values):
    allowed = {"status", "result", "error_code", "error_message", "worker_id", "cancel_requested"}
    if not values.keys() <= allowed:
        raise ValueError("Unsupported job update")
    values["updated_at"] = now()
    with database() as con:
        con.execute("UPDATE jobs SET " + ",".join(f"{k}=?" for k in values) + " WHERE id=?",
                    [*values.values(), job_id])


def claim(worker_id, eligible=None):
    with database() as con:
        con.execute("BEGIN IMMEDIATE")
        if (DATA / "drain").exists():
            return None
        rows = con.execute("SELECT * FROM jobs WHERE status='queued' AND cancel_requested=0 "
                           "ORDER BY priority DESC, created_at").fetchall()
        row = next((row for row in rows if eligible is None or eligible(row)), None)
        if row:
            con.execute("UPDATE jobs SET status='starting', worker_id=?, updated_at=? WHERE id=?",
                        (worker_id, now(), row["id"]))
        return dict(row) if row else None


def cancelled(job_id):
    with database() as con:
        row = con.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job_id,)).fetchone()
        return not row or bool(row[0])


def public_job(row):
    result = dict(row)
    request = json.loads(result.pop("payload"))
    request["input"] = {k: v for k, v in request["input"].items() if not k.startswith("_")}
    request["parameters"] = {k: v for k, v in request["parameters"].items() if k != "frames"}
    result["request"] = request
    result.pop("cancel_requested", None)
    result["result"] = json.loads(result["result"]) if result["result"] else None
    result["prompt"] = request["input"].get("prompt", "")[:300]
    return result
