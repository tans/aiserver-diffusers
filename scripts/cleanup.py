#!/usr/bin/env python3
"""Retain job metadata while expiring this service's old terminal job artifacts."""
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path

from h3service.store import DATA, JOBS, database, now


def main():
    days = int(os.getenv('H3_RETENTION_DAYS', '7'))
    if days < 1:
        raise ValueError('Retention must be at least one day')
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with database() as con:
        rows = con.execute("SELECT id,result FROM jobs WHERE status IN ('succeeded','failed','cancelled') AND updated_at < ?", (cutoff,)).fetchall()
        for row in rows:
            path = JOBS / row['id']
            if not path.is_symlink() and path.is_dir() and path.resolve().parent == JOBS.resolve():
                shutil.rmtree(path)
            result = json.loads(row['result'] or '{}')
            if not result.get('expired_at'):
                result['expired_at'] = now()
                con.execute('UPDATE jobs SET result=? WHERE id=?', (json.dumps(result), row['id']))
    # Cache pruning is skipped whenever any task could be reading a cache entry.
    with database() as con:
        active = con.execute("SELECT count(*) FROM jobs WHERE status IN ('queued','starting','running')").fetchone()[0]
    cache = DATA / 'conditioning'
    if not active and cache.exists():
        for path in cache.iterdir():
            if path.is_dir() and not path.is_symlink() and datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() < cutoff:
                shutil.rmtree(path)


if __name__ == '__main__':
    main()
