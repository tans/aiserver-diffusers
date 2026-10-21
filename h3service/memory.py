"""Serialize CPU-heavy model staging across independent GPU groups."""
from contextlib import contextmanager
import fcntl

from .store import DATA


@contextmanager
def model_staging_lock(enabled=True):
    if not enabled:
        yield
        return
    path = DATA / "locks" / "model-staging.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield
