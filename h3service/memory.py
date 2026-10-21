"""Serialize CPU-heavy model staging across independent GPU groups."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

from .store import DATA


def prefer_inference_eviction():
    """Let the kernel evict inference before the API if host memory is exhausted."""
    Path("/proc/self/oom_score_adj").write_text("800")


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
