"""Crash-isolated worker with durable claims, cancellation and GPU admission."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

from .store import DATA, JOBS, ROOT, cancelled, claim, database, now, update
from .telemetry import Monitor

STOP = threading.Event()
ACTIVE = {"job": None, "process": None, "stage": "idle"}
WORKER_ID = os.getenv("H3_WORKER_ID", "worker-a")
GROUP_DEVICES = [int(x) for x in os.getenv("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5").split(",")]
DEVICES = list(GROUP_DEVICES)
PYTHON = ROOT / ".venv/bin/python"


class Cancelled(Exception):
    pass


def heartbeat():
    while not STOP.is_set():
        try:
            with database() as con:
                con.execute("INSERT INTO workers VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                            "pid=excluded.pid, devices=excluded.devices,status=excluded.status,job_id=excluded.job_id,heartbeat=excluded.heartbeat",
                            (WORKER_ID, os.getpid(), json.dumps(DEVICES), ACTIVE["stage"], ACTIVE["job"], now()))
        except Exception:
            traceback.print_exc()
        STOP.wait(5)


def check_cancel(job_id):
    if STOP.is_set() or cancelled(job_id):
        raise Cancelled()


def tp3_eligible(row):
    cap = json.loads((DATA / "verified-capabilities.json").read_text())
    p = json.loads(row["payload"])["parameters"]
    return (cap.get("dual_tp3_verified") and row["task"] == "text_to_video"
            and [p["width"], p["height"], p["frames"]] in cap.get("dual_tp3_geometries", [])
            and p.get("profile", "quality") in cap.get("dual_tp3_profiles", []))


@contextmanager
def gpu_admission(job_id):
    locks = DATA / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    paths = [locks / "admission-intent.lock",
             Path(os.getenv("H3_SHARED_GPU_LOCK", "/home/minixm/AIserver/data/gpu0-5.lock")),
             *[locks / f"gpu-{i}.lock" for i in sorted(DEVICES)]]
    handles = []
    started = time.monotonic()
    try:
        for i, path in enumerate(paths):
            handle = path.open("a+")
            handles.append(handle)
            mode = fcntl.LOCK_SH if i <= 1 and len(DEVICES) < 6 else fcntl.LOCK_EX
            while True:
                check_cancel(job_id)
                if time.monotonic() - started > 7200:
                    raise TimeoutError("GPU admission timed out")
                try:
                    fcntl.flock(handle, mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.5)
        # Once global admission succeeds, release intent. A waiting TP6 task
        # can then reserve admission while existing TP3 work drains.
        fcntl.flock(handles[0], fcntl.LOCK_UN)
        # Reject unrelated GPU workloads that do not participate in the lock.
        usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                                        text=True, timeout=10)
        for line in usage.splitlines():
            index, used = (int(x.strip()) for x in line.split(","))
            if index in DEVICES and used > 512:
                raise RuntimeError(f"GPU {index} already has an unrelated workload ({used} MiB)")
        yield
    finally:
        for handle in reversed(handles):
            fcntl.flock(handle, fcntl.LOCK_UN)
            handle.close()


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait()
        return
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.1)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def stage(job_id, name, command, directory, timeout=2400):
    check_cancel(job_id)
    ACTIVE["stage"] = name
    started = time.perf_counter()
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES=",".join(str(i) for i in DEVICES),
               PYTORCH_ALLOC_CONF="expandable_segments:True",
               LD_LIBRARY_PATH="/usr/local/cuda-12.4/lib64:" + env.get("LD_LIBRARY_PATH", ""))
    with (directory / f"{name}.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ACTIVE["process"] = process
        try:
            while process.poll() is None:
                check_cancel(job_id)
                if time.perf_counter() - started > timeout:
                    raise TimeoutError(f"{name} exceeded {timeout} seconds")
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f"{name} process failed with exit code {process.returncode}")
        except BaseException:
            terminate(process)
            raise
        finally:
            ACTIVE["process"] = None
    return time.perf_counter() - started


def conditioning_key(payload):
    from .components import MODEL_ROOT
    identity = []
    for part in ("text_encoder", "tokenizer", "processor", "vae", "audio_vae"):
        for file in sorted((MODEL_ROOT / part).iterdir()):
            if file.is_file():
                s = file.stat()
                identity.append((str(file), s.st_size, s.st_mtime_ns))
    refs = payload["input"].get("_references", {})
    def file_sha(path):
        with Path(path).open("rb") as source:
            return hashlib.file_digest(source, "sha256").hexdigest()
    reference_hashes = {kind: [file_sha(p) for p in paths]
                        for kind, paths in refs.items()}
    # Source adapter/dependency changes invalidate conditioning independently.
    key = {"prompt": payload["input"]["prompt"], "references": reference_hashes,
           "geometry": {k: payload["parameters"][k] for k in ("width", "height", "frames")},
           "source": hashlib.sha256((ROOT / "h3service/conditioning.py").read_bytes()).hexdigest(),
           "dependencies": (ROOT / "requirements.txt").read_text(), "model_version": identity}
    return hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()


def execute(row):
    global DEVICES
    DEVICES = ([0, 1, 2, 3, 4, 5] if GROUP_DEVICES == [0, 1, 2] and not tp3_eligible(row)
               else list(GROUP_DEVICES))
    job_id = row["id"]
    payload = json.loads(row["payload"])
    p = payload["parameters"]
    directory = JOBS / job_id
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "request.json"
    request.write_text(json.dumps(payload))
    started = time.perf_counter()
    times = {}
    ACTIVE.update(job=job_id, stage="waiting_gpu")
    cache = DATA / "conditioning" / conditioning_key(payload)
    cache.mkdir(parents=True, exist_ok=True)
    cond = cache / "condition.safetensors"
    with gpu_admission(job_id), Monitor(DEVICES) as monitor:
        update(job_id, status="running")
        with (cache / "publish.lock").open("a+") as lock:
            while True:
                check_cancel(job_id)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(0.5)
            complete = cache / "complete"
            if not complete.exists() or not cond.exists():
                complete.unlink(missing_ok=True)
                temp = cache / (WORKER_ID + ".safetensors")
                times["conditioning"] = stage(job_id, "conditioning", [str(PYTHON), "-m", "h3service.conditioning",
                    "--request", str(request), "--output", str(temp)], directory)
                for suffix in (".report.json", ".references.json"):
                    target = cond.with_suffix(suffix)
                    target.unlink(missing_ok=True)
                    if temp.with_suffix(suffix).exists():
                        temp.with_suffix(suffix).replace(target)
                temp.replace(cond)
                complete.write_text("complete\n")
                cache_hit = False
            else:
                times["conditioning"] = 0
                cache_hit = True
        latent = directory / "latents.safetensors"
        times["dit"] = stage(job_id, "dit", [str(PYTHON), "-m", "torch.distributed.run", "--standalone",
            f"--nproc_per_node={len(DEVICES)}", "-m", "h3service.denoise", "--condition", str(cond),
            "--output", str(latent), "--width", str(p["width"]), "--height", str(p["height"]),
            "--frames", str(p["frames"]), "--steps", str(p["steps"]), "--seed", str(p["seed"]),
            "--profile", p.get("profile", "quality")], directory)
        output = directory / "output.mp4"
        times["decode"] = stage(job_id, "decode", [str(PYTHON), "-m", "h3service.decode",
            "--latents", str(latent), "--output", str(output)], directory)
        metrics = monitor.finish()
        (directory / "resources.json").write_text(json.dumps(metrics, indent=2))
    check_cancel(job_id)
    times["export"] = stage(job_id, "export", ["ffmpeg", "-y", "-v", "error", "-i", str(output),
                    "-frames:v", "1", "-vf", "scale=360:-2", str(directory / "thumb.jpg")], directory, timeout=60)
    report = {"engine": "diffusers", "stages_seconds": times, "total_seconds": time.perf_counter() - started,
              "conditioning_cache_hit": cache_hit, "worker_id": WORKER_ID, "devices": DEVICES,
              "profile": p.get("profile", "quality"), "actual_forwards": p["steps"], "requested_steps": p["steps"],
              "frames": p["frames"], "width": p["width"], "height": p["height"],
              "actual_duration_seconds": p["frames"] / 24,
              "resources": {k: v for k, v in metrics.items() if k != "samples"},
              "media_revision": hashlib.sha256(output.read_bytes()).hexdigest(),
              "download_url": f"/v1/jobs/{job_id}/result"}
    (directory / "performance.json").write_text(json.dumps(report, indent=2))
    check_cancel(job_id)
    with database() as con:
        changed = con.execute("UPDATE jobs SET status='succeeded',result=?,updated_at=? "
                              "WHERE id=? AND status='running' AND cancel_requested=0",
                              (json.dumps(report), now(), job_id)).rowcount
        if not changed:
            raise Cancelled()


def main():
    global DEVICES
    if DEVICES not in ([0, 1, 2, 3, 4, 5], [0, 1, 2], [3, 4, 5]):
        raise ValueError("Only audited full or three-card NVLink groups are permitted")
    if len(DEVICES) == 3:
        cap = json.loads((DATA / "verified-capabilities.json").read_text())
        if not cap.get("dual_tp3_verified"):
            raise RuntimeError("Dual TP3 has not passed admission")
    DATA.mkdir(parents=True, exist_ok=True)
    with (DATA / (WORKER_ID + ".lock")).open("a+") as singleton:
        fcntl.flock(singleton, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # systemd KillMode=control-group reaps children before restarting worker.
        with database() as con:
            con.execute("UPDATE jobs SET status='failed',error_code='WORKER_RESTARTED',error_message='Worker restarted before completion',updated_at=? "
                        "WHERE worker_id=? AND status IN ('starting','running')", (now(), WORKER_ID))
        threading.Thread(target=heartbeat, daemon=True).start()
        while not STOP.is_set():
            row = claim(WORKER_ID, tp3_eligible if GROUP_DEVICES == [3, 4, 5] else None)
            if not row:
                STOP.wait(1)
                continue
            try:
                execute(row)
            except Cancelled:
                if STOP.is_set() and not cancelled(row["id"]):
                    update(row["id"], status="failed", error_code="WORKER_STOPPED",
                           error_message="Worker stopped before completion")
                else:
                    update(row["id"], status="cancelled")
            except BaseException as exc:
                (JOBS / row["id"]).mkdir(parents=True, exist_ok=True)
                (JOBS / row["id"] / "error.log").write_text(traceback.format_exc())
                update(row["id"], status="failed", error_code="INFERENCE_FAILED",
                       error_message=f"Diffusers {ACTIVE['stage']} stage failed")
            finally:
                DEVICES = list(GROUP_DEVICES)
                ACTIVE.update(job=None, process=None, stage="idle")


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: STOP.set())
    signal.signal(signal.SIGINT, lambda *_: STOP.set())
    main()
