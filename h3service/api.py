"""Video API compatible with AIserver's public job/file/result contract."""
from __future__ import annotations

import json
import math
import os
import re
import secrets
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .store import DATA, INPUTS, JOBS, ROOT, database, now, public_job

app = FastAPI(title="AIserver Diffusers", version="0.1.0")


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=128)
    task: str = Field(min_length=1, max_length=64)
    input: dict[str, Any] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    options: dict[str, Any] = Field(default_factory=dict)


def reject(code, message, status=400):
    raise HTTPException(status, {"code": code, "message": message})


def capabilities():
    path = DATA / "verified-capabilities.json"
    return json.loads(path.read_text()) if path.exists() else {"enabled": False, "tasks": [], "profiles": []}


def storage_admission():
    DATA.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(DATA).free < int(os.getenv("H3_MIN_FREE_BYTES", str(12 * 1024**3))):
        reject("INSUFFICIENT_STORAGE", "Insufficient free space for a new video job", 507)


@app.get("/health")
def health():
    return {"status": "ok", "engine": "diffusers", "service": "aiserver-diffusers"}


@app.get("/ready")
def ready():
    cap = capabilities()
    with database() as con:
        workers = [dict(r) for r in con.execute("SELECT * FROM workers")]
    alive = [w for w in workers if (datetime.now(timezone.utc) - datetime.fromisoformat(w["heartbeat"])).total_seconds() < 30]
    return {"status": "ready" if cap.get("enabled") and alive else "not_ready", "engine": "diffusers",
            "capabilities": cap, "workers": alive}


@app.get("/v1/models")
def models():
    cap = capabilities()
    return {"data": [{"id": "minimax-h3", "name": "MiniMax H3 Diffusers", "engine": "diffusers",
                      "status": "experimental" if cap.get("enabled") else "validating",
                      "tasks": cap["tasks"], "profiles": cap["profiles"], "output_format": "mp4",
                      "weight_format": "bf16", "compute_profile": "fp16_with_fp32_sensitive_layers"}]}


@app.post("/v1/files", status_code=201)
async def upload(file: UploadFile = File(...)):
    storage_admission()
    INPUTS.mkdir(parents=True, exist_ok=True)
    file_id = secrets.token_hex(16)
    target, size = INPUTS / file_id, 0
    try:
        with target.open("xb") as out:
            while chunk := await file.read(1024**2):
                size += len(chunk)
                if size > 2 * 1024**3:
                    reject("INPUT_TOO_LARGE", "Maximum upload is 2 GiB", 413)
                if shutil.disk_usage(DATA).free < 2 * 1024**3:
                    reject("INSUFFICIENT_STORAGE", "Upload storage is full", 507)
                out.write(chunk)
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"file_id": file_id, "filename": Path(file.filename or "input").name, "size_bytes": size}


@app.post("/v1/jobs", status_code=202)
def submit(req: JobRequest):
    cap = capabilities()
    if req.model != "minimax-h3":
        reject("UNKNOWN_MODEL", "Only the Diffusers minimax-h3 model is available")
    task = {"t2v": "text_to_video", "image_to_video": "ref2av", "i2v": "ref2av"}.get(req.task, req.task)
    if task not in {"text_to_video", "ref2av"}:
        reject("UNSUPPORTED_TASK", "Unsupported task")
    if not cap.get("enabled") or task not in cap["tasks"]:
        reject("MODEL_NOT_READY", "This Diffusers workflow is not yet verified", 503)
    if any(str(k).startswith("_") for k in req.input):
        reject("INVALID_INPUT", "Reserved input fields are not public")
    prompt = req.input.get("prompt")
    if not isinstance(prompt, str) or not 1 <= len(prompt.strip()) <= 2000:
        reject("INVALID_PROMPT", "prompt must contain 1–2000 characters")
    p = dict(req.parameters)
    if "frames" in p:
        reject("FRAMES_NOT_PUBLIC", "Use duration_seconds")
    duration = p.get("duration_seconds")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not math.isfinite(duration) or not 5 <= duration <= 15:
        reject("UNSUPPORTED_DURATION", "duration_seconds must be between 5 and 15")
    frames = min(range(124, 363, 17), key=lambda n: abs(n - duration * 24))
    width, height = p.get("width", 512), p.get("height", 320)
    if any(type(x) is not int or x < 32 or x % 32 for x in (width, height)) or width * height > 983040:
        reject("INVALID_DIMENSIONS", "Dimensions must be positive multiples of 32; area <= 983040")
    if (width, height, frames) not in [tuple(x) for x in cap.get("geometries", [])]:
        reject("UNVERIFIED_GEOMETRY", "Requested geometry has not passed Diffusers admission", 422)
    steps = p.get("steps", 20)
    if type(steps) is not int or not 4 <= steps <= 64:
        reject("INVALID_STEPS", "steps must be an integer between 4 and 64")
    if p.get("loras"):
        reject("UNVERIFIED_LORA", "LoRA support has not passed verification")
    if steps < 20:
        reject("UNVERIFIED_PROFILE", "Few-step generation requires a verified Turbo profile")
    seed = p.get("seed", secrets.randbelow(2**31))
    if type(seed) is not int or not 0 <= seed < 2**63:
        reject("INVALID_SEED", "seed must be a nonnegative 63-bit integer")
    inp = dict(req.input)
    if task == "ref2av":
        refs = inp.get("references")
        if not isinstance(refs, dict):
            reject("REFERENCE_REQUIRED", "ref2av requires references")
        resolved = {}
        for kind, maximum in (("images", 9), ("videos", 3), ("audios", 3)):
            ids = refs.get(kind, [])
            if not isinstance(ids, list) or len(ids) > maximum:
                reject("REFERENCE_LIMIT", "Invalid reference count")
            resolved[kind] = []
            for fid in ids:
                if not isinstance(fid, str) or not re.fullmatch("[0-9a-f]{32}", fid) or not (INPUTS / fid).is_file():
                    reject("INPUT_FILE_NOT_FOUND", "Unknown uploaded reference")
                resolved[kind].append(str(INPUTS / fid))
        if not resolved["images"] and not resolved["videos"]:
            reject("REFERENCE_REQUIRED", "At least one image or video reference is required")
        if resolved["videos"] or resolved["audios"]:
            reject("UNVERIFIED_REFERENCE", "Video/audio reference workflows are not yet verified")
        inp["_references"] = resolved
    priority = req.options.get("priority", 0)
    if type(priority) is not int or not 0 <= priority <= 10:
        reject("INVALID_PRIORITY", "priority must be 0–10")
    storage_admission()
    p.update(width=width, height=height, frames=frames, steps=steps, seed=seed)
    payload = {"model": req.model, "task": task, "input": inp, "parameters": p, "options": req.options}
    job_id, stamp = "job_" + uuid.uuid4().hex, now()
    with database() as con:
        con.execute("BEGIN IMMEDIATE")
        waiting = con.execute("SELECT count(*) FROM jobs WHERE status='queued'").fetchone()[0]
        if waiting >= int(os.getenv("H3_MAX_QUEUED", "32")):
            reject("QUEUE_FULL", "Task queue is full", 429)
        con.execute("INSERT INTO jobs(id,model,task,payload,status,priority,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                    (job_id, req.model, task, json.dumps(payload), "queued", priority, stamp, stamp))
    return {"job_id": job_id, "status": "queued", "model": req.model, "task": task,
            "created_at": stamp, "poll_url": f"/v1/jobs/{job_id}"}


def find_job(job_id):
    with database() as con:
        row = con.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if not row:
        reject("JOB_NOT_FOUND", job_id, 404)
    return row


@app.get("/v1/jobs")
def list_jobs(limit: int = 50, offset: int = 0, model: str = "", status: str = "", task: str = ""):
    limit, offset = min(200, max(1, limit)), max(0, offset)
    filters = [(k, v) for k, v in (("model", model), ("status", status), ("task", task)) if v]
    clause = " WHERE " + " AND ".join(k + "=?" for k, v in filters) if filters else ""
    args = [v for k, v in filters]
    with database() as con:
        total = con.execute("SELECT count(*) FROM jobs" + clause, args).fetchone()[0]
        rows = con.execute("SELECT * FROM jobs" + clause + " ORDER BY created_at DESC LIMIT ? OFFSET ?", [*args, limit, offset]).fetchall()
    return {"total": total, "limit": limit, "offset": offset, "items": [public_job(r) for r in rows]}


@app.get("/v1/jobs/{job_id}")
def job(job_id: str):
    return public_job(find_job(job_id))


@app.post("/v1/jobs/{job_id}:cancel")
def cancel(job_id: str):
    row = find_job(job_id)
    if row["status"] not in {"succeeded", "failed", "cancelled"}:
        with database() as con:
            con.execute("UPDATE jobs SET cancel_requested=1, status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END, updated_at=? WHERE id=? AND status IN ('queued','starting','running')",
                        (now(), job_id))
    return {"job_id": job_id, "status": find_job(job_id)["status"]}


@app.get("/v1/jobs/{job_id}/result")
def result(job_id: str):
    if find_job(job_id)["status"] != "succeeded":
        reject("RESULT_NOT_READY", "Task has not succeeded", 409)
    path = JOBS / job_id / "output.mp4"
    if not path.is_file():
        reject("RESULT_NOT_FOUND", job_id, 404)
    return FileResponse(path, media_type="video/mp4", filename=job_id + ".mp4")


@app.get("/v1/jobs/{job_id}/thumbnail")
def thumbnail(job_id: str):
    if find_job(job_id)["status"] != "succeeded":
        reject("RESULT_NOT_READY", "Task has not succeeded", 409)
    path = JOBS / job_id / "thumb.jpg"
    if not path.is_file():
        reject("RESULT_NOT_FOUND", job_id, 404)
    return FileResponse(path, media_type="image/jpeg")
