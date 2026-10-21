#!/usr/bin/env python3
"""Explicit real-generation benchmark, not an API health check."""
import argparse
import concurrent.futures
import csv
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


def write_csv(rows, path):
    """Keep detailed engine measurements beside caller-observed wall latency."""
    columns = ["index", "job_id", "task", "status", "profile", "width", "height", "frames", "seed",
               "actual_forwards", "conditioning_cache_hit", "wall_seconds", "queue_wait_seconds",
               "admission_wait_seconds", "inference_seconds", "conditioning_seconds", "dit_seconds",
               "decode_seconds", "export_seconds", "peak_gpu_used_mib", "peak_process_tree_pss_bytes",
               "output_sha256"]
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            job = row["job"]
            result = job.get("result") or {}
            stages = result.get("stages_seconds", {})
            record = {key: result.get(key, "") for key in columns}
            record.update(index=row["index"], job_id=job.get("id", ""), task=job.get("task", ""),
                          status=job["status"], wall_seconds=row["wall_seconds"],
                          inference_seconds=result.get("total_seconds", ""),
                          peak_gpu_used_mib=result.get("gpu_memory", {}).get("peak_memory_used_mib", ""),
                          peak_process_tree_pss_bytes=result.get("resources", {}).get("peak_process_tree_pss_bytes", ""),
                          output_sha256=result.get("media_revision", ""))
            record.update({name + "_seconds": stages.get(name, "") for name in ("conditioning", "dit", "decode", "export")})
            writer.writerow(record)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://127.0.0.1:8190")
    p.add_argument("--count", type=int, default=1)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--profile", choices=["quality", "turbo4", "turbo8"], default="quality")
    p.add_argument("--seed", type=int, default=20261021, help="First seed; each submission increments it")
    p.add_argument("--prompt-file")
    p.add_argument("--reference-id")
    p.add_argument("--mixed", action="store_true", help="Use one reference job per ten submissions")
    p.add_argument("--reference-prompt-file")
    p.add_argument("--reference-profile", choices=["quality", "turbo4", "turbo8"], default="turbo8")
    p.add_argument("--output", default="data/benchmarks/run")
    p.add_argument("--timeout", type=int, default=3600)
    a = p.parse_args()
    if not 1 <= a.count <= 100 or not 1 <= a.concurrency <= 2:
        p.error("count must be 1–100 and concurrency 1–2")
    if not 0 <= a.seed <= 2**63 - a.count:
        p.error("seed sequence must fit unsigned 63-bit values")
    if a.mixed and not a.reference_id:
        p.error("mixed workloads require an uploaded reference-id")
    def request(path, payload=None):
        req = urllib.request.Request(a.url.rstrip("/") + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.load(response)
    ready_deadline = time.monotonic() + 30
    while True:
        try:
            if request("/ready")["status"] == "ready":
                break
        except (OSError, ValueError, KeyError):
            pass
        if time.monotonic() >= ready_deadline:
            p.error("API did not become ready before benchmark submission")
        time.sleep(0.2)
    def run(index):
        start = time.monotonic()
        is_ref = bool(a.reference_id) and (not a.mixed or index % 10 == 9)
        prompt_file = a.reference_prompt_file if is_ref and a.reference_prompt_file else a.prompt_file
        inp = {"prompt": Path(prompt_file).read_text().strip() if prompt_file else "A cheerful boy in a blue and white hoodie greets his mother in a warm living room. A continuous 3D animated shot. Gentle movement and synchronized friendly speech."}
        if is_ref:
            inp["references"] = {"images": [a.reference_id]}
        profile, steps = (a.reference_profile, {"quality": 20, "turbo4": 4, "turbo8": 8}[a.reference_profile]) if a.mixed and is_ref else (a.profile, a.steps)
        payload = {"model": "minimax-h3", "task": "ref2av" if is_ref else "text_to_video", "input": inp,
                   "parameters": {"width": a.width, "height": a.height, "duration_seconds": 5,
                                  "steps": steps, "seed": a.seed + index, "profile": profile}}
        submitted = None
        try:
            submitted = request("/v1/jobs", payload)
            while time.monotonic() - start < a.timeout:
                try:
                    row = request(submitted["poll_url"])
                except OSError:
                    # A transient API restart must not orphan an accepted job.
                    time.sleep(2)
                    continue
                if row["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                time.sleep(2)
            else:
                row = {"status": "observation_timeout", "id": submitted["job_id"]}
                try:
                    row["cancel_response"] = request(submitted["poll_url"] + ":cancel", {})
                except OSError as error:
                    row["cancel_error"] = str(error)
            return {"index": index, "wall_seconds": time.monotonic() - start, "job": row}
        except (OSError, ValueError, urllib.error.HTTPError) as e:
            return {"index": index, "wall_seconds": time.monotonic() - start,
                    "job": {"status": "observation_failed" if submitted else "submission_failed",
                            "id": submitted["job_id"] if submitted else None, "error": str(e)}}
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        rows = list(pool.map(run, range(a.count)))
    duration = time.monotonic() - started
    success = [r for r in rows if r["job"]["status"] == "succeeded"]
    timings = sorted(r["wall_seconds"] for r in success)
    percentile = lambda fraction: timings[min(len(timings) - 1, max(0, __import__("math").ceil(fraction * len(timings)) - 1))] if timings else None
    summary = {"elapsed_seconds": duration, "submitted": a.count, "succeeded": len(success),
               "success_rate": len(success) / a.count, "videos_per_hour": len(success) * 3600 / duration,
               "video_seconds_per_hour": sum(r["job"]["result"]["actual_duration_seconds"] for r in success) * 3600 / duration,
               "mean_latency_seconds": statistics.mean(timings) if timings else None,
               "p50_latency_seconds": percentile(0.5), "p95_latency_seconds": percentile(0.95),
               "quality_review": "pending human/visual review; success rate is not usable-video rate"}
    output = Path(a.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps({"configuration": vars(a), "summary": summary, "runs": rows}, indent=2))
    write_csv(rows, output.with_suffix(".csv"))
    output.with_suffix(".md").write_text("# Diffusers benchmark\n\n```json\n" + json.dumps(summary, indent=2) + "\n```\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
