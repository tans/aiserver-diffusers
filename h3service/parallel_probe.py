"""Explicit, locked dual-TP3 real-generation feasibility benchmark."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from .store import DATA, ROOT
from .telemetry import Monitor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--height", type=int, default=320)
    p.add_argument("--stagger-load", action="store_true")
    a = p.parse_args()
    directory = Path(a.output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    processes = []
    with Path('/home/minixm/AIserver/data/gpu0-5.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        start = time.perf_counter()
        with Monitor() as monitor:
            try:
                for label, devices, node, seed in [('a','0,1,2','0',20261021),('b','3,4,5','8',20261022)]:
                    env = os.environ.copy()
                    env.update(CUDA_VISIBLE_DEVICES=devices, OMP_NUM_THREADS='4', PYTORCH_ALLOC_CONF='expandable_segments:True',
                               LD_LIBRARY_PATH='/usr/local/cuda-12.4/lib64', PYTHONUNBUFFERED='1')
                    cmd = ['numactl', '--cpunodebind=' + node, sys.executable, '-m', 'torch.distributed.run', '--standalone',
                           '--nproc_per_node=3', '-m', 'h3service.denoise', '--condition', str(Path(a.condition).resolve()),
                           '--output', str(directory / (label + '.safetensors')), '--steps', str(a.steps),
                           '--width', str(a.width), '--height', str(a.height), '--seed', str(seed)]
                    with (directory / (label + '.log')).open('wb') as log:
                        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append((label, process, env))
                    if a.stagger_load and label == 'a':
                        deadline = time.monotonic() + 300
                        while time.monotonic() < deadline:
                            if process.poll() is not None:
                                raise RuntimeError('Worker A failed during staged loading')
                            loaded = (directory / 'a.log').read_text(errors='replace').count('"event": "loaded"')
                            if loaded >= 3:
                                break
                            time.sleep(0.5)
                        else:
                            raise TimeoutError('Worker A load did not finish')
                deadline = time.monotonic() + 1800
                while any(process.poll() is None for label, process, env in processes):
                    if any(process.poll() not in (None, 0) for label, process, env in processes):
                        raise RuntimeError('A dual TP3 rank group failed')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Dual TP3 exceeded its feasibility timeout')
                    time.sleep(0.5)
                codes = {label: process.returncode for label, process, env in processes}
                denoise_elapsed = time.perf_counter() - start
                if any(codes.values()):
                    raise RuntimeError('Dual TP3 process failures: ' + str(codes))
                for label, process, env in list(processes):
                    with (directory / (label + '.decode.log')).open('wb') as log:
                        decoded = subprocess.Popen([sys.executable, '-m', 'h3service.decode', '--latents',
                            str(directory / (label + '.safetensors')), '--output', str(directory / (label + '.mp4'))],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    processes.append((label + '-decode', decoded, env))
                for label, process, env in processes[2:]:
                    codes[label] = process.wait(timeout=1200)
                if any(codes.values()):
                    raise RuntimeError('Dual decode failed: ' + str(codes))
                report = {'status': 'succeeded', 'denoise_pair_seconds': denoise_elapsed,
                          'total_pair_seconds': time.perf_counter() - start,
                          'conditioning': 'precomputed by native Diffusers; cache-warm benchmark',
                          'geometry': [a.width, a.height, 124], 'actual_forwards_per_task': a.steps,
                          'ranks': {label: json.loads((directory / (label + '.report.json')).read_text()) for label in ('a','b')}}
            except BaseException as error:
                for label, process, env in processes:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                report = {'status': 'failed', 'error': str(error), 'elapsed_seconds': time.perf_counter() - start}
                raise
            finally:
                for label, process, env in processes:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                report['resources'] = monitor.finish()
                (directory / 'summary.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    def stopped(*_):
        raise KeyboardInterrupt("Benchmark stopped")
    signal.signal(signal.SIGTERM, stopped)
    main()
