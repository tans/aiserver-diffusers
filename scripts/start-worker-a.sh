#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
export H3_WORKER_ID=worker-a
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5}"
exec .venv/bin/python -m h3service.worker
