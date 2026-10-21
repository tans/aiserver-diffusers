# AIserver Diffusers

Independent MiniMax H3 audio/video inference service for IBM POWER9 and six V100 16 GB GPUs.

- Public API: **http://minixm.minapp.xin:8190**
- Interactive API documentation: **http://minixm.minapp.xin:8190/docs**
- Repository: https://github.com/tans/aiserver-diffusers
- Implementation and acceptance issue: https://github.com/tans/aiserver-diffusers/issues/1

The initial verified generation geometry is **512×320, 124 frames, 24 fps**, with native Diffusers video and stereo-audio VAEs. `quality` uses at least 20 actual transformer forwards. `turbo8` is admitted for text-to-video after two fixed-seed runs. Larger geometries and image-reference profiles are gated separately. This is an experimental service while sustained-load acceptance continues.

## Isolation

Server directory: `/home/minixm/AIserver-diffusers`, including its own `.venv`, SQLite queue, inputs and outputs. The original LightX2V service remains in `/home/minixm/AIserver`, public port 8189. No original code or environment package is changed.

Model weights are opened read-only from `/srv/models/lightx2v/h3`. POWER9 torch/CUDA binary packages are reused read-only through a `.pth` path; Python package writes go only into the new venv. This is not a copied, fully independent binary stack. The new public reverse tunnel is a separate root-managed unit and leaves the old tunnel unit unchanged.

## API example

The existing `model / task / input / parameters / options` envelope and `/v1/files`, `/v1/jobs`, status, cancel, result and thumbnail routes are retained. Duration uses the original nearest legal frame-slot rule: `5.17` seconds maps to 124 frames (actual duration 5.1667 seconds).

```sh
curl -sS http://minixm.minapp.xin:8190/v1/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimax-h3",
    "task": "text_to_video",
    "input": {"prompt": "A cheerful boy greets his mother in a warm living room. Natural gentle movement and synchronized friendly speech."},
    "parameters": {"width": 512, "height": 320, "duration_seconds": 5.17, "profile": "turbo8", "steps": 8, "seed": 20261021}
  }'
```

Use the returned `poll_url` until `succeeded`, then download `/v1/jobs/<job_id>/result`. Cancel with `POST /v1/jobs/<job_id>:cancel`. `GET /ready` reports admitted capabilities and live worker heartbeats; `/health` only establishes API availability.

For the base profile, set `profile` to `quality` and `steps` to 20. Unknown models, arbitrary LoRAs, unverified profiles, video/audio references and unverified geometry are rejected. Image generation and super resolution are outside this new service's initial scope.

## Runtime and operations

Run `scripts/bootstrap.sh` on the audited POWER9 host. Dependency pins are in `requirements.txt`; Diffusers 0.41.0 corresponds to tag commit `086bf9578c0f4acbc66e48cd1e7cc26befd9e10f`.

The default enabled API and worker units are supplied under `deploy/`. The API listens on loopback 8190. The separate tunnel forwards cloud port 8190. Worker A defaults to TP6; Worker B remains disabled until cold dual-worker admission passes. All generation coordinates with the old service's existing all-GPU lock, so queue waiting must be separated from actual inference timing.

A worker restart explicitly terminates its process groups and fails interrupted work; queued jobs persist in SQLite. Running cancellation reaps the stage process group. Low storage rejects new work. The optional cleanup timer only expires this service's terminal artifacts and retains task metadata.

## Evidence

- [Measured baseline](reports/measured-baseline.md), [CSV](reports/measured-baseline.csv), [raw JSON](reports/evidence/)
- [Benchmark method and current results](docs/benchmark.md)
- [Environment audit](docs/environment-audit.md), [compatibility](docs/compatibility.md), [architecture](docs/architecture.md)
- [Known limitations](docs/known-limitations.md), [working state](STATE.md), [original execution specification](docs/execution-spec-v1.md)

A cache-warm dual TP3 pair completed two verified MP4s in 188.75 seconds, including staged weight loading. This is one pair with precomputed conditioning, not a proven sustained or cold end-to-end rate. The public default remains TP6 while those separate gates are evaluated.

Real generation benchmark:

```sh
.venv/bin/python scripts/benchmark.py --count 2 --concurrency 1 \
  --profile turbo8 --steps 8 --output data/benchmarks/turbo8-cold-warm
```

Benchmarks produce JSON, CSV and Markdown. A completed video is not automatically a visually or audibly usable video; quality review is recorded separately. Models, generated media and credentials are never committed.
