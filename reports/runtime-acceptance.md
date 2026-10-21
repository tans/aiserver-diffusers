# Running cancellation and worker crash acceptance

2026-10-21. Candidate API 8192, GPU group 0–2, full twenty-forward quality request using cached conditioning. Both probes waited for an actual denoiser `forward` log event.

1. SIGKILL of the entire candidate worker cgroup: API `/health` stayed available. Restarted worker marked interrupted work `failed / WORKER_RESTARTED`. GPU process query was empty after restart and before the next job.
2. A new job subsequently entered actual denoising. `POST /v1/jobs/<id>:cancel` reached `cancelled`, and the stage process group was reaped. GPU process query was empty afterwards. Failure/cancellation telemetry is retained in each job directory.

Raw job IDs, terminal states, API response and GPU process observations: `evidence/runtime-acceptance.json`. These are process crash/restart probes; no host reboot has occurred. A SIGKILL is not an induced CUDA or cgroup OOM, and this evidence does not claim a tested OOM boundary.
