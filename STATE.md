# Working state — 2026-10-21

Issue truth: https://github.com/tans/aiserver-diffusers/issues/1 (open).

## Current deployment

- Independent server directory `/home/minixm/AIserver-diffusers`, own venv and SQLite/data. Public API http://minixm.minapp.xin:8190, docs `/docs`; original service remains `/home/minixm/AIserver` on public 8189.
- API, worker A, worker B and the separate reverse tunnel are enabled. User linger is enabled. `scripts.set_mode dual-tp3` was exercised successfully; `/ready` reports A [0,1,2], B [3,4,5]. Text quality/Turbo8 uses dual TP3; reference Turbo4 obtains exclusive TP6 through A.
- Public geometry remains 512×320/124 frames. Current admission: text quality (20+ actual forwards) and text Turbo8 (eight). Reference admission is withheld after the visual-quality failure recorded in issue #2.
- Daily cleanup timer is enabled; next scheduled run 2026-10-22 00:00 CST. It expires this service's seven-day terminal job artifacts/cache and preserves metadata; inputs are not automatically pruned. No old-service data is touched.
- Inference memory slice: MemoryHigh 22 GiB, MemoryMax 24 GiB, MemorySwapMax 1 GiB, inference OOM eviction priority 800. Subsequent max/oom/oom_kill counters remain zero; host historical oom_kill remains 15.

## Passed evidence

- Native Diffusers TP6 and TP3, original video/audio VAEs and verified MP4 output. Two different-prompt cold dual quality jobs completed twenty actual forwards each after the FP32 row-output fix, pair wall 276.39 s.
- Turbo8 text and Turbo4 reference fixed-seed complete outputs. Public browser submission and result download/hash passed.
- Ten consecutive cached-prompt Turbo8 dual-TP3 tasks: 10/10 succeeded, five per group. Batch wall 878.44 s includes initial legacy GPU waiting; active window 586.61 s gives 61.37 videos/hour. Mean inference 113.15 s.
- Two serial TP6 cached-prompt Turbo8 tasks: 2/2 succeeded, mean inference 105.34 s, active-window rate 34.16/hour. Observed short-window ratio about 1.80x; sample sizes differ and no hour-long capacity claim is made.
- Upload, queued cancellation, nearest-frame compatibility (5.17 seconds maps to 124), isolated storage-admission 507.
- SIGKILL during actual denoising: API stayed available; restarted worker marked interrupted work failed/WORKER_RESTARTED. Subsequent actual-forward user cancellation reached cancelled; GPU process groups were reaped.

## Active acceptance

- The first public mixed workload was stopped after a reference Turbo4 seed 20261030 showed block artifacts at full size. The accepted text jobs were allowed to finish. This is an aborted diagnostic batch, not a successful twenty-task acceptance. Raw rows are in reports/evidence/public-mixed20-turbo4-aborted.json.
- Public reference admission is withheld; text dual TP3 remains available. Exact reference conditioning comparison with the earlier CLI run is bit-identical. An isolated reference-quality candidate on 8192 will compare the same seed with Turbo8, then other configurations as needed.
- Tracking quality issue: https://github.com/tans/aiserver-diffusers/issues/2. The previous higher-resolution waiting loop was stopped; it will be scheduled again after visual-quality correction and the repeated mixed workload.

## Remaining boundaries

- Complete active workloads, collect JSON/CSV/Markdown, inspect higher-resolution output and reconcile final server Git SHA.
- No real host reboot has been performed. Worker restart and enabled units do not prove host reboot recovery. A reboot affects the original service and other active jobs and needs an agreed maintenance boundary.
- INT8, cross-task DiT residency and DiT group offload are unimplemented candidates. Native sm70 attention passed one historical complete seed and remains an explicit CLI candidate, not the public default. Exact H2D/compute/idle decomposition and human lip-sync ratings are unmeasured.
- The initial simultaneous-load attempt caused kernel-confirmed host memory-policy OOM that killed original and new APIs before systemd recovery. The later bounded configuration has not repeated it; do not describe the initial API as continuously available.

Every source/doc change is committed and pushed on main. No test code is added. Real inference benchmarks use GPU locks and yield to recent original-service work. The new venv reuses the original POWER9 binary packages read-only; package writes are isolated but the binaries remain a dependency.
